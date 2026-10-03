"""Contract, resume and transport tests without model dependencies or a GPU."""

import copy
import io
import json
import sqlite3
import urllib.error

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from prisma import RemotePredictor
from prisma.predictors import RemotePredictionError


class Endpoint:
    def __init__(self):
        self.fingerprint = "a" * 64
        self.calls = []
        self.fail_on = None
        self.failure = RemotePredictionError("service stopped")
        self.corrupt = None
        self.outputs = {
            "value": {"type": "float", "shape": [], "unit": "eV"},
            "tensor": {"type": "float", "shape": [2, 2], "unit": "model_native"},
            "flag": {"type": "bool"},
            "formula": {"type": "string"},
        }

    def request(self, route, payload=None):
        if route == "/info":
            return {
                "protocol_version": 1,
                "input_schema": "pymatgen.structure.mson.v1",
                "model_fingerprint": self.fingerprint,
                "model": {
                    "name": "fake",
                    "revision": "v1",
                    "checkpoint_sha256": "b" * 64,
                },
                "outputs": copy.deepcopy(self.outputs),
            }
        assert route == "/predict"
        self.calls.append(copy.deepcopy(payload))
        if len(self.calls) == self.fail_on:
            raise self.failure
        predictions = []
        for item in payload["structures"]:
            structure = Structure.from_dict(item)
            a = structure.lattice.a
            predictions.append(
                {
                    "value": a,
                    "tensor": [[a, 0.0], [0.0, a]],
                    "flag": True,
                    "formula": structure.formula,
                }
            )
        result = {"model_fingerprint": self.fingerprint, "predictions": predictions}
        if self.corrupt:
            self.corrupt(result)
        return result


@pytest.fixture
def structures():
    return [Structure(Lattice.cubic(a), ["Si"], [[0, 0, 0]]) for a in (3, 4, 5, 6, 7)]


@pytest.fixture
def endpoint():
    return Endpoint()


@pytest.fixture
def client_factory(tmp_path, monkeypatch, endpoint):
    def create(**kwargs):
        options = {
            "chunk_size": 2,
            "progress": False,
            "cache_path": tmp_path / "cache.sqlite",
        }
        options.update(kwargs)
        client = RemotePredictor("http://test", **options)
        monkeypatch.setattr(client, "_request", endpoint.request)
        return client

    return create


def cached_rows(client):
    with sqlite3.connect(client.cache_path) as conn:
        return conn.execute("SELECT count(*) FROM predictions_v1").fetchone()[0]


def test_chunking_order_duplicates_full_cache_and_selection(
    client_factory, structures, endpoint
):
    client = client_factory()
    batch = [structures[1], structures[0], structures[1], structures[2], structures[0]]
    assert client.predict(batch, select="value").tolist() == [4, 3, 4, 5, 3]
    assert [len(call["structures"]) for call in endpoint.calls] == [2, 1]
    full = client_factory().predict(batch)
    assert len(endpoint.calls) == 2
    assert full[0]["tensor"] == [[4, 0], [0, 4]]
    full[0]["tensor"][0][0] = -1
    assert full[2]["tensor"][0][0] == 4  # duplicate rows must not alias mutable objects
    assert client.predict(batch, select="tensor").shape == (5, 2, 2)
    assert client.predict(batch, select="flag").dtype == np.bool_
    assert client.predict(batch, select="formula").tolist() == ["Si1"] * 5
    assert cached_rows(client) == 3


@pytest.mark.parametrize(
    "failure", [RemotePredictionError("offline"), KeyboardInterrupt()]
)
def test_new_client_resumes_completed_chunks(
    client_factory, structures, endpoint, failure
):
    client = client_factory()
    endpoint.fail_on = 2
    endpoint.failure = failure
    with pytest.raises(type(failure)) as error:
        client.predict(structures, select="value")
    assert cached_rows(client) == 2
    assert "2/5" in error.value.__notes__[0]
    endpoint.fail_on = None
    assert client_factory().predict(structures, select="value").tolist() == [
        3,
        4,
        5,
        6,
        7,
    ]
    assert [len(call["structures"]) for call in endpoint.calls] == [2, 2, 2, 1]


def test_new_fingerprint_does_not_reuse_previous_model(
    client_factory, structures, endpoint
):
    client = client_factory()
    client.predict(structures[:1])
    endpoint.fingerprint = "c" * 64
    client.predict(structures[:1])
    assert len(endpoint.calls) == 2
    assert cached_rows(client) == 2


def test_refresh_and_cache_disabled(client_factory, structures, endpoint, tmp_path):
    client = client_factory()
    client.predict(structures[:1])
    client.predict(structures[:1], refresh=True)
    assert len(endpoint.calls) == 2
    assert cached_rows(client) == 1
    other = client_factory(cache=False, cache_path=tmp_path / "disabled.sqlite")
    other.predict(structures[:1])
    other.predict(structures[:1])
    assert len(endpoint.calls) == 4
    assert not other.cache_path.exists()


def test_empty_results_follow_selected_shape(client_factory, endpoint):
    client = client_factory()
    assert client.predict([]) == []
    assert client.predict([], select="value").shape == (0,)
    assert client.predict([], select="tensor").shape == (0, 2, 2)
    assert endpoint.calls == []


def test_invalid_selection_fails_before_prediction(
    client_factory, structures, endpoint
):
    with pytest.raises(ValueError, match="available outputs"):
        client_factory().predict(structures, select="unknown")
    assert endpoint.calls == []


def test_numpy_site_properties_use_pymatgen_encoder(client_factory, endpoint):
    structure = Structure(
        Lattice.cubic(4),
        ["Si"],
        [[0, 0, 0]],
        site_properties={"vector": [np.array([1.0, 2.0, 3.0])], "index": [np.int64(7)]},
    )
    assert client_factory().predict([structure], select="value").tolist() == [4]
    json.dumps(endpoint.calls[0], allow_nan=False)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda result: result.update(model_fingerprint="changed"),
        lambda result: result.update(predictions=[]),
        lambda result: result["predictions"][0].update(value=float("nan")),
        lambda result: result["predictions"][0].update(value=True),
        lambda result: result["predictions"][0].update(tensor=[[1.0]]),
        lambda result: result["predictions"][0].pop("value"),
    ],
)
def test_bad_responses_never_enter_cache(client_factory, structures, endpoint, corrupt):
    client = client_factory()
    endpoint.corrupt = corrupt
    with pytest.raises(RemotePredictionError):
        client.predict(structures[:1])
    assert cached_rows(client) == 0
    endpoint.corrupt = None
    assert client.predict(structures[:1], select="value").tolist() == [3]
    assert len(endpoint.calls) == 2


def test_changed_fingerprint_mid_call_preserves_only_prior_chunks(
    client_factory, structures, endpoint
):
    client = client_factory()
    original = endpoint.request

    def changing(route, payload=None):
        if route == "/predict" and endpoint.calls:
            endpoint.fingerprint = "changed"
        return original(route, payload)

    client._request = changing
    with pytest.raises(RemotePredictionError, match="fingerprint changed"):
        client.predict(structures)
    assert cached_rows(client) == 2


@pytest.mark.parametrize(
    "status,retries", [(503, 1), (502, 1), (504, 1), (400, 0), (500, 0)]
)
def test_http_retry_policy(monkeypatch, status, retries):
    calls = []

    def failing(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(
            request.full_url,
            status,
            "failed",
            {},
            io.BytesIO(b'{"detail":"test failure"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", failing)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    with pytest.raises(RemotePredictionError, match=f"HTTP {status}"):
        RemotePredictor("http://test", cache=False).info()
    assert len(calls) == 1 + retries


def test_transport_retry_and_json_validation(monkeypatch):
    calls = []

    def delayed(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            raise TimeoutError("slow")
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr("urllib.request.urlopen", delayed)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    client = RemotePredictor("http://test", cache=False)
    assert client._request("/info") == {"ok": True}
    assert len(calls) == 2
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b"not JSON")
    )
    with pytest.raises(RemotePredictionError, match="not valid JSON"):
        client._request("/info")


def test_invalid_protocol_and_input(client_factory, structures, endpoint):
    client = client_factory()
    with pytest.raises(TypeError, match="sequence"):
        client.predict(structures[0])
    with pytest.raises(TypeError, match=r"structures\[0\]"):
        client.predict([{}])
    endpoint.outputs["tensor"]["shape"] = [-1]
    with pytest.raises(RemotePredictionError, match="shape"):
        client.info()
