"""Model-independent, resumable HTTP inference for Pymatgen structures."""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from tqdm.auto import tqdm

from ._cache import PredictionCache
from ._protocol import (
    RemotePredictionError,
    validate_info,
    validate_prediction,
    validate_response,
)


class RemotePredictor:
    """Predict properties using a protocol-v1 container at an explicit URL.

    No model dependencies are loaded locally. Complete responses are cached on
    disk by model fingerprint and structure content; repeating a failed call
    reuses completed chunks.

    Args:
        url: Service URL, for example http://127.0.0.1:18000.
        chunk_size: Maximum number of structures in one HTTP request.
        timeout: Socket timeout in seconds, including waiting for a response.
        retries: Extra attempts for transient transport errors and HTTP 502/503/504.
        cache: Persist complete predictions in SQLite.
        cache_path: Override the default XDG cache path.
        progress: Display progress, including structures loaded from cache.

    Example:
        >>> predictor = RemotePredictor("http://127.0.0.1:18000")
        >>> values = predictor.predict(structures, select="bandgap_pbe")
    """

    def __init__(
        self,
        url: str,
        *,
        chunk_size: int = 128,
        timeout: float = 300.0,
        retries: int = 1,
        cache: bool = True,
        cache_path: str | Path | None = None,
        progress: bool = True,
    ):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute http:// or https:// URL")
        if parsed.query or parsed.fragment:
            raise ValueError("url must not contain a query or fragment")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if timeout <= 0 or not np.isfinite(timeout):
            raise ValueError("timeout must be finite and positive")
        if type(retries) is not int or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        self.url = url.rstrip("/")
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.retries = retries
        self.cache = cache
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        self.cache_path = (
            Path(cache_path).expanduser()
            if cache_path is not None
            else base / "prisma" / "predictions.sqlite3"
        )
        self.progress = progress

    def _request(self, route: str, payload: dict | None = None) -> Any:
        body = (
            None
            if payload is None
            else json.dumps(payload, allow_nan=False).encode("utf-8")
        )
        request = urllib.request.Request(
            self.url + route,
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                detail = error.read(2048).decode("utf-8", errors="replace")
                error.close()
                if error.code in {502, 503, 504} and attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise RemotePredictionError(
                    f"{self.url}{route}: HTTP {error.code}: {detail}"
                ) from error
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
            ) as error:
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise RemotePredictionError(
                    f"{self.url}{route}: connection failed: {error}"
                ) from error
            try:
                return json.loads(raw)
            except (ValueError, UnicodeError) as error:
                raise RemotePredictionError(
                    f"{self.url}{route}: response is not valid JSON"
                ) from error
        raise AssertionError("unreachable")

    def info(self) -> dict:
        """Fetch current model identity, output schema, units and runtime settings."""
        return validate_info(self._request("/info"))

    def predict(
        self,
        structures: Sequence[Structure],
        *,
        select: str | None = None,
        refresh: bool = False,
    ) -> list[dict] | np.ndarray:
        """Return complete dictionaries, or an array for the selected output.

        select only extracts an output field; it never selects an endpoint
        or model. Both cached and fresh responses are checked against /info.
        Each successfully validated chunk is committed before the next request.
        refresh=True recalculates all requested structures.
        """
        if isinstance(structures, Structure) or not isinstance(structures, Sequence):
            raise TypeError(
                "structures must be a sequence of pymatgen Structure objects"
            )
        info = (
            self.info()
        )  # Re-fetch each call: the model at this URL may have changed.
        outputs = info["outputs"]
        if select is not None and select not in outputs:
            raise ValueError(
                f"Unknown output {select!r}; available outputs: {', '.join(outputs)}"
            )
        fingerprint = info["model_fingerprint"]
        results = []
        completed = 0
        database = None
        bar = tqdm(
            total=len(structures),
            desc="Predicting",
            unit="structure",
            disable=not self.progress,
        )
        try:
            if self.cache and len(structures):
                database = PredictionCache(self.cache_path)
                database.register(info, self.url)
            for start in range(0, len(structures), self.chunk_size):
                chunk = structures[start : start + self.chunk_size]
                hashes = []
                payloads = {}
                for offset, structure in enumerate(chunk):
                    if not isinstance(structure, Structure):
                        raise TypeError(
                            f"structures[{start + offset}] is not a pymatgen Structure"
                        )
                    # Pymatgen's encoder handles NumPy values in site properties.
                    payload = json.loads(structure.to(fmt="json"))
                    canonical = json.dumps(
                        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    hashes.append(key)
                    payloads[key] = payload
                cached = (
                    database.get_many(fingerprint, list(payloads))
                    if database is not None and not refresh
                    else {}
                )
                for prediction in cached.values():
                    validate_prediction(prediction, outputs)
                missing = [key for key in payloads if key not in cached]
                if missing:
                    response = self._request(
                        "/predict", {"structures": [payloads[key] for key in missing]}
                    )
                    predictions = validate_response(response, info, len(missing))
                    fresh = dict(zip(missing, predictions, strict=True))
                    if database is not None:
                        database.put_many(fingerprint, fresh)
                    cached.update(fresh)
                for key in hashes:
                    prediction = cached[key]
                    results.append(
                        copy.deepcopy(prediction)
                        if select is None
                        else prediction[select]
                    )
                completed += len(chunk)
                bar.update(len(chunk))
        except (Exception, KeyboardInterrupt) as error:
            resume = (
                f"Completed chunks are saved in {self.cache_path}. Repeat the same predict() call "
                "to reuse them."
                if self.cache
                else "Caching is disabled; completed chunks were not saved."
            )
            error.add_note(
                f"RemotePredictor {self.url}: {completed}/{len(structures)} input positions "
                f"completed; failed while processing positions starting at {completed}. {resume}"
            )
            raise
        finally:
            bar.close()
            if database is not None:
                database.close()
        if select is None:
            return results
        spec = outputs[select]
        dtype = {"float": np.float64, "int": np.int64, "bool": np.bool_, "string": str}[
            spec["type"]
        ]
        if not results:
            return np.empty((0, *spec.get("shape", [])), dtype=dtype)
        return np.asarray(results, dtype=dtype)
