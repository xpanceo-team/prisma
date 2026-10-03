# Remote property prediction

Use a container predictor from the same Python environment that runs PRISMA:

```python
from prisma import RemotePredictor

predictor = RemotePredictor("http://127.0.0.1:18000")
bandgaps = predictor.predict(structures, select="bandgap_pbe")
```

`structures` is a sequence of `pymatgen.core.Structure` objects. The endpoint is
always supplied by the caller. There is no built-in model registry or model-specific
client class. No predictor-model packages are installed in the PRISMA environment.

## Inputs and outputs

Load existing CIF, POSCAR or Pymatgen JSON structures using Pymatgen:

```python
import json
from pymatgen.core import Structure

structures = [
    Structure.from_file("candidate.cif"),
    Structure.from_file("POSCAR"),
]
# For a string previously produced by Structure.to(fmt="json"):
structure = Structure.from_dict(json.loads(structure_json))
```

The client sends Pymatgen MSON JSON over HTTP. It does not require shared
input/output volumes or intermediate dataset files.

Inspect the remote model before choosing an output:

```python
info = predictor.info()
print(info["model"])
print(info["outputs"])
```

`predict(structures)` returns a list of complete output dictionaries in input order.
`select="field"` extracts one declared field as a NumPy array. It does not select
an endpoint, checkpoint or model. For example, an SHG service may expose
`d_matrix`, `d_max`, and symmetry flags:

```python
shg = RemotePredictor("http://127.0.0.1:18001")

shg_max = shg.predict(structures, select="d_max")       # shape (N,)
tensors = shg.predict(structures, select="d_matrix")   # shape (N, 3, 6)
all_outputs = shg.predict(structures)                 # list[dict]
```

Names and units come from `/info.outputs`. For the current OptiXNet adapter,
`d_max` is the largest absolute tensor component, in the training-label units
reported as `model_native`; do not assume these are pm/V.

Empty selected outputs retain their declared shape, for example `(0, 3, 6)`.
Duplicate structures retain their original positions.

## Large runs and recovery

The same call accepts a large list:

```python
predictor = RemotePredictor(
    "http://127.0.0.1:18000",
    chunk_size=128,
    timeout=300,
)
values = predictor.predict(structures, select="bandgap_pbe")
```

Only one chunk is serialized and sent at a time. HTTP chunk size is independent
of the service's GPU batch size. A progress bar includes cached results.

After each successful response, the client validates the fingerprint, output
count, fields, types, shapes and finite numeric values, then saves that complete
chunk in SQLite. An invalid response is never saved. If a request fails or you
interrupt the call, repeat the same call, including after restarting Python.
Completed chunks are reused; at most the in-flight chunk needs recalculation.

The default cache is `$XDG_CACHE_HOME/prisma/predictions.sqlite3`, or
`~/.cache/prisma/predictions.sqlite3` when XDG_CACHE_HOME is unset. It is a private
implementation detail; no database server is required.

```python
predictor = RemotePredictor(
    "http://127.0.0.1:18000",
    cache_path="./prediction-cache.sqlite3",
    progress=False,
)
values = predictor.predict(structures, select="bandgap_pbe")

# Explicit recalculation (successful new responses replace existing entries):
values = predictor.predict(structures, select="bandgap_pbe", refresh=True)

# Disable disk persistence:
uncached = RemotePredictor("http://127.0.0.1:18000", cache=False)
```

The key includes the model fingerprint and a hash of the complete structure JSON,
including site properties. Changing the model invalidates its old predictions.
The endpoint URL is not a model identity: identical fingerprints may reuse results
across URLs. This assumes the service correctly changes its fingerprint whenever
prediction semantics change.

The full result is saved even when `select=` is used. Selecting another output
later does not repeat inference. The service must remain reachable for `/info`
so the client can verify which model is current.

The client makes one additional attempt by default for connection failures,
timeouts and HTTP 502/503/504. Use `retries=0` to disable retry. Model errors such
as HTTP 500 and invalid inputs such as HTTP 422 are not retried. A timeout can
cause the read-only inference request to be repeated. No server job queue is
required.

Calls are synchronous. When two services share a GPU, run their predictions
sequentially unless you have measured that concurrent workloads fit.

## Container interface

The service must implement predictor-container protocol version 1:

- `GET /info`: model name, revision, checkpoint SHA-256, model fingerprint,
  `input_schema="pymatgen.structure.mson.v1"`, and output declarations.
- `POST /predict`: accepts `{"structures": [MSON, ...]}` and returns
  `{"model_fingerprint": "...", "predictions": [{...}, ...]}` in input order.

Supported declared output types are `float`, `int`, `bool`, and `string`, with
an optional fixed `shape` list. A changed fingerprint during a call raises
`RemotePredictionError`; results from different model revisions are never
combined into one returned array.

Model/container startup, GPU assignment and URL choice belong to the deployment.
`RemotePredictor` does not start or stop containers.
