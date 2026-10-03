"""Validation for predictor-container protocol version 1."""

from __future__ import annotations

import math
from typing import Any

PROTOCOL_VERSION = 1
INPUT_SCHEMA = "pymatgen.structure.mson.v1"
OUTPUT_TYPES = {"float": float, "int": int, "bool": bool, "string": str}


class RemotePredictionError(RuntimeError):
    """A remote predictor could not provide a valid, complete result."""


def validate_info(info: Any) -> dict:
    if not isinstance(info, dict):
        raise RemotePredictionError("/info must return a JSON object")
    if type(info.get("protocol_version")) is not int or info["protocol_version"] != 1:
        raise RemotePredictionError(
            "Unsupported predictor protocol_version; expected 1"
        )
    if info.get("input_schema") != INPUT_SCHEMA:
        raise RemotePredictionError(
            f"Unsupported input_schema; expected {INPUT_SCHEMA}"
        )
    fingerprint = info.get("model_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RemotePredictionError("/info is missing model_fingerprint")
    model = info.get("model")
    if not isinstance(model, dict) or not all(
        isinstance(model.get(key), str) and model[key]
        for key in ("name", "revision", "checkpoint_sha256")
    ):
        raise RemotePredictionError(
            "/info must declare model name, revision and checkpoint_sha256"
        )
    outputs = info.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise RemotePredictionError("/info must declare non-empty outputs")
    for field, spec in outputs.items():
        if not isinstance(field, str) or not isinstance(spec, dict):
            raise RemotePredictionError("Invalid output declaration in /info")
        if not isinstance(spec.get("type"), str) or spec["type"] not in OUTPUT_TYPES:
            raise RemotePredictionError(f"Unsupported output type for {field!r}")
        shape = spec.get("shape", [])
        if not isinstance(shape, list) or any(
            type(n) is not int or n < 1 for n in shape
        ):
            raise RemotePredictionError(f"Invalid output shape for {field!r}")
    return info


def _validate_value(value: Any, kind: str, shape: list[int], field: str) -> None:
    if shape:
        if not isinstance(value, list) or len(value) != shape[0]:
            raise RemotePredictionError(
                f"Wrong output shape for {field!r}: expected {shape}"
            )
        for item in value:
            _validate_value(item, kind, shape[1:], field)
        return
    if kind == "float":
        valid = type(value) in (int, float)
        if valid:
            try:
                valid = math.isfinite(value)
            except OverflowError:
                valid = False
    else:
        valid = type(value) is OUTPUT_TYPES[kind]
    if not valid:
        raise RemotePredictionError(f"Invalid {kind} output for {field!r}: {value!r}")


def validate_prediction(prediction: Any, outputs: dict) -> dict:
    if not isinstance(prediction, dict) or prediction.keys() != outputs.keys():
        raise RemotePredictionError("Prediction fields do not match /info.outputs")
    for field, spec in outputs.items():
        _validate_value(prediction[field], spec["type"], spec.get("shape", []), field)
    return prediction


def validate_response(response: Any, info: dict, count: int) -> list[dict]:
    if not isinstance(response, dict):
        raise RemotePredictionError("/predict must return a JSON object")
    if response.get("model_fingerprint") != info["model_fingerprint"]:
        raise RemotePredictionError(
            "Model fingerprint changed during prediction; repeat the call to use the current model"
        )
    predictions = response.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != count:
        raise RemotePredictionError(f"/predict must return exactly {count} predictions")
    for prediction in predictions:
        validate_prediction(prediction, info["outputs"])
    return predictions
