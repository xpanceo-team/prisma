"""Small, transactional cache of complete per-structure predictions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ._protocol import PROTOCOL_VERSION


class PredictionCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictor_models_v1 (
                    fingerprint TEXT PRIMARY KEY,
                    info TEXT NOT NULL,
                    endpoint TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions_v1 (
                    protocol INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    structure_hash TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (protocol, fingerprint, structure_hash)
                );
                """
            )
        except BaseException:
            self.connection.close()
            raise

    def register(self, info: dict, endpoint: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO predictor_models_v1 VALUES (?, ?, ?)",
                (
                    info["model_fingerprint"],
                    json.dumps(info, allow_nan=False),
                    endpoint,
                ),
            )

    def get_many(self, fingerprint: str, hashes: list[str]) -> dict[str, dict]:
        result = {}
        # Keep the bound-parameter count below SQLite's older 999-parameter limit.
        for start in range(0, len(hashes), 500):
            chunk = hashes[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                "SELECT structure_hash, prediction FROM predictions_v1 "
                f"WHERE protocol=? AND fingerprint=? AND structure_hash IN ({placeholders})",
                (PROTOCOL_VERSION, fingerprint, *chunk),
            )
            result.update((key, json.loads(value)) for key, value in rows)
        return result

    def put_many(self, fingerprint: str, predictions: dict[str, dict]) -> None:
        # One transaction per validated HTTP response; no partial chunk is committed.
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO predictions_v1 "
                "(protocol, fingerprint, structure_hash, prediction) VALUES (?, ?, ?, ?)",
                [
                    (
                        PROTOCOL_VERSION,
                        fingerprint,
                        key,
                        json.dumps(value, allow_nan=False),
                    )
                    for key, value in predictions.items()
                ],
            )

    def close(self) -> None:
        self.connection.close()
