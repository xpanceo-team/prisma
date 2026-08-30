from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h1: int = 128,
        h2: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ClusterModelBundle:
    model: nn.Module
    scaler: Any
    cluster_ids: tuple[int, ...]
    name: str


class ClusterRegressor:
    def __init__(
        self,
        bundles_by_name: dict[str, ClusterModelBundle],
        cluster_to_bundle_name: dict[int, str],
        device: str = "cpu",
    ):
        self.bundles_by_name = bundles_by_name
        self.cluster_to_bundle_name = cluster_to_bundle_name
        self.device = torch.device(device)

    def predict(
        self,
        embeddings: list[Any],
        cluster_ids: list[Any],
    ) -> list[float | None]:
        if len(embeddings) != len(cluster_ids):
            raise ValueError("embeddings and cluster_ids must have the same length")

        outputs: list[float | None] = [None] * len(cluster_ids)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        grouped_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)

        for idx, (embedding, cluster_id) in enumerate(zip(embeddings, cluster_ids)):
            embedding_arr = _coerce_embedding(embedding)
            cluster_id_int = _coerce_cluster_id(cluster_id)

            if embedding_arr is None or cluster_id_int is None:
                continue

            bundle_name = self.cluster_to_bundle_name.get(cluster_id_int)
            if bundle_name is None:
                continue

            grouped_indices[bundle_name].append(idx)
            grouped_embeddings[bundle_name].append(embedding_arr)

        for bundle_name, indices in grouped_indices.items():
            bundle = self.bundles_by_name[bundle_name]
            x = np.vstack(grouped_embeddings[bundle_name]).astype(np.float32)
            x_scaled = bundle.scaler.transform(x).astype(np.float32)

            with torch.no_grad():
                x_tensor = torch.tensor(
                    x_scaled,
                    dtype=torch.float32,
                    device=self.device,
                )
                preds = bundle.model(x_tensor).detach().cpu().numpy().reshape(-1)

            for idx, pred in zip(indices, preds):
                outputs[idx] = float(pred)

        return outputs


def _coerce_embedding(embedding: Any) -> np.ndarray | None:
    if embedding is None:
        return None

    try:
        arr = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if arr.ndim != 1 or arr.size == 0:
        return None

    if not np.isfinite(arr).all():
        return None

    return arr


def _coerce_cluster_id(cluster_id: Any) -> int | None:
    if cluster_id is None:
        return None

    try:
        return int(cluster_id)
    except (TypeError, ValueError):
        return None


def load_single_bundle(
    model_path: str | Path,
    scaler_path: str | Path,
    device: str = "cpu",
    name: str | None = None,
) -> ClusterModelBundle:
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    checkpoint = torch.load(model_path, map_location=device)

    model = MLP(
        in_dim=checkpoint["input_size"],
        h1=checkpoint["arch"]["h1"],
        h2=checkpoint["arch"]["h2"],
        dropout=checkpoint["arch"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    scaler = joblib.load(scaler_path)

    cluster_ids = tuple(int(x) for x in checkpoint["cluster_ids"])
    bundle_name = name or checkpoint.get("group") or model_path.stem

    return ClusterModelBundle(
        model=model,
        scaler=scaler,
        cluster_ids=cluster_ids,
        name=str(bundle_name),
    )


def load_model(
    bundles_config: dict[str, dict[str, str]],
    device: str = "cpu",
) -> ClusterRegressor:
    """
    Example bundles_config:
    {
        "0_2_7": {
            "model_path": "/tmp/mace_clusters_gen_new/mace_mlp_dn_clusters_0_2_7_trained_on_newdf.pt",
            "scaler_path": "/tmp/mace_clusters_gen_new/mace_scaler_dn_clusters_0_2_7_trained_on_newdf.joblib",
        },
        "1_6": {
            "model_path": "/tmp/mace_clusters_gen_new/mace_mlp_dn_clusters_1_6_trained_on_newdf.pt",
            "scaler_path": "/tmp/mace_clusters_gen_new/mace_scaler_dn_clusters_1_6_trained_on_newdf.joblib",
        },
    }
    """
    bundles_by_name: dict[str, ClusterModelBundle] = {}
    cluster_to_bundle_name: dict[int, str] = {}

    for bundle_name, cfg in bundles_config.items():
        bundle = load_single_bundle(
            model_path=cfg["model_path"],
            scaler_path=cfg["scaler_path"],
            device=device,
            name=bundle_name,
        )
        bundles_by_name[bundle_name] = bundle

        for cluster_id in bundle.cluster_ids:
            if cluster_id in cluster_to_bundle_name:
                raise ValueError(
                    f"Cluster {cluster_id} is assigned to multiple bundles: "
                    f"{cluster_to_bundle_name[cluster_id]} and {bundle_name}"
                )
            cluster_to_bundle_name[cluster_id] = bundle_name

    return ClusterRegressor(
        bundles_by_name=bundles_by_name,
        cluster_to_bundle_name=cluster_to_bundle_name,
        device=device,
    )


def predict(
    embeddings,
    cluster_ids,
    model: ClusterRegressor,
):
    return model.predict(
        embeddings=embeddings,
        cluster_ids=cluster_ids,
    )


def predict_dataset_batch_properties(
    batch,
    models_by_column,
    embedding_key: str = "embedding",
    cluster_key: str = "cluster_id",
    overwrite: bool = False,
):
    if embedding_key not in batch:
        raise KeyError(f"Column {embedding_key!r} is not present in the batch.")
    if cluster_key not in batch:
        raise KeyError(f"Column {cluster_key!r} is not present in the batch.")

    embeddings = batch[embedding_key]
    cluster_ids = batch[cluster_key]
    outputs = {}

    for output_column, model in models_by_column.items():
        preds = predict(
            embeddings=embeddings,
            cluster_ids=cluster_ids,
            model=model,
        )

        if not overwrite and output_column in batch:
            outputs[output_column] = [
                old_value if old_value is not None else pred
                for old_value, pred in zip(batch[output_column], preds)
            ]
        else:
            outputs[output_column] = preds

    return outputs


def predict_dataset_properties(
    ds,
    models_by_column,
    embedding_key: str = "embedding",
    cluster_key: str = "cluster_id",
    overwrite: bool = False,
    batch_size: int = 512,
    desc: str | None = None,
):
    return ds.map(
        partial(
            predict_dataset_batch_properties,
            models_by_column=models_by_column,
            embedding_key=embedding_key,
            cluster_key=cluster_key,
            overwrite=overwrite,
        ),
        batched=True,
        batch_size=batch_size,
        desc=desc,
    )