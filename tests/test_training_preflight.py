from types import SimpleNamespace

import lightning.pytorch as pl
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from prisma.training import preflight


class _TinyTrainingModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.1)


class _TinyDataModule:
    dataset_path = "materials"
    dataset_name = None
    condition = {"bandgap": {}}
    max_num_atoms = 20
    condition_stats = {}
    train_dataset = range(2)
    valid_dataset = range(1)

    def preflight_dataloader(self):
        return DataLoader(
            [0, 1],
            batch_size=2,
            collate_fn=lambda examples: {
                "data": SimpleNamespace(num_atoms=torch.tensor([20, 18]))
            },
        )


def test_preflight_runs_optimizer_step_and_writes_report(tmp_path, monkeypatch):
    module = _TinyTrainingModule()
    monkeypatch.setattr(
        preflight,
        "instantiate_training_module",
        lambda cfg, datamodule: module,
    )
    cfg = OmegaConf.create(
        {
            "data": {"batch_size": 2},
            "training": {
                "module": {"_target_": "unused"},
                "trainer": {
                    "accelerator": "cpu",
                    "devices": 1,
                    "precision": 32,
                    "max_epochs": 1,
                    "accumulate_grad_batches": 2,
                },
                "strategy": {"_target_": "unused"},
            },
        }
    )

    report = preflight.run_preflight(
        cfg,
        _TinyDataModule(),
        ckpt_path=None,
        run_dir=tmp_path,
    )

    assert module.weight.item() != 1.0
    assert report.batch_size == 2
    assert report.total_atoms == 38
    assert report.max_atoms == 20
    assert report.loss is not None
    assert (tmp_path / "preflight.yaml").is_file()
