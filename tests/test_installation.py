from importlib.resources import files
import subprocess

import pytest
import torch
from torch_scatter import scatter_add
from torch_sparse import SparseTensor


def test_required_package_assets_are_installed():
    assert files("prisma.backbones.equiformer_v2").joinpath("Jd.pt").is_file()
    assert files("prisma.configs").joinpath("training/default.yaml").is_file()


def test_compiled_extensions_execute():
    values = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 0, 1])
    assert torch.equal(scatter_add(values, index), torch.tensor([3.0, 3.0]))

    sparse = SparseTensor(
        row=torch.tensor([0, 1]),
        col=torch.tensor([0, 1]),
        value=torch.tensor([2.0, 3.0]),
        sparse_sizes=(2, 2),
    )
    assert torch.equal(sparse.matmul(torch.ones(2, 1)), torch.tensor([[2.0], [3.0]]))


@pytest.mark.parametrize("command", [[], ["data"], ["train"]])
def test_installed_cli_is_available(command):
    result = subprocess.run(
        ["prisma", *command, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: prisma" in result.stdout
