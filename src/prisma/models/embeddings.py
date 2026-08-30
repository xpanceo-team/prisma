import math

import torch
from pymatgen.core import Composition
from torch import nn


class NoiseLevelEncoding(nn.Module):
    """
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
            )
            * (-math.log(10000.0) / d_model),
        )
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        x = torch.zeros((t.shape[0], self.d_model), device=self.div_term.device)
        x[:, 0::2] = torch.sin(t[:, None] * self.div_term[None])
        x[:, 1::2] = torch.cos(t[:, None] * self.div_term[None])

        return x


class CategoricalEncoding(nn.Module):
    """
    Expects input tensor of integer category indices in the range [0, num_categories-1].
    Learns an embedding of size `hidden_dim` for each category index.
    """
    def __init__(self, num_categories: int, hidden_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_categories, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long())


class SpaceGroupEncoding(CategoricalEncoding):
    def __init__(self, hidden_dim: int):
        super().__init__(num_categories=230, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return embedding of the space group,
        1 is subtracted from the space group number to make it zero-indexed.
        """
        return super().forward(x - 1)


class ChemicalSystemEncoding(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding = torch.nn.Linear(input_dim, out_features=hidden_dim)

    def forward(self, x: list[str]) -> torch.Tensor:
        """
        Transforms list of formulas into multi hot representation of chemical system.
        """
        compositions = [Composition(formula) for formula in x]

        device = next(self.parameters()).device
        idx = [
            [element.number - 1 for element in composition.elements]
            for composition in compositions
        ]

        batch_indices = []
        element_indices = []
        for batch_idx, el_idx in enumerate(idx):
            batch_indices.extend([batch_idx] * len(el_idx))
            element_indices.extend(el_idx)

        batch_indices = torch.LongTensor(batch_indices).to(device)
        element_indices = torch.LongTensor(element_indices).to(device)

        max_atomic_number = self.input_dim

        multi_hot = torch.zeros(
            len(idx),
            max_atomic_number,
            dtype=torch.float,
            device=device,
        )
        multi_hot[batch_indices, element_indices] = 1.0

        return self.embedding(multi_hot)


class ControlNetAdapter(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()

        self.emb_size = emb_size

        self.proj = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
        )
        self.mixin = nn.Linear(self.emb_size, self.emb_size, bias=False)
        nn.init.zeros_(self.mixin.weight)

    def forward(self, x, cond):
        out = self.proj(torch.cat([x, cond], dim=-1))
        out = self.mixin(out)

        return out


class VectorEncoding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)

        return self.proj(x)
