import torch

from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.configuration_utils import register_to_config
from torch_scatter import segment_coo

from prisma.configuration_utils import ConfigMixin
from prisma.schedulers.scheduling_utils import SchedulerMixin


class VarianceExplodingScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_max: float | None = 5.0,
        sigma_min: float | None = 0.01,
        signal_to_noise_ratio: float = 0.4,
        max_cell_offset: int = 3,
        max_num_atoms: int | None = None,
        drop_sigma_min: bool = False,
    ):
        if sigma_max is None or sigma_min is None:
            if max_num_atoms is None:
                raise ValueError(
                    f"Either 'sigma_max', 'sigma_min' or 'max_num_atoms' are required "
                    f"to initialize scheduler."
                )

            # set noise accoring to techniques in
            # Song & Ermon (2020)
            # Improved Techniques for Training Score-Based Generative Model

            # max euclidean distance between frac coords
            max_dist = 3 ** (1 / 2)

            # multiply with max_atoms_num because we normalize on atom density
            sigma_max = max_num_atoms ** (1 / 3) * max_dist
            sigma_min = max_num_atoms ** (1 / 3) * 0.01

            self.register_to_config(sigma_min=sigma_min, sigma_max=sigma_max)

        if drop_sigma_min:
            num_timesteps = num_train_timesteps + 1
        else:
            num_timesteps = num_train_timesteps

        self.sigmas = torch.logspace(
            torch.log10(torch.tensor(sigma_min)),
            torch.log10(torch.tensor(sigma_max)),
            num_timesteps,
        )

        if drop_sigma_min:
            self.sigmas = self.sigmas[1:]

        k_values = torch.arange(-max_cell_offset, max_cell_offset + 1)

        self._k_grid = torch.stack(
            torch.meshgrid(k_values, k_values, k_values, indexing="ij"),
            dim=-1,
        ).float()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        num_atoms: torch.Tensor,
        batch_idx: torch.Tensor,
        corrector: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        std = self._get_sigma(timestep, num_atoms).unsqueeze(1)
        model_output /= std

        if not corrector:
            prev_sample = self._predictor_algorithm(
                model_output,
                timestep,
                sample,
                num_atoms,
                generator,
            )

        else:
            prev_sample = self._corrector_algorithm(
                model_output,
                sample,
                batch_idx,
                generator,
            )

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        num_atoms: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """Sample from q(x_t | x_start) using PyTorch (i.e., add noise to the data)."""
        sigma_t = self._get_sigma(timesteps, num_atoms)

        assert len(sigma_t) == len(original_samples)

        noised_sample = (
            original_samples + sigma_t.unsqueeze(1) * torch.randn_like(original_samples)
        ) % 1

        return noised_sample

    def sample_prior(
        self,
        num_atoms: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        size = (num_atoms.sum().item(), 3)

        x_init = torch.rand(
            size=size,
            device=num_atoms.device,
            generator=generator,
        )

        return x_init

    def get_loss(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        noised_sample: torch.Tensor,
        num_atoms: torch.Tensor,
        batch_idx: torch.Tensor,
    ):

        sigma_t = self._get_sigma(timestep, num_atoms)

        true_frac_coords_score = self.get_score(
            noised_sample,
            sample,
            sigma_t,
        )
        std = sigma_t.unsqueeze(1)

        coords_score_diff_squared_norm = torch.mean(
            (model_output - true_frac_coords_score * std) ** 2, dim=-1
        )

        coords_loss_per_atom = coords_score_diff_squared_norm
        coords_loss_per_structure = segment_coo(
            coords_loss_per_atom, batch_idx, reduce="sum"
        )
        coords_loss = coords_loss_per_structure.mean()

        return coords_loss

    def get_score(self, noised_sample, sample, std):
        """Compute fractional coordinates score"""

        # shape: num_atoms x 1
        variance = (std**2).unsqueeze(1)

        # shape: max_offset x max_offset x max_offset x dim
        k_grid = self._k_grid.to(sample.device)

        # shape: 1 x grid_size x dim
        k_grid = k_grid.view(1, -1, 3)

        # Calculate the norm squared ||x - x_0 + k||^2

        # shape: num_atoms x grid_size x dim
        coords_diff_with_pbc = (sample - noised_sample).view(-1, 1, 3) + k_grid

        # shape: num_atoms x grid_size
        distances_squared = (coords_diff_with_pbc**2).sum(dim=-1)

        # Compute weights
        weights = -distances_squared / (2 * variance)

        # shape: num_atoms x grid_size x 1
        weights_normalized = weights.softmax(-1).unsqueeze(-1)

        # Compute the weighted sum
        score = (weights_normalized * coords_diff_with_pbc).sum(dim=1) / variance

        return score

    def _get_sigma(self, timestep: torch.Tensor, num_atoms: torch.Tensor):
        sigma_t = self.sigmas.to(num_atoms.device)[timestep]

        sigma_t /= num_atoms ** (1 / 3)

        return torch.repeat_interleave(sigma_t, num_atoms)

    def _predictor_algorithm(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        num_atoms: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        device = model_output.device

        sigma_t = self._get_sigma(timestep, num_atoms)

        # FIXME: torch.where t==0
        if timestep[0] == 0:
            sigma_t_prev = torch.zeros_like(sigma_t, device=device)
        else:
            sigma_t_prev = self._get_sigma(timestep - 1, num_atoms)

        sigma_squares_diff = (sigma_t**2 - sigma_t_prev**2).unsqueeze(1).to(device)

        noise = torch.randn(size=sample.size(), device=device, generator=generator)

        norm_noise = noise * sigma_squares_diff ** (1 / 2)

        sample_diff = sigma_squares_diff * model_output + norm_noise

        prev_sample = (sample + sample_diff) % 1

        return prev_sample

    def _corrector_algorithm(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        batch_idx: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        device = model_output.device

        noise = torch.randn(size=sample.size(), device=device, generator=generator)

        noise_norm = segment_coo((noise**2).sum(-1), batch_idx, reduce="sum") ** (1 / 2)

        model_output_norm = segment_coo(
            (model_output**2).sum(-1), batch_idx, reduce="sum"
        ) ** (1 / 2)

        noise_norm = noise_norm.mean()
        model_output_norm = model_output_norm.mean()

        step_size = (
            2
            * (self.config.signal_to_noise_ratio * noise_norm / model_output_norm) ** 2
        )

        norm_noise = (2 * step_size) ** (1 / 2) * noise
        sample_diff = step_size * model_output + norm_noise

        prev_sample = (sample + sample_diff) % 1

        return prev_sample
