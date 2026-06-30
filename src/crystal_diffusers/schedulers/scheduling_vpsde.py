import torch

from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.configuration_utils import register_to_config

from crystal_diffusers.configuration_utils import ConfigMixin
from crystal_diffusers.schedulers.scheduling_utils import SchedulerMixin


class VariancePreservingScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        signal_to_noise_ratio: float = 0.2,
        prior_mean_std_ratio: float = 5.18,
        inverse_average_density: float = 17.331,
    ):

        self.betas = self._get_betas()

        self.alphas = 1.0 - self.betas
        self.alpha_prod = torch.cumprod(self.alphas, dim=0)
        self.alpha_prod_sqrt = torch.sqrt(self.alpha_prod)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        num_atoms: torch.Tensor,
        batch_idx: torch.Tensor,
        corrector: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        device = model_output.device

        lattice_prior_std = self.get_prior_std(num_atoms)
        std = torch.sqrt(1 - self.alpha_prod.to(device)[timestep]) * lattice_prior_std
        std = std.view(-1, 1, 1)

        model_output = model_output / std

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
                timestep,
                sample,
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
        """Sample from q(x_t | x_start) (i.e., add noise to the data)."""
        alpha_prod_sqrt_t = (
            (self.alpha_prod_sqrt.to(num_atoms.device)[timesteps])
            .unsqueeze(1)
            .unsqueeze(1)
        )
        alpha_prod_t = (
            (self.alpha_prod.to(num_atoms.device)[timesteps]).unsqueeze(1).unsqueeze(1)
        )

        prior_mean = torch.eye(3, device=original_samples.device).unsqueeze(
            0
        ) * self.get_prior_mean(num_atoms).unsqueeze(1).unsqueeze(1)
        prior_std = self.get_prior_std(num_atoms).unsqueeze(1).unsqueeze(1)

        loc = (
            alpha_prod_sqrt_t * original_samples + (1 - alpha_prod_sqrt_t) * prior_mean
        )
        scale = torch.sqrt(1 - alpha_prod_t) * prior_std

        noise = torch.randn(len(original_samples), 3, 3, device=original_samples.device)

        # symmetrize
        noise = torch.tril(noise) + torch.triu(noise.transpose(1, 2), 1)

        noised_sample = loc + noise * scale

        return noised_sample

    def sample_prior(
        self,
        num_atoms: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        prior_mean = self.get_prior_mean(num_atoms).unsqueeze(-1).unsqueeze(-1)
        prior_std = self.get_prior_std(num_atoms).unsqueeze(-1).unsqueeze(-1)

        loc = torch.eye(3, device=num_atoms.device).unsqueeze(0) * prior_mean

        noise = torch.randn(
            num_atoms.shape[0],
            3,
            3,
            device=num_atoms.device,
            generator=generator,
        )

        prior_lattice = loc + noise * prior_std

        # symmetrize
        prior_lattice = torch.tril(prior_lattice) + torch.triu(
            prior_lattice.transpose(1, 2), 1
        )

        return prior_lattice

    def get_loss(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        noised_sample: torch.Tensor,
        num_atoms: torch.Tensor,
        batch_idx: torch.Tensor,
    ):
        true_lattice_score = self.get_score(noised_sample, sample, timestep, num_atoms)

        lattice_prior_std = self.get_prior_std(num_atoms)
        alpha_prod_t = self.alpha_prod.to(num_atoms.device)[timestep]
        std = torch.sqrt(1 - alpha_prod_t) * lattice_prior_std
        std = std.view(-1, 1, 1)

        lattice_score_diff_norm = torch.mean(
            (model_output - true_lattice_score * std) ** 2, dim=(1, 2)
        )

        loss_pre_structure = lattice_score_diff_norm
        loss = loss_pre_structure.mean()

        return loss

    def get_score(self, noised_sample, sample, timestep, num_atoms):
        """Compute lattice score"""

        alpha_prod_sqrt_t = (
            (self.alpha_prod_sqrt.to(num_atoms.device)[timestep])
            .unsqueeze(1)
            .unsqueeze(1)
        )
        alpha_prod_t = (
            (self.alpha_prod.to(num_atoms.device)[timestep]).unsqueeze(1).unsqueeze(1)
        )

        prior_mean = torch.eye(3, device=sample.device).unsqueeze(
            0
        ) * self.get_prior_mean(num_atoms).unsqueeze(1).unsqueeze(1)
        prior_std = self.get_prior_std(num_atoms).unsqueeze(1).unsqueeze(1)

        loc = alpha_prod_sqrt_t * sample + (1 - alpha_prod_sqrt_t) * prior_mean
        scale = torch.sqrt(1 - alpha_prod_t) * prior_std

        score = -(noised_sample - loc) / scale**2

        return score

    def get_prior_mean(self, num_atoms: torch.Tensor):
        inverse_average_density = self.config.inverse_average_density
        prior_mean = (num_atoms * inverse_average_density) ** (1 / 3)

        return prior_mean

    def get_prior_std(self, num_atoms: torch.Tensor):
        prior_mean_std_ratio = self.config.prior_mean_std_ratio
        prior_mean = self.get_prior_mean(num_atoms)

        std = prior_mean / prior_mean_std_ratio

        return std

    def _predictor_algorithm(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        num_atoms: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        device = model_output.device

        noise = torch.randn(size=sample.size(), device=device, generator=generator)

        symm_noise = torch.tril(noise) + torch.triu(noise.transpose(2, 1), 1)

        beta_t = self.betas.to(device)[timestep].unsqueeze(1).unsqueeze(1)

        norm_noise = beta_t ** (1 / 2) * symm_noise

        alpha_t = self.alphas.to(device)[timestep].unsqueeze(1).unsqueeze(1)
        mean_coeff = 2 - alpha_t ** (1 / 2)

        prior_mean = torch.eye(3, device=device).unsqueeze(0) * self.get_prior_mean(
            num_atoms
        ).unsqueeze(1).unsqueeze(1)

        sample_mean = (
            mean_coeff * sample + beta_t * model_output + (1 - mean_coeff) * prior_mean
        )

        prev_sample = sample_mean + norm_noise

        return prev_sample

    def _corrector_algorithm(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        device = model_output.device

        noise = torch.randn(size=sample.size(), device=device, generator=generator)

        symm_noise = torch.tril(noise) + torch.triu(noise.transpose(2, 1), 1)

        noise_norm = torch.norm(symm_noise, p=2, dim=[1, 2]).mean()
        model_output_norm = torch.norm(model_output, p=2, dim=[1, 2]).mean()

        alpha_t = self.alphas.to(device)[timestep].unsqueeze(1).unsqueeze(1)
        step_size = (
            2
            * alpha_t
            * (self.config.signal_to_noise_ratio * noise_norm / model_output_norm) ** 2
        )

        norm_noise = (2 * step_size) ** (1 / 2) * symm_noise

        sample_mean = step_size * model_output

        prev_sample = sample + sample_mean + norm_noise

        return prev_sample

    def _get_betas(self):
        beta_start = self.config.beta_start
        beta_end = self.config.beta_end
        num_timesteps = self.config.num_train_timesteps

        if self.config.beta_schedule == "linear":
            pass
        elif self.config.beta_schedule == "scaled_linear":
            beta_start **= 0.5
            beta_end **= 0.5
        elif self.config.beta_schedule == "from_sde":
            beta_0 = beta_start * num_timesteps
            beta_1 = beta_end * num_timesteps

            def alpha_bar_fn(t):
                log_mean_coeff = -0.5 * t**2 * (beta_1 - beta_0) - t * beta_0
                return torch.exp(log_mean_coeff)

            timesteps = torch.linspace(0, 1, num_timesteps + 1)

            alpha_prod = alpha_bar_fn(timesteps)
            betas = 1 - alpha_prod[1:] / alpha_prod[:-1]

            return betas
        else:
            raise ValueError(f"Wrong beta_schedule: {self.config.beta_schedule}")

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

        if self.config.beta_schedule == "scaled_linear":
            betas **= 2

        return betas
