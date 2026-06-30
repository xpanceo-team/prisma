from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.configuration_utils import register_to_config

from crystal_diffusers.evaluation.losses import (
    vb_terms_bpd,
    accuracy_per_structure,
    cross_entropy_x_start,
)
from crystal_diffusers.schedulers.scheduling_utils import SchedulerMixin
from crystal_diffusers.configuration_utils import ConfigMixin


def _cumulative_matrix_product(tensors):
    result = []
    cumprod = tensors[0]
    result.append(cumprod)

    for tensor in tensors[1:]:
        cumprod = cumprod @ tensor
        result.append(cumprod)

    return torch.stack(result)


def categorical_sample(
    logits,
    deterministic_mask: Optional = None,
    generator: torch.Generator | None = None,
):
    noise = torch.rand(size=logits.shape, device=logits.device, generator=generator)

    # To avoid numerical issues, clip the noise to a minimum value
    noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny)
    gumbel_noise = -torch.log(-torch.log(noise))

    if deterministic_mask is not None:
        gumbel_noise[deterministic_mask] = 0.0

    sample = torch.argmax(logits + gumbel_noise, dim=-1)

    return sample


class D3PMScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_categories: int = 101,
        absorbing_state: int = 0,
    ):

        timesteps = torch.arange(0, num_train_timesteps, dtype=torch.int)

        self.betas = 1 / (num_train_timesteps - timesteps)
        self.betas_prod = torch.cumprod(self.betas, dim=0)

        forward_step_matrices = torch.stack(
            [self._get_forward_step_transition_matrix(t) for t in timesteps]
        )

        self.cumprod_transition_matrices = _cumulative_matrix_product(
            forward_step_matrices
        )

        self.forward_step_transposed_matrices = forward_step_matrices.transpose(1, 2)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        num_atoms: torch.Tensor,
        batch_idx: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        """Sample one timestep from the model p(x_{t-1} | x_t) using PyTorch."""
        t = torch.repeat_interleave(timestep, num_atoms)

        model_logits, pred_x_start_logits = self.reverse_diffusion_logits(
            model_output=model_output, x=sample, t=t
        )

        prev_sample = categorical_sample(
            model_logits,
            deterministic_mask=t == 0,
            generator=generator,
        )

        assert prev_sample.shape == sample.shape
        assert pred_x_start_logits.shape == model_logits.shape

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
        timesteps = torch.repeat_interleave(timesteps, num_atoms)

        transition_probs = self.cumprod_transition_matrices.to(num_atoms.device)
        logits = transition_probs[timesteps, original_samples]
        zero_mask = torch.isclose(logits, torch.tensor(0.0))

        eps = torch.finfo(logits.dtype).tiny
        logits = logits.clamp(min=eps).log()

        logits[zero_mask] = -torch.inf

        noised_sample = categorical_sample(logits)

        return noised_sample

    def sample_prior(
        self,
        num_atoms: torch.Tensor,
    ):
        size = (num_atoms.sum().item(),)

        # Kronecker delta distribution focused on the absorbing state.
        x_init = torch.full(
            size=size,
            fill_value=self.config.absorbing_state,
            device=num_atoms.device,
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
        lambda_coef: float = 0.01,
    ):
        t_broadcast = torch.repeat_interleave(timestep, num_atoms)

        (
            prev_atom_types_logits,
            pred_initial_atom_types_logits,
        ) = self.reverse_diffusion_logits(model_output, x=noised_sample, t=t_broadcast)

        # Calculate hybrid loss for atom types diffusion
        prev_true_atom_types_logits = self.reverse_posterior_logits(
            sample,
            noised_sample,
            t_broadcast,
            x_0_is_logits=False,
        )

        types_vb_loss_per_structure = vb_terms_bpd(
            true_logits=prev_true_atom_types_logits,
            model_logits=prev_atom_types_logits,
            x_start=sample,
            t=timestep,
            batch=batch_idx,
        )
        types_vb_loss = types_vb_loss_per_structure.mean()

        types_ce_loss_per_structure = cross_entropy_x_start(
            sample, pred_initial_atom_types_logits, batch=batch_idx
        )
        types_ce_loss = types_ce_loss_per_structure.mean()

        types_accuracy_per_structure = accuracy_per_structure(
            sample, pred_initial_atom_types_logits, batch=batch_idx
        )
        types_accuracy = types_accuracy_per_structure.mean()

        atom_types_loss = types_vb_loss + lambda_coef * types_ce_loss

        return (
            atom_types_loss,
            types_vb_loss.item(),
            types_ce_loss.item(),
            types_accuracy.item(),
        )

    def reverse_diffusion_logits(
        self, model_output, x, t, model_output_type: str = "logits"
    ):
        """Compute logits of p(x_{t-1} | x_t) using PyTorch."""
        assert t.shape == (x.shape[0],)

        if model_output_type == "logits":
            pred_x_start_logits = model_output
        elif model_output_type == "logistic_pars":
            # Get logits out of discretized logistic distribution.
            loc, log_scale = model_output
            pred_x_start_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        else:
            raise NotImplementedError(model_output_type)

        # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
        # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)

        t_broadcast = torch.unsqueeze(t, -1)
        # here t is actually an index of timestep in range [0, ... T-1]
        # so when t is 0 we have x = x_1,
        # and model_output is logits p(x_0 | x_1) so we don't need to reparameterize them
        pred_x_prev_logits = torch.where(
            t_broadcast == 0,
            pred_x_start_logits,
            self.reverse_posterior_logits(
                pred_x_start_logits, x, t, x_0_is_logits=True
            ),
        )

        return pred_x_prev_logits, pred_x_start_logits

    def reverse_posterior_logits(self, x_0, x_t, t, x_0_is_logits):
        """Computes the logits of the reverse posterior distribution q(x_{t-1} | x_t, x_start).

        This function calculates the logits for the reverse process in a diffusion model,
        specifically estimating the distribution over the state at time t-1 given the state
        at time t and the initial data point x_start.
        """

        # Get q(X_t = x | X_{t-1}) probabilities as a row vector
        fact1 = self.forward_step_transposed_matrices.to(x_0.device)[t, x_t]

        eps = torch.finfo(fact1.dtype).tiny
        if x_0_is_logits:
            fact2 = self.get_marginalized_reverse_step_probs(
                F.softmax(x_0, dim=-1), t - 1
            )
            x_0_logits = x_0
        else:
            # Get q(X_t | X_0 = x) probabilities as a row vector
            fact2 = self.cumprod_transition_matrices.to(x_0.device)[t - 1, x_0]

            # logits of one hot distribution with all mass on x_0
            x_0_logits = (
                F.one_hot(
                    x_0.to(torch.int64),
                    num_classes=self.config.num_categories,
                )
                .clamp(min=eps)
                .log()
            )

        x_prev_logits = torch.log(fact1.clamp(min=eps)) + torch.log(
            fact2.clamp(min=eps)
        )
        t_broadcast = torch.unsqueeze(t, -1)

        # Here t is actually an index of a timestep in range [0, ... T-1],
        # so when t is 0 we have x = x_1 and the logits should be equal to the log of x_0
        logits = torch.where(t_broadcast == 0, x_0_logits, x_prev_logits)

        return logits

    def get_marginalized_reverse_step_probs(self, x, t):
        """Computes for reverse transitions at time t.

        This function sums over the product of the conditional probabilities q(x_{t-1} | x_0)
        with the predicted probabilities x (representing p_theta(x_0 | x_t, t)) for all possible
        initial states x_0, yielding the probability of stepping back from state x_t to x_{t-1}.
        """
        output = torch.matmul(
            x.unsqueeze(1), self.cumprod_transition_matrices.to(x.device)[t]
        ).squeeze(1)

        return output

    def _get_forward_step_transition_matrix(self, t: int, device="cpu"):
        """Computes transition matrix with absorbing state m:
        [Q_t]_ij = 1 if (i = j = m)
        [Q_t]_ij = 1-beta_t if (i = j != m)
        [Q_t]_ij = beta_t if (j=m, i!=m)
        """

        absorbing_state = self.config.absorbing_state
        num_categories = self.config.num_categories
        beta_t = self.betas[t]

        q = torch.zeros(num_categories, num_categories, device=device)
        q.fill_diagonal_(1 - beta_t)
        q[:, absorbing_state] = beta_t
        q[absorbing_state, absorbing_state] = 1

        # check that all rows sum to 1
        # assert torch.isclose(q.sum(dim=1), torch.tensor(1.0)).all()

        return q
