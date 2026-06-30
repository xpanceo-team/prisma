import torch
import torch.nn.functional as F
from torch_scatter import segment_coo


def categorical_kl_logits(logits1, logits2):
    """KL divergence between categorical distributions parameterized by logits in PyTorch.

    Args:
      logits1: logits of the first distribution. Last dim is class dim.
      logits2: logits of the second distribution. Last dim is class dim.
      eps: float small number to avoid numerical issues.

    Returns:
      KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
    """
    probs1 = torch.softmax(logits1, dim=-1)

    eps = torch.finfo(logits1.dtype).tiny
    log_probs1 = logits1.softmax(dim=-1).clamp(min=eps).log()
    log_probs2 = logits2.softmax(dim=-1).clamp(min=eps).log()

    # Compute the KL divergence
    kl_div = probs1 * (log_probs1 - log_probs2)

    return kl_div.sum(dim=-1)


def vb_terms_bpd(true_logits, model_logits, x_start, t, batch):
    """Calculate specified terms of the variational bound"""

    kl = categorical_kl_logits(true_logits, model_logits)
    kl = segment_coo(kl, batch, reduce="sum")

    decoder_nll = F.cross_entropy(model_logits, x_start, reduction="none")
    decoder_nll = segment_coo(decoder_nll, batch, reduce="sum")

    # At the first timestep return the decoder NLL, otherwise return KL
    result = torch.where(t == 0, decoder_nll, kl)

    return result


def cross_entropy_x_start(x_start, pred_x_start_logits, batch):
    """Calculate crossentropy between x_start and predicted x_start"""
    ce = F.cross_entropy(pred_x_start_logits, x_start, reduction="none")
    ce = segment_coo(ce, batch, reduce="sum")

    return ce


def accuracy_per_structure(x_start, pred_x_start_logits, batch):
    """Calculate accuracy between x_start and predicted x_start"""
    pred_x_start = pred_x_start_logits.argmax(dim=-1)
    equal = (pred_x_start == x_start).float()

    return segment_coo(
        equal, batch, reduce="mean"
    )  # TODO: MatterGen sums losses of individuals
