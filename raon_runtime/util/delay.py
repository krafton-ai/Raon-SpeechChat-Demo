"""Acoustic delay utilities for deployed codebook decoding.

These functions are in a separate module to avoid circular imports between
model.py (which imports from inference.py) and inference.py (which needs undelay).
"""

import torch


def undelay_audio_codes(
    delays: list[int],
    audio_codes: torch.Tensor,
    padding_value: int = 0,
) -> torch.Tensor:
    """Inverse of delay_audio_codes: shift codes back to original alignment."""
    if all(d == 0 for d in delays):
        return audio_codes
    squeeze_batch = False
    if audio_codes.dim() == 2:
        audio_codes = audio_codes.unsqueeze(0)
        squeeze_batch = True
    B, T, K = audio_codes.shape
    audio_codes_t = audio_codes.transpose(1, 2)
    undelayed = []
    for k, delay in enumerate(delays):
        if delay == 0:
            undelayed.append(audio_codes_t[:, k])
        else:
            line = audio_codes_t[:, k].roll(-delay, dims=1)
            line[:, -delay:] = padding_value
            undelayed.append(line)
    result = torch.stack(undelayed, dim=1).transpose(1, 2)
    if squeeze_batch:
        result = result.squeeze(0)
    return result
