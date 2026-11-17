"""GRU history network inference: pack the native frame into [sequence, lead, feature] tensors and run the network.

One sequence is one (county, issue). Features are standardised with the fitted normaliser, NaN set to 0 and clipped
to [-8, 8]; the history and state inputs come from features.history.sequence_inputs. Each sequence is evaluated on its
own (batch of one), as when the reference predictions were produced.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch

from .features.history import sequence_inputs
from .model_store import feature_matrix

TIME = ['issue_sin', 'issue_cos', 'day_sin', 'day_cos']


def predict(model, norm: dict, frame: pl.DataFrame, raw: dict, lead_step: int, leads: int, config, device: str,
            q95: float) -> np.ndarray:
    """frame: native rows sorted by (county, issue, lead_minutes); raw: (county, issue) -> 672-step float32 history.
    Returns float32 predictions aligned with frame rows."""
    keys = frame.select('county', 'issue').unique(maintain_order=True)
    sid = frame.select('county', 'issue').join(keys.with_row_index('sequence'), on=['county', 'issue'],
                                               maintain_order='left')['sequence'].to_numpy()
    lead = frame['lead_minutes'].to_numpy() // lead_step - 1
    features = norm['features']
    x = np.zeros((keys.height, leads, len(features)), np.float32)
    x[sid, lead] = np.clip(np.nan_to_num((feature_matrix(frame, features) - norm['feature_mean']) / norm['feature_std'], nan=0.), -8, 8)
    present = np.zeros((keys.height, leads), bool)
    present[sid, lead] = True
    index = {c: i for i, c in enumerate(config.county_order)}
    county = np.array([index[c] for c in keys['county']], np.int64)
    denominator = np.array([config.denominators[c] for c in keys['county']], np.float32)
    time = frame.group_by('county', 'issue', maintain_order=True).agg(pl.col(n).first() for n in TIME).select(TIME).to_numpy().astype(np.float32)
    raws = np.stack([raw[c, int(t)] for c, t in keys.iter_rows()])
    hist, state, _ = sequence_inputs(raws, keys['issue'].to_numpy(), q95)
    state = ((state - norm['state_mean']) / norm['state_std']).astype(np.float32)
    out = np.zeros((keys.height, leads), np.float32)
    with torch.no_grad():
        for i in range(keys.height):
            s = slice(i, i + 1)
            p = model(torch.as_tensor(x[s], device=device).float(), torch.as_tensor(county[s], device=device),
                      torch.as_tensor(time[s], device=device), torch.as_tensor(denominator[s], device=device),
                      torch.as_tensor(present[s], device=device), torch.as_tensor(hist[s], device=device).float(),
                      torch.as_tensor(state[s], device=device).float())
            out[i] = p.float().cpu().numpy()[0]
    return out[sid, lead]
