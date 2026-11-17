"""Network definitions of the two submitted sequence models (Task A and Task B GRU history networks).

Both are the same architecture, `HistoryGRUForecaster`: a residual temporal-convolution trunk over the forecast leads,
whose context vector also receives a two-layer GRU encoding of the last 7 days of outage history and an MLP encoding of
11 event-state values. Module names and parameter shapes match the saved state dicts exactly, so the submitted weights
load unchanged. Inference only; training code is not part of the package.
"""
import torch
import torch.nn.functional as F
from torch import nn

STATE_COLUMNS = ['x0', 'x0_known', 'peak7', 'peak24', 'since_peak7_h', 'ratio7', 'slope2h', 'slope6h', 'dur_q95_h', 'area24', 'obs24']


def weather_columns(features):
    """Indices of the weather inputs that also feed the weather pathway."""
    return [i for i, n in enumerate(features)
            if '__' in n or n.startswith(('neighbor_', 'path_')) or n in ['weather_available', 'weather_run_age_hours']]


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width, kernel_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))

    def forward(self, x):
        h = self.norm(x)
        h = self.depthwise(h.transpose(1, 2)).transpose(1, 2)
        return x + self.mlp(h)


class HistoryGRUForecaster(nn.Module):
    """Sequence forecaster over `leads` steps with GRU history and event-state context."""

    def __init__(self, features, leads, width=96, hidden=96):
        super().__init__()
        self.leads = leads
        weather = weather_columns(features)
        self.register_buffer('weather_index', torch.tensor(weather))
        self.input = nn.Linear(len(features), width)
        self.county = nn.Embedding(41, 16)
        self.context = nn.Linear(width * 2 + 16 + 4, width)
        self.lead = nn.Embedding(leads, 8)
        self.lead_projection = nn.Linear(8, width)
        self.blocks = nn.ModuleList([ResidualTemporalBlock(width, k, .1) for k in [3, 7, 11]])
        self.weather_path = nn.Sequential(nn.LayerNorm(len(weather) * 2), nn.Linear(len(weather) * 2, 48), nn.GELU(), nn.Linear(48, width))
        self.decoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.prop = nn.Linear(width, 1)
        self.count = nn.Linear(width, 1)
        self.gru = nn.GRU(6, hidden, num_layers=2, batch_first=True, dropout=.1)
        self.hist_out = nn.Linear(hidden, width)
        self.state = nn.Sequential(nn.Linear(len(STATE_COLUMNS), 64), nn.GELU(), nn.Linear(64, width))

    def forward(self, x, county, time_features, denominator, input_mask, hist, state):
        """x [N, leads, F] standardised features; returns the predicted outage proportion [N, leads] in [0, 1]."""
        h = self.input(x)
        weight = input_mask.unsqueeze(-1).to(h.dtype)
        mean = (h * weight).sum(1) / weight.sum(1).clamp_min(1)
        maximum = h.masked_fill(~input_mask.unsqueeze(-1), -1e4).amax(1)
        maximum = torch.where(input_mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        context = self.context(torch.cat([mean, maximum, self.county(county), time_features], dim=-1))
        _, g = self.gru(hist)
        context = context + self.hist_out(g[-1]) + self.state(state)
        h = h + context[:, None, :] + self.lead_projection(self.lead(torch.arange(self.leads, device=x.device)))[None, :, :]
        for block in self.blocks:
            h = block(h)
        weather = x.index_select(-1, self.weather_index)
        delta = torch.cat([torch.zeros_like(weather[:, :1]), weather[:, 1:] - weather[:, :-1]], dim=1)
        h = self.decoder(h + self.weather_path(torch.cat([weather, delta], dim=-1)))
        zp = F.softplus(self.prop(h).squeeze(-1)).clamp_max(10)
        zc = F.softplus(self.count(h).squeeze(-1)).clamp_max(14)
        return (.45 * 1e-4 * torch.expm1(zp) + .55 * 10 * torch.expm1(zc) / denominator[:, None]).clamp(0, 1)
