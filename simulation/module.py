from __future__ import annotations
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from module_interface import ExternalLayerModule

class KAC_v0_1_GRULoop(ExternalLayerModule):
    name = 'KAC'
    version = 'v0.1-gru-loop'

    def __init__(self, hidden_dim: int, kac_dim: int=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kac_dim = kac_dim
        self.in_proj = nn.Linear(hidden_dim, kac_dim, bias=False)
        self.gru_cell = nn.GRUCell(kac_dim, kac_dim)
        self.out_proj = nn.Linear(kac_dim, hidden_dim, bias=False)
        self.aux_head = nn.Linear(kac_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        B, T, C = x.shape
        device = x.device
        h = state if state is not None else torch.zeros(B, self.kac_dim, device=device, dtype=x.dtype)
        x_proj = self.in_proj(x)
        outputs = []
        t0 = time.perf_counter()
        for t in range(T):
            h = self.gru_cell(x_proj[:, t, :], h)
            outputs.append(h)
        elapsed = time.perf_counter() - t0
        kac_states = torch.stack(outputs, dim=1)
        output = self.out_proj(kac_states)
        log = {'loop_seconds': elapsed, 'tokens_per_second_module_only': B * T / max(elapsed, 1e-09), 'kac_state_norm_mean': kac_states.norm(dim=-1).mean().item(), 'kac_state_std_mean': kac_states.std(dim=-1).mean().item()}
        self._last_kac_states = kac_states.detach()
        self._last_input = x.detach()
        return (output, h.detach(), log)

    def compute_aux_loss(self) -> Optional[float]:
        if not hasattr(self, '_last_kac_states'):
            return None
        kac_states = self._last_kac_states
        target_emb = self._last_input
        if kac_states.shape[1] < 2:
            return None
        pred = self.aux_head(kac_states[:, :-1, :])
        target = target_emb[:, 1:, :]
        loss = F.mse_loss(pred, target)
        return loss.item()

class KAC_v0_2_CausalConv(ExternalLayerModule):
    name = 'KAC'
    version = 'v0.2-causal-conv'

    def __init__(self, hidden_dim: int, kac_dim: int=128, kernel_size: int=4, num_conv_layers: int=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kac_dim = kac_dim
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(hidden_dim, kac_dim, bias=False)
        self.conv_layers = nn.ModuleList([nn.Conv1d(kac_dim, kac_dim, kernel_size=kernel_size, padding=0) for _ in range(num_conv_layers)])
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(kac_dim, hidden_dim, bias=False)
        self.aux_head = nn.Linear(kac_dim, hidden_dim, bias=False)

    def _causal_pad(self, x_ch_first: torch.Tensor) -> torch.Tensor:
        pad_amount = self.kernel_size - 1
        return F.pad(x_ch_first, (pad_amount, 0))

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        B, T, C = x.shape
        x_proj = self.in_proj(x)
        h = x_proj.transpose(1, 2)
        t0 = time.perf_counter()
        for conv in self.conv_layers:
            h = self._causal_pad(h)
            h = conv(h)
            h = self.activation(h)
        elapsed = time.perf_counter() - t0
        kac_states = h.transpose(1, 2)
        output = self.out_proj(kac_states)
        carry_len = self.kernel_size - 1
        new_state = x_proj[:, -carry_len:, :].detach() if carry_len > 0 else None
        log = {'loop_seconds': elapsed, 'tokens_per_second_module_only': B * T / max(elapsed, 1e-09), 'kac_state_norm_mean': kac_states.norm(dim=-1).mean().item(), 'kac_state_std_mean': kac_states.std(dim=-1).mean().item()}
        self._last_kac_states = kac_states.detach()
        self._last_input = x.detach()
        return (output, new_state, log)

    def compute_aux_loss(self) -> Optional[float]:
        if not hasattr(self, '_last_kac_states'):
            return None
        kac_states = self._last_kac_states
        target_emb = self._last_input
        if kac_states.shape[1] < 2:
            return None
        pred = self.aux_head(kac_states[:, :-1, :])
        target = target_emb[:, 1:, :]
        loss = F.mse_loss(pred, target)
        return loss.item()
MODULE_REGISTRY = {'KAC_v0.1': KAC_v0_1_GRULoop, 'KAC_v0.2': KAC_v0_2_CausalConv}

def get_module_class(key: str):
    if key not in MODULE_REGISTRY:
        raise KeyError(f"Module '{key}' not found in MODULE_REGISTRY. Available choices: {sorted(MODULE_REGISTRY.keys())}. Add a new class in module.py and register it in MODULE_REGISTRY for new version/research direction experiments.")
    return MODULE_REGISTRY[key]

def build_module(key: str, hidden_dim: int, **kwargs) -> ExternalLayerModule:
    cls = get_module_class(key)
    return cls(hidden_dim=hidden_dim, **kwargs)