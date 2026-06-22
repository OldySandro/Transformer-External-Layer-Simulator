from __future__ import annotations
import math
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_loader import ArchConfig

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return x_norm * self.weight

class IdentityNorm(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def build_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == 'RMSNorm':
        return RMSNorm(dim)
    if norm_type == 'LayerNorm':
        return nn.LayerNorm(dim)
    if norm_type == 'None':
        return IdentityNorm()
    raise ValueError(f'Unrecognized norm_type: {norm_type}')

def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    assert head_dim % 2 == 0, 'RoPE requires an even head_dim'
    half = head_dim // 2
    freqs = 1.0 / theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return (cos, sin)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[-2]
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[:seq_len].view(1, 1, seq_len, half)
    s = sin[:seq_len].view(1, 1, seq_len, half)
    rotated_x1 = x1 * c - x2 * s
    rotated_x2 = x1 * s + x2 * c
    return torch.cat([rotated_x1, rotated_x2], dim=-1)

def build_alibi_bias(num_heads: int, seq_len: int, device, dtype) -> torch.Tensor:

    def get_slopes(n):

        def slopes_power_of_2(n):
            start = 2 ** (-2 ** (-(math.log2(n) - 3)))
            return [start * start ** i for i in range(n)]
        if math.log2(n).is_integer():
            return slopes_power_of_2(n)
        closest = 2 ** math.floor(math.log2(n))
        base = slopes_power_of_2(closest)
        extra = slopes_power_of_2(2 * closest)[0::2][:n - closest]
        return base + extra
    slopes = torch.tensor(get_slopes(num_heads), dtype=dtype, device=device)
    pos = torch.arange(seq_len, device=device)
    rel = (pos.view(1, -1) - pos.view(-1, 1)).clamp(max=0).to(dtype)
    bias = rel.view(1, seq_len, seq_len) * slopes.view(num_heads, 1, 1)
    return bias

def build_sinusoidal_pe(seq_len: int, dim: int, device, dtype) -> torch.Tensor:
    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(dtype)

class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.window_size = cfg.window_size
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_dim, q_out, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, kv_out, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, cfg.hidden_dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor, rope_cache=None, alibi_bias: Optional[torch.Tensor]=None, is_global_layer: bool=False, capture: Optional[Dict[str, Any]]=None, capture_attn_probs: bool=False) -> torch.Tensor:
        B, T, C = x.shape
        H, KVH, D = (self.num_heads, self.num_kv_heads, self.head_dim)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, KVH, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KVH, D).transpose(1, 2)
        if rope_cache is not None:
            cos, sin = rope_cache
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if KVH != H:
            repeat_factor = H // KVH
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.view(1, 1, T, T), float('-inf'))
        effective_window = self.window_size
        if not is_global_layer and effective_window is not None and (effective_window < T):
            idx = torch.arange(T, device=x.device)
            rel = idx.view(T, 1) - idx.view(1, T)
            window_mask = rel >= effective_window
            scores = scores.masked_fill(window_mask.view(1, 1, T, T), float('-inf'))
        if alibi_bias is not None:
            scores = scores + alibi_bias.unsqueeze(0)
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        if capture is not None:
            with torch.no_grad():
                p = attn_probs.clamp_min(1e-12)
                entropy = -(p * p.log()).sum(dim=-1)
                capture['attn_entropy_mean'] = entropy.mean().item()
                capture['attn_entropy_per_head'] = entropy.mean(dim=(0, 2)).tolist()
                capture['attn_max_prob_mean'] = attn_probs.max(dim=-1).values.mean().item()
                if capture_attn_probs:
                    capture['_attn_probs_tensor'] = attn_probs.detach().clone()
        if self.dropout_p > 0.0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.dropout_p)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.o_proj(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.ffn_type = cfg.ffn_type
        hidden = int(cfg.hidden_dim * cfg.ffn_multiplier)
        if cfg.ffn_type == 'MLP':
            self.fc1 = nn.Linear(cfg.hidden_dim, hidden, bias=False)
            self.fc2 = nn.Linear(hidden, cfg.hidden_dim, bias=False)
        else:
            self.gate_proj = nn.Linear(cfg.hidden_dim, hidden, bias=False)
            self.up_proj = nn.Linear(cfg.hidden_dim, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == 'MLP':
            return self.fc2(F.gelu(self.fc1(x)))
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.ffn_type == 'SwiGLU':
            act_gate = F.silu(gate)
        elif self.ffn_type == 'GEGLU':
            act_gate = F.gelu(gate)
        elif self.ffn_type == 'GLU':
            act_gate = torch.sigmoid(gate)
        else:
            raise ValueError(f'Unrecognized ffn_type: {self.ffn_type}')
        return self.down_proj(act_gate * up)

class MoEFeedForward(nn.Module):

    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k_experts
        self.capacity_factor = cfg.expert_capacity_factor
        self.experts = nn.ModuleList([FeedForward(cfg) for _ in range(cfg.num_experts)])
        self.router = nn.Linear(cfg.hidden_dim, cfg.num_experts, bias=False)

    def forward(self, x: torch.Tensor, capture: Optional[Dict[str, Any]]=None) -> torch.Tensor:
        B, T, C = x.shape
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-09)
        flat_idx = topk_idx.view(-1, self.top_k)
        flat_prob = topk_probs.view(-1, self.top_k)
        flat_x = x.view(-1, C)
        flat_out = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            for k in range(self.top_k):
                mask = flat_idx[:, k] == e
                if mask.any():
                    contribution = self.experts[e](flat_x[mask]) * flat_prob[mask, k].unsqueeze(-1)
                    flat_out[mask] += contribution
        out = flat_out.view(B, T, C)
        if capture is not None:
            with torch.no_grad():
                usage = torch.zeros(self.num_experts, device=x.device)
                for e in range(self.num_experts):
                    usage[e] = (topk_idx == e).float().sum()
                usage = usage / usage.sum().clamp_min(1.0)
                capture['expert_usage'] = usage.tolist()
        return out

class TransformerBlock(nn.Module):

    def __init__(self, cfg: ArchConfig, layer_idx: int, is_global_layer: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global_layer = is_global_layer
        self.norm1 = build_norm(cfg.norm_type, cfg.hidden_dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = build_norm(cfg.norm_type, cfg.hidden_dim)
        if cfg.num_experts > 1:
            self.ffn = MoEFeedForward(cfg)
            self.is_moe = True
        else:
            self.ffn = FeedForward(cfg)
            self.is_moe = False

    def forward(self, x: torch.Tensor, rope_cache=None, alibi_bias: Optional[torch.Tensor]=None, capture: Optional[Dict[str, Any]]=None, capture_attn_probs: bool=False, capture_hidden_states: bool=False) -> torch.Tensor:
        attn_capture = {} if capture is not None else None
        attn_out = self.attn(self.norm1(x), rope_cache=rope_cache, alibi_bias=alibi_bias, is_global_layer=self.is_global_layer, capture=attn_capture, capture_attn_probs=capture_attn_probs)
        x = x + attn_out
        ffn_capture = {} if capture is not None else None
        normed = self.norm2(x)
        if self.is_moe:
            ffn_out = self.ffn(normed, capture=ffn_capture)
        else:
            ffn_out = self.ffn(normed)
        x = x + ffn_out
        if capture is not None:
            capture['layer_idx'] = self.layer_idx
            capture['attn'] = attn_capture
            capture['ffn'] = ffn_capture
            with torch.no_grad():
                capture['hidden_norm_mean'] = x.norm(dim=-1).mean().item()
                capture['hidden_std_mean'] = x.std(dim=-1).mean().item()
                capture['has_nan'] = bool(torch.isnan(x).any().item())
                capture['has_inf'] = bool(torch.isinf(x).any().item())
                if capture_hidden_states:
                    capture['_hidden_state_tensor'] = x.detach().clone()
        return x

class TransformerStack(nn.Module):

    def __init__(self, cfg: ArchConfig, external_modules: Optional[Dict[str, nn.Module]]=None):
        super().__init__()
        self.cfg = cfg
        self.external_modules = nn.ModuleDict(external_modules or {})
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.use_learned_pe = cfg.pos_enc == 'Learned'
        if self.use_learned_pe:
            self.pos_emb = nn.Embedding(cfg.seq_len, cfg.hidden_dim)
        n_global = cfg.global_attn_layers
        self.blocks = nn.ModuleList([TransformerBlock(cfg, layer_idx=i, is_global_layer=i >= cfg.num_layers - n_global) for i in range(cfg.num_layers)])
        self.final_norm = build_norm(cfg.norm_type, cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self._rope_cache = None
        self._alibi_bias = None
        self._sinusoidal_pe = None
        self._injections_by_layer: Dict[int, List[dict]] = {}
        valid_min, valid_max = (-1, cfg.num_layers - 1)
        for ip in cfg.injection_points:
            al = ip['after_layer']
            if not valid_min <= al <= valid_max:
                raise ValueError(f'injection_point after_layer={al} is outside the valid range [{valid_min}, {valid_max}] for num_layers={cfg.num_layers}. Model NOT built -- this prevents an injection that would silently never get called.')
            name = ip.get('name')
            if name not in self.external_modules:
                raise KeyError(f"injection_point references module '{name}' but that module is not present in external_modules (available: {sorted(self.external_modules.keys())}). Model NOT built -- this prevents an injection that would silently never get called.")
            self._injections_by_layer.setdefault(al, []).append(ip)

    def _ensure_pos_caches(self, device, dtype):
        cfg = self.cfg
        if cfg.pos_enc == 'RoPE' and self._rope_cache is None:
            self._rope_cache = build_rope_cache(cfg.seq_len, cfg.head_dim, cfg.rope_theta, device, dtype)
        if cfg.pos_enc == 'ALiBi' and self._alibi_bias is None:
            self._alibi_bias = build_alibi_bias(cfg.num_heads, cfg.seq_len, device, dtype)
        if cfg.pos_enc == 'Sinusoidal' and self._sinusoidal_pe is None:
            self._sinusoidal_pe = build_sinusoidal_pe(cfg.seq_len, cfg.hidden_dim, device, dtype)

    def _run_injection(self, after_layer: int, x: torch.Tensor, module_state: Dict[str, Any], trace: Optional[List[dict]]) -> torch.Tensor:
        injections = self._injections_by_layer.get(after_layer, [])
        for ip in injections:
            name = ip['name']
            mode = ip.get('mode', 'parallel')
            if name not in self.external_modules:
                raise KeyError(f"injection_point references module '{name}' but that module is not present in the external_modules given to TransformerStack. Check module.py / simulator.py.")
            module = self.external_modules[name]
            prev_state = module_state.get(name)
            module_out, new_state, mod_log = module(x, prev_state)
            module_state[name] = new_state
            if mode == 'parallel':
                x = x + module_out
            elif mode == 'sequential':
                x = module_out
            else:
                raise ValueError(f'Unrecognized injection mode: {mode}')
            if trace is not None:
                mod_log = dict(mod_log)
                mod_log['module_name'] = name
                mod_log['after_layer'] = after_layer
                mod_log['mode'] = mode
                with torch.no_grad():
                    mod_log['output_norm_mean'] = x.norm(dim=-1).mean().item()
                    mod_log['output_has_nan'] = bool(torch.isnan(x).any().item())
                trace.append(mod_log)
        return x

    def forward(self, input_ids: torch.Tensor, module_state: Optional[Dict[str, Any]]=None, capture_layers: bool=True, capture_modules: bool=True, capture_attn_probs: bool=False, capture_hidden_states: bool=False) -> Dict[str, Any]:
        cfg = self.cfg
        device = input_ids.device
        dtype = self.tok_emb.weight.dtype
        self._ensure_pos_caches(device, dtype)
        x = self.tok_emb(input_ids)
        if self.use_learned_pe:
            T = input_ids.shape[1]
            pos_ids = torch.arange(T, device=device)
            x = x + self.pos_emb(pos_ids).unsqueeze(0)
        elif cfg.pos_enc == 'Sinusoidal':
            T = input_ids.shape[1]
            x = x + self._sinusoidal_pe[:T].unsqueeze(0)
        if module_state is None:
            module_state = {}
        layer_traces: List[dict] = []
        module_trace: List[dict] = []
        x = self._run_injection(-1, x, module_state, module_trace if capture_modules else None)
        embedding_snapshot = x.detach().clone() if capture_hidden_states else None
        rope_cache = self._rope_cache if cfg.pos_enc == 'RoPE' else None
        alibi = None
        if cfg.pos_enc == 'ALiBi':
            T = input_ids.shape[1]
            alibi = self._alibi_bias[:, :T, :T]
        for i, block in enumerate(self.blocks):
            block_capture = {} if capture_layers else None
            x = block(x, rope_cache=rope_cache, alibi_bias=alibi, capture=block_capture, capture_attn_probs=capture_attn_probs, capture_hidden_states=capture_hidden_states)
            if capture_layers:
                layer_traces.append(block_capture)
            x = self._run_injection(i, x, module_state, module_trace if capture_modules else None)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return {'logits': logits, 'final_hidden': x, 'layer_traces': layer_traces, 'module_trace': module_trace, 'module_state': module_state, 'embedding_snapshot': embedding_snapshot}

    def count_parameters(self) -> Dict[str, int]:
        total = sum((p.numel() for p in self.parameters()))
        return {'total_including_external': total}

def build_model(cfg: ArchConfig, external_modules: Optional[Dict[str, nn.Module]]=None, seed: int=42) -> TransformerStack:
    torch.manual_seed(seed)
    model = TransformerStack(cfg, external_modules=external_modules)
    return model