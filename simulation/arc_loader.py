from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
SUPPORTED_ATTN_TYPES = {'MHA', 'GQA', 'MQA'}
SUPPORTED_FFN_TYPES = {'GLU', 'SwiGLU', 'GEGLU', 'MLP'}
SUPPORTED_NORM_TYPES = {'RMSNorm', 'LayerNorm', 'None'}
SUPPORTED_POS_ENC = {'RoPE', 'ALiBi', 'Learned', 'Sinusoidal', 'None'}

class ArchConfigError(ValueError):
    pass

@dataclass
class ArchConfig:
    arch_id: str = 'unnamed'
    arch_name: str = 'unnamed'
    arch_family: str = 'unnamed'
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_layers: int = 12
    seq_len: int = 1024
    batch_size: int = 1
    attn_type: str = 'GQA'
    num_heads: int = 12
    num_kv_heads: int = 12
    head_dim: Optional[int] = None
    window_size: Optional[int] = None
    global_attn_layers: int = 0
    ffn_type: str = 'SwiGLU'
    ffn_multiplier: float = 4.0
    num_experts: int = 1
    top_k_experts: int = 1
    expert_capacity_factor: float = 1.25
    norm_type: str = 'RMSNorm'
    pos_enc: str = 'RoPE'
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0
    optimizer_type: str = 'AdamW'
    optimizer_lr: float = 0.0003
    use_flash_attn: bool = False
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = False
    mixed_precision_dtype: str = 'float32'
    use_tf32: bool = False
    use_torch_compile: bool = False
    param_count: Optional[int] = None
    injection_points: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop('raw', None)
        return d

def _get(d: dict, key: str, default):
    if key not in d:
        return default
    val = d[key]
    return val

def load_arch_config(path: str) -> ArchConfig:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return parse_arch_config(raw)

def parse_arch_config(raw: dict) -> ArchConfig:
    defaults = ArchConfig()
    kwargs = {}
    for fname in defaults.__dataclass_fields__:
        if fname == 'raw':
            continue
        default_val = getattr(defaults, fname)
        kwargs[fname] = _get(raw, fname, default_val)
    kwargs['raw'] = raw
    cfg = ArchConfig(**kwargs)
    _validate(cfg)
    _resolve_derived_fields(cfg)
    return cfg

def _resolve_derived_fields(cfg: ArchConfig) -> None:
    if cfg.head_dim is None:
        if cfg.hidden_dim % cfg.num_heads != 0:
            raise ArchConfigError(f'head_dim was not provided and hidden_dim ({cfg.hidden_dim}) is not evenly divisible by num_heads ({cfg.num_heads}). Provide an explicit head_dim in arc.json if you actually want head_dim * num_heads != hidden_dim (e.g. ARC-AF49 with head_dim=192 != hidden_dim/num_heads).')
        cfg.head_dim = cfg.hidden_dim // cfg.num_heads
    if cfg.attn_type == 'MHA':
        cfg.num_kv_heads = cfg.num_heads
    elif cfg.attn_type == 'MQA':
        cfg.num_kv_heads = 1
    if cfg.window_size is None:
        cfg.window_size = cfg.seq_len

def _validate(cfg: ArchConfig) -> None:
    errors = []
    if cfg.attn_type not in SUPPORTED_ATTN_TYPES:
        errors.append(f"attn_type='{cfg.attn_type}' is not supported. Choices: {sorted(SUPPORTED_ATTN_TYPES)}")
    if cfg.ffn_type not in SUPPORTED_FFN_TYPES:
        errors.append(f"ffn_type='{cfg.ffn_type}' is not supported. Choices: {sorted(SUPPORTED_FFN_TYPES)}")
    if cfg.norm_type not in SUPPORTED_NORM_TYPES:
        errors.append(f"norm_type='{cfg.norm_type}' is not supported. Choices: {sorted(SUPPORTED_NORM_TYPES)}")
    if cfg.pos_enc not in SUPPORTED_POS_ENC:
        errors.append(f"pos_enc='{cfg.pos_enc}' is not supported. Choices: {sorted(SUPPORTED_POS_ENC)}")
    if cfg.num_layers <= 0:
        errors.append('num_layers must be > 0')
    if cfg.hidden_dim <= 0:
        errors.append('hidden_dim must be > 0')
    if cfg.num_heads <= 0:
        errors.append('num_heads must be > 0')
    if cfg.attn_type == 'GQA':
        if cfg.num_kv_heads <= 0:
            errors.append('num_kv_heads must be > 0 for GQA')
        elif cfg.num_heads % cfg.num_kv_heads != 0:
            errors.append(f'For GQA, num_heads ({cfg.num_heads}) must be evenly divisible by num_kv_heads ({cfg.num_kv_heads}).')
    if cfg.head_dim is not None and cfg.head_dim <= 0:
        errors.append('head_dim must be > 0')
    if cfg.global_attn_layers < 0 or cfg.global_attn_layers > cfg.num_layers:
        errors.append('global_attn_layers must be in the range [0, num_layers]')
    if cfg.ffn_multiplier <= 0:
        errors.append('ffn_multiplier must be > 0')
    if cfg.num_experts < 1:
        errors.append('num_experts must be >= 1')
    if cfg.top_k_experts < 1 or cfg.top_k_experts > cfg.num_experts:
        errors.append('top_k_experts must be in the range [1, num_experts]')
    if not 0.0 <= cfg.dropout < 1.0:
        errors.append('dropout must be in the range [0, 1)')
    for ip in cfg.injection_points:
        if 'after_layer' not in ip:
            errors.append(f"injection_point {ip} requires the field 'after_layer'")
            continue
        al = ip['after_layer']
        if not -1 <= al <= cfg.num_layers - 1:
            errors.append(f'injection_point after_layer={al} is outside the valid range [-1, {cfg.num_layers - 1}] (-1 means before the first layer)')
        mode = ip.get('mode', 'parallel')
        if mode not in ('parallel', 'sequential'):
            errors.append(f"injection_point mode='{mode}' is not recognized (use 'parallel'/'sequential')")
    if errors:
        raise ArchConfigError('arc.json is invalid:\n  - ' + '\n  - '.join(errors))
if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else 'arc.json'
    cfg = load_arch_config(p)
    print(json.dumps(cfg.to_dict(), indent=2))