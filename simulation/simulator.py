from __future__ import annotations
import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from arc_loader import ArchConfig, load_arch_config
from transformer import TransformerStack, build_model
from module import build_module
from module_interface import ExternalLayerModule, validate_module_output
FAILURE_THRESHOLDS = {'collapse_std_below': 0.0001, 'explosion_norm_above': 10000.0, 'drift_score_high': 1.0, 'throughput_drop_ratio': 0.5}

@dataclass
class SimulationReport:
    arch_id: str
    seed: int
    batch_size: int
    seq_len: int
    timestamp: float
    param_counts: Dict[str, int]
    baseline_run: Dict[str, Any]
    module_run: Optional[Dict[str, Any]]
    comparison: Optional[Dict[str, Any]]
    failures_detected: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    per_layer_drift: Optional[Dict[str, Any]] = None
    sequential_drift: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {'arch_id': self.arch_id, 'seed': self.seed, 'batch_size': self.batch_size, 'seq_len': self.seq_len, 'timestamp': self.timestamp, 'param_counts': self.param_counts, 'baseline_run': self.baseline_run, 'module_run': self.module_run, 'comparison': self.comparison, 'failures_detected': self.failures_detected, 'notes': self.notes, 'per_layer_drift': self.per_layer_drift, 'sequential_drift': self.sequential_drift}

class Simulator:

    def __init__(self, cfg: ArchConfig, seed: int=42, device: str='cpu'):
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device(device)
        self._external_modules: Dict[str, ExternalLayerModule] = {}
        self._pending_injection_points: List[dict] = []

    def attach_module(self, module_key: str, injection_points: List[dict], module_kwargs: Optional[dict]=None) -> None:
        module_kwargs = module_kwargs or {}
        if not injection_points:
            raise ValueError('injection_points must not be empty when attach_module is called')
        names = {ip['name'] for ip in injection_points}
        if len(names) != 1:
            raise ValueError(f"All injection_points within a single attach_module() call must use the same 'name' (one module instance, multiple injection points). Found: {names}")
        name = names.pop()
        valid_min = -1
        valid_max = self.cfg.num_layers - 1
        for ip in injection_points:
            if 'after_layer' not in ip:
                raise ValueError(f"injection_point {ip} requires the field 'after_layer'")
            al = ip['after_layer']
            if not isinstance(al, int):
                raise ValueError(f'after_layer must be an integer, got: {al!r} ({type(al).__name__})')
            if not valid_min <= al <= valid_max:
                raise ValueError(f'after_layer={al} is outside the valid range [{valid_min}, {valid_max}] for an architecture with num_layers={self.cfg.num_layers}. -1 = before the first layer, 0..{valid_max} = after block i. Module NOT attached -- this bug used to pass silently and produce a report that looked valid even though the module was never called.')
            mode = ip.get('mode', 'parallel')
            if mode not in ('parallel', 'sequential'):
                raise ValueError(f"Unrecognized injection mode: '{mode}'. Choices: 'parallel', 'sequential'")
        module = build_module(module_key, hidden_dim=self.cfg.hidden_dim, **module_kwargs)
        module = module.to(self.device)
        self._external_modules[name] = module
        self._pending_injection_points.extend(injection_points)

    def _build_stack(self, with_modules: bool) -> TransformerStack:
        cfg_copy = copy.deepcopy(self.cfg)
        if with_modules:
            cfg_copy.injection_points = list(self._pending_injection_points)
            external = self._external_modules
        else:
            cfg_copy.injection_points = []
            external = {}
        model = build_model(cfg_copy, external_modules=external, seed=self.seed)
        return model.to(self.device)

    def _make_input(self, batch_size: int, seq_len: int) -> torch.Tensor:
        g = torch.Generator(device='cpu').manual_seed(self.seed + 1)
        ids = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len), generator=g)
        return ids.to(self.device)

    def _run_forward(self, model: TransformerStack, input_ids: torch.Tensor, capture_attn_probs: bool=False, capture_hidden_states: bool=False) -> Dict[str, Any]:
        model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids, capture_layers=True, capture_modules=True, capture_attn_probs=capture_attn_probs, capture_hidden_states=capture_hidden_states)
        elapsed = time.perf_counter() - t0
        B, T = input_ids.shape
        tokens_per_second = B * T / max(elapsed, 1e-09)
        logits = out['logits']
        if T > 1:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_targets = input_ids[:, 1:].reshape(-1)
            proxy_loss = torch.nn.functional.cross_entropy(shift_logits, shift_targets).item()
        else:
            proxy_loss = None
        aux_losses = {}
        for name, module in model.external_modules.items():
            if hasattr(module, 'compute_aux_loss'):
                aux_losses[name] = module.compute_aux_loss()
        return {'elapsed_seconds': elapsed, 'tokens_per_second': tokens_per_second, 'proxy_lm_loss': proxy_loss, 'external_aux_losses': aux_losses, 'final_hidden_norm_mean': out['final_hidden'].norm(dim=-1).mean().item(), 'final_hidden_std_mean': out['final_hidden'].std(dim=-1).mean().item(), 'logits_has_nan': bool(torch.isnan(logits).any().item()), 'logits_has_inf': bool(torch.isinf(logits).any().item()), 'layer_traces': out['layer_traces'], 'module_trace': out['module_trace'], '_final_hidden_tensor': out['final_hidden'], '_logits_tensor': logits, '_embedding_snapshot': out['embedding_snapshot']}

    def run(self, batch_size: int=1, seq_len: Optional[int]=None, capture_attn_probs: bool=False, capture_hidden_states: bool=True, per_head_attn_drift: bool=False) -> SimulationReport:
        seq_len = seq_len or min(self.cfg.seq_len, 64)
        if seq_len > self.cfg.seq_len:
            raise ValueError(f'seq_len ({seq_len}) must not exceed cfg.seq_len ({self.cfg.seq_len})')
        input_ids = self._make_input(batch_size, seq_len)
        baseline_model = self._build_stack(with_modules=False)
        baseline_result = self._run_forward(baseline_model, input_ids, capture_attn_probs=capture_attn_probs, capture_hidden_states=capture_hidden_states)
        module_result = None
        comparison = None
        per_layer_drift = None
        sequential_drift = None
        if self._external_modules:
            module_model = self._build_stack(with_modules=True)
            module_result = self._run_forward(module_model, input_ids, capture_attn_probs=capture_attn_probs, capture_hidden_states=capture_hidden_states)
            comparison = self._compare(baseline_result, module_result)
            if capture_hidden_states:
                per_layer_drift = self._compare_per_layer(baseline_result, module_result, capture_attn_probs=capture_attn_probs, per_head_attn_drift=per_head_attn_drift)
                sequential_drift = {'baseline': self._sequential_drift(baseline_result), 'with_module': self._sequential_drift(module_result)}
        param_counts = {'baseline': baseline_model.count_parameters()['total_including_external']}
        if self._external_modules:
            module_model_params = module_model.count_parameters()['total_including_external']
            external_only = sum((sum((p.numel() for p in m.parameters())) for m in self._external_modules.values()))
            param_counts['with_modules'] = module_model_params
            param_counts['external_modules_only'] = external_only
            param_counts['external_modules_pct'] = round(100.0 * external_only / module_model_params, 4)
        failures = self._detect_failures(baseline_result, module_result)
        report = SimulationReport(arch_id=self.cfg.arch_id, seed=self.seed, batch_size=batch_size, seq_len=seq_len, timestamp=time.time(), param_counts=param_counts, baseline_run=self._strip_tensors(baseline_result), module_run=self._strip_tensors(module_result) if module_result else None, comparison=comparison, failures_detected=failures, per_layer_drift=per_layer_drift, sequential_drift=sequential_drift)
        return report

    @staticmethod
    def _strip_tensors(result: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {k: v for k, v in result.items() if not k.startswith('_')}
        cleaned_traces = []
        for trace in cleaned.get('layer_traces', []):
            t = {k: v for k, v in trace.items() if not k.startswith('_')}
            if 'attn' in t and isinstance(t['attn'], dict):
                t['attn'] = {k: v for k, v in t['attn'].items() if not k.startswith('_')}
            cleaned_traces.append(t)
        if 'layer_traces' in cleaned:
            cleaned['layer_traces'] = cleaned_traces
        return cleaned

    def _compare(self, baseline: Dict[str, Any], with_module: Dict[str, Any]) -> Dict[str, Any]:
        h_base = baseline['_final_hidden_tensor']
        h_mod = with_module['_final_hidden_tensor']
        cos_sim = torch.nn.functional.cosine_similarity(h_base.reshape(-1, h_base.shape[-1]), h_mod.reshape(-1, h_mod.shape[-1]), dim=-1)
        drift_score = (1.0 - cos_sim).mean().item()
        baseline_tps = baseline['tokens_per_second']
        module_tps = with_module['tokens_per_second']
        throughput_ratio = module_tps / max(baseline_tps, 1e-09)
        loss_delta = None
        if baseline['proxy_lm_loss'] is not None and with_module['proxy_lm_loss'] is not None:
            loss_delta = with_module['proxy_lm_loss'] - baseline['proxy_lm_loss']
        return {'final_hidden_drift_score': drift_score, 'final_hidden_norm_delta': with_module['final_hidden_norm_mean'] - baseline['final_hidden_norm_mean'], 'final_hidden_std_delta': with_module['final_hidden_std_mean'] - baseline['final_hidden_std_mean'], 'proxy_lm_loss_delta': loss_delta, 'throughput_ratio_module_vs_baseline': throughput_ratio, 'baseline_tokens_per_second': baseline_tps, 'module_tokens_per_second': module_tps}

    @staticmethod
    def _cosine_distance(a: torch.Tensor, b: torch.Tensor, dim: int=-1) -> float:
        cos_sim = torch.nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]) if dim == -1 else a, b.reshape(-1, b.shape[-1]) if dim == -1 else b, dim=-1)
        return (1.0 - cos_sim).mean().item()

    def _compare_per_layer(self, baseline: Dict[str, Any], with_module: Dict[str, Any], capture_attn_probs: bool, per_head_attn_drift: bool) -> Dict[str, Any]:
        base_traces = baseline['layer_traces']
        mod_traces = with_module['layer_traces']
        if len(base_traces) != len(mod_traces):
            raise RuntimeError(f'The number of baseline layers ({len(base_traces)}) and with_module layers ({len(mod_traces)}) do not match -- cannot compare per layer. This indicates a different arc.json was used for the two runs, which should never happen.')
        per_layer = []
        for layer_idx, (bt, mt) in enumerate(zip(base_traces, mod_traces)):
            entry: Dict[str, Any] = {'layer_idx': layer_idx}
            h_base = bt.get('_hidden_state_tensor')
            h_mod = mt.get('_hidden_state_tensor')
            if h_base is not None and h_mod is not None:
                entry['hidden_state_drift_score'] = self._cosine_distance(h_base, h_mod)
                entry['hidden_norm_delta'] = mt['hidden_norm_mean'] - bt['hidden_norm_mean']
                entry['hidden_std_delta'] = mt['hidden_std_mean'] - bt['hidden_std_mean']
            else:
                entry['hidden_state_drift_score'] = None
                entry['hidden_norm_delta'] = mt['hidden_norm_mean'] - bt['hidden_norm_mean']
                entry['hidden_std_delta'] = mt['hidden_std_mean'] - bt['hidden_std_mean']
            base_attn = bt.get('attn', {}) or {}
            mod_attn = mt.get('attn', {}) or {}
            if 'attn_entropy_mean' in base_attn and 'attn_entropy_mean' in mod_attn:
                entry['attn_entropy_delta'] = mod_attn['attn_entropy_mean'] - base_attn['attn_entropy_mean']
            else:
                entry['attn_entropy_delta'] = None
            if 'attn_max_prob_mean' in base_attn and 'attn_max_prob_mean' in mod_attn:
                entry['attn_max_prob_delta'] = mod_attn['attn_max_prob_mean'] - base_attn['attn_max_prob_mean']
            else:
                entry['attn_max_prob_delta'] = None
            entry['attn_prob_drift_score'] = None
            entry['attn_prob_drift_per_head'] = None
            if capture_attn_probs:
                p_base = base_attn.get('_attn_probs_tensor')
                p_mod = mod_attn.get('_attn_probs_tensor')
                if p_base is not None and p_mod is not None:
                    entry['attn_prob_drift_score'] = self._cosine_distance(p_base, p_mod)
                    if per_head_attn_drift:
                        num_heads = p_base.shape[1]
                        per_head = []
                        for h in range(num_heads):
                            per_head.append(self._cosine_distance(p_base[:, h], p_mod[:, h]))
                        entry['attn_prob_drift_per_head'] = per_head
            per_layer.append(entry)
        logits_base = baseline.get('_logits_tensor')
        logits_mod = with_module.get('_logits_tensor')
        logits_drift = None
        if logits_base is not None and logits_mod is not None:
            probs_base = torch.nn.functional.softmax(logits_base, dim=-1)
            probs_mod = torch.nn.functional.softmax(logits_mod, dim=-1)
            logits_drift = self._cosine_distance(probs_base, probs_mod)
        return {'per_layer': per_layer, 'logits_distribution_drift_score': logits_drift}

    def _sequential_drift(self, run_result: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        traces = run_result['layer_traces']
        embedding_snapshot = run_result.get('_embedding_snapshot')
        if embedding_snapshot is None or not traces or traces[0].get('_hidden_state_tensor') is None:
            return None
        sequential = []
        prev_hidden = embedding_snapshot
        for trace in traces:
            h = trace['_hidden_state_tensor']
            drift = self._cosine_distance(prev_hidden, h)
            sequential.append({'layer_idx': trace['layer_idx'], 'drift_from_previous': drift})
            prev_hidden = h
        return sequential

    def _detect_failures(self, baseline: Dict[str, Any], with_module: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failures = []
        th = FAILURE_THRESHOLDS

        def check_run(label: str, result: Dict[str, Any]):
            if result['logits_has_nan'] or result['logits_has_inf']:
                failures.append({'type': 'numerical_instability', 'run': label, 'detail': 'logits contain NaN/Inf', 'logits_has_nan': result['logits_has_nan'], 'logits_has_inf': result['logits_has_inf']})
            if result['final_hidden_std_mean'] < th['collapse_std_below']:
                failures.append({'type': 'representation_collapse', 'run': label, 'detail': f'final hidden state std ({result['final_hidden_std_mean']:.2e}) is below the threshold {th['collapse_std_below']:.2e}', 'final_hidden_std_mean': result['final_hidden_std_mean']})
            if result['final_hidden_norm_mean'] > th['explosion_norm_above']:
                failures.append({'type': 'representation_explosion', 'run': label, 'detail': f'final hidden state norm ({result['final_hidden_norm_mean']:.2e}) is above the threshold {th['explosion_norm_above']:.2e}', 'final_hidden_norm_mean': result['final_hidden_norm_mean']})
            for layer_trace in result['layer_traces']:
                if layer_trace['has_nan'] or layer_trace['has_inf']:
                    failures.append({'type': 'numerical_instability', 'run': label, 'detail': f'layer {layer_trace['layer_idx']} produced NaN/Inf', 'layer_idx': layer_trace['layer_idx']})
        check_run('baseline', baseline)
        if with_module is not None:
            check_run('with_module', with_module)
            for mod_trace in with_module['module_trace']:
                if mod_trace.get('output_has_nan'):
                    failures.append({'type': 'numerical_instability', 'run': 'with_module', 'detail': f"module '{mod_trace['module_name']}' at after_layer={mod_trace['after_layer']} produced a NaN output", 'module_name': mod_trace['module_name']})
                tps_module_only = mod_trace.get('tokens_per_second_module_only')
                if tps_module_only is not None and baseline['tokens_per_second'] > 0:
                    ratio = tps_module_only / baseline['tokens_per_second']
                    if ratio < th['throughput_drop_ratio']:
                        failures.append({'type': 'throughput_degradation', 'run': 'with_module', 'detail': f"module '{mod_trace['module_name']}' is much slower than the overall baseline throughput ({tps_module_only:.0f} tok/s vs baseline {baseline['tokens_per_second']:.0f} tok/s)", 'module_name': mod_trace['module_name'], 'module_tokens_per_second': tps_module_only, 'baseline_tokens_per_second': baseline['tokens_per_second']})
        return failures

    @staticmethod
    def save_report(report: SimulationReport, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2)

def print_report_summary(report: SimulationReport) -> None:
    print('=' * 70)
    print(f'SIMULATION REPORT  arch_id={report.arch_id}  seed={report.seed}')
    print('=' * 70)
    print(f'batch_size       = {report.batch_size}')
    print(f'seq_len          = {report.seq_len}')
    print(f'param.baseline   = {report.param_counts.get('baseline')}')
    if 'with_modules' in report.param_counts:
        print(f'param.with_mod   = {report.param_counts.get('with_modules')}')
        print(f'param.ext_only   = {report.param_counts.get('external_modules_only')}')
        print(f'param.ext_pct    = {report.param_counts.get('external_modules_pct')}%')
    print('-' * 70)
    print('[BASELINE RUN]')
    for k, v in report.baseline_run.items():
        if k in ('layer_traces', 'module_trace'):
            continue
        print(f'  {k:32s} = {v}')
    if report.module_run:
        print('-' * 70)
        print('[MODULE RUN]')
        for k, v in report.module_run.items():
            if k in ('layer_traces', 'module_trace'):
                continue
            print(f'  {k:32s} = {v}')
    if report.comparison:
        print('-' * 70)
        print('[COMPARISON: module vs baseline]')
        for k, v in report.comparison.items():
            print(f'  {k:32s} = {v}')
    if report.per_layer_drift:
        print('-' * 70)
        print('[PER-LAYER DRIFT: module vs baseline]')
        logits_drift = report.per_layer_drift.get('logits_distribution_drift_score')
        print(f'  logits_distribution_drift_score = {logits_drift}')
        for entry in report.per_layer_drift.get('per_layer', []):
            line = f'  layer={entry['layer_idx']:<3d} hidden_drift={_fmt(entry.get('hidden_state_drift_score')):<10s} attn_entropy_delta={_fmt(entry.get('attn_entropy_delta')):<10s} '
            if entry.get('attn_prob_drift_score') is not None:
                line += f'attn_prob_drift={_fmt(entry.get('attn_prob_drift_score')):<10s} '
            print(line)
    if report.sequential_drift:
        print('-' * 70)
        print('[SEQUENTIAL DRIFT: layer-to-layer, per run]')
        for label in ('baseline', 'with_module'):
            seq = report.sequential_drift.get(label)
            if not seq:
                continue
            print(f'  [{label}]')
            for entry in seq:
                print(f'    layer={entry['layer_idx']:<3d} drift_from_previous={_fmt(entry['drift_from_previous'])}')
    print('-' * 70)
    print(f'[FAILURES DETECTED: {len(report.failures_detected)}]')
    for fail in report.failures_detected:
        print(f'  - type={fail['type']:28s} run={fail.get('run', '-'):12s} detail={fail['detail']}')
    print('=' * 70)

def _fmt(v) -> str:
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        return f'{v:.6f}'
    return str(v)