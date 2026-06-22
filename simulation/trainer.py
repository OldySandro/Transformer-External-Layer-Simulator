from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from arc_loader import ArchConfig
from dataset import TextDataset
from module_interface import ExternalLayerModule
from transformer import TransformerStack, build_model

@dataclass
class StepRecord:
    step: int
    loss: float
    grad_norm_total: float
    grad_norm_per_layer: List[float]
    hidden_norm_mean: float
    hidden_std_mean: float
    has_nan: bool
    has_inf: bool
    elapsed_seconds: float
    external_aux_loss: Optional[Dict[str, float]] = None

@dataclass
class TrainingRunResult:
    label: str
    steps: List[StepRecord]
    final_loss: float
    initial_loss: float
    loss_delta: float
    total_seconds: float
    failures_detected: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'label': self.label, 'initial_loss': self.initial_loss, 'final_loss': self.final_loss, 'loss_delta': self.loss_delta, 'total_seconds': self.total_seconds, 'failures_detected': self.failures_detected, 'steps': [{'step': s.step, 'loss': s.loss, 'grad_norm_total': s.grad_norm_total, 'grad_norm_per_layer': s.grad_norm_per_layer, 'hidden_norm_mean': s.hidden_norm_mean, 'hidden_std_mean': s.hidden_std_mean, 'has_nan': s.has_nan, 'has_inf': s.has_inf, 'elapsed_seconds': s.elapsed_seconds, 'external_aux_loss': s.external_aux_loss} for s in self.steps]}
TRAIN_FAILURE_THRESHOLDS = {'grad_norm_explosion_above': 1000.0, 'loss_explosion_above': 10000.0, 'collapse_std_below': 0.0001}

def _validate_injection_points(injection_points: List[dict], num_layers: int, external_modules: Optional[Dict[str, ExternalLayerModule]]) -> None:
    valid_min, valid_max = (-1, num_layers - 1)
    available_names = set(external_modules.keys()) if external_modules else set()
    for ip in injection_points:
        if 'after_layer' not in ip:
            raise ValueError(f"injection_point {ip} requires the field 'after_layer'")
        al = ip['after_layer']
        if not isinstance(al, int):
            raise ValueError(f'after_layer must be an integer, got: {al!r} ({type(al).__name__})')
        if not valid_min <= al <= valid_max:
            raise ValueError(f'after_layer={al} is outside the valid range [{valid_min}, {valid_max}] for an architecture with num_layers={num_layers}. -1 = before the first layer, 0..{valid_max} = after block i.')
        mode = ip.get('mode', 'parallel')
        if mode not in ('parallel', 'sequential'):
            raise ValueError(f"Unrecognized injection mode: '{mode}'. Choices: 'parallel', 'sequential'")
        name = ip.get('name')
        if external_modules is not None and name not in available_names:
            raise KeyError(f"injection_point references module '{name}' but that module is not present in the given external_modules (available: {sorted(available_names)}). Training NOT run -- this kind of mismatch used to pass silently and the module was never actually called.")

class TrainingSession:

    def __init__(self, cfg: ArchConfig, dataset: TextDataset, seed: int, lr: float, device: str='cpu', external_modules: Optional[Dict[str, ExternalLayerModule]]=None, injection_points: Optional[List[dict]]=None):
        self.dataset = dataset
        self.device = torch.device(device)
        self.lr = lr
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.vocab_size = dataset.tokenizer.vocab_size
        if injection_points:
            _validate_injection_points(injection_points, num_layers=cfg_copy.num_layers, external_modules=external_modules)
            cfg_copy.injection_points = list(injection_points)
        else:
            cfg_copy.injection_points = []
        self.cfg = cfg_copy
        self.model = build_model(cfg_copy, external_modules=external_modules, seed=seed).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

    def _grad_norms_per_layer(self) -> List[float]:
        norms = []
        for block in self.model.blocks:
            total_sq = 0.0
            for p in block.parameters():
                if p.grad is not None:
                    total_sq += p.grad.detach().pow(2).sum().item()
            norms.append(total_sq ** 0.5)
        return norms

    def _grad_norm_total(self) -> float:
        total_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().pow(2).sum().item()
        return total_sq ** 0.5

    def run_steps(self, num_steps: int, batch_size: int, seq_len: int, label: str) -> TrainingRunResult:
        self.model.train()
        records: List[StepRecord] = []
        t_start = time.perf_counter()
        for step in range(num_steps):
            t0 = time.perf_counter()
            batch = self.dataset.get_batch(batch_size, seq_len).to(self.device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            self.optimizer.zero_grad()
            out = self.model(input_ids, capture_layers=True, capture_modules=False)
            logits = out['logits']
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            grad_norm_total = self._grad_norm_total()
            grad_norm_per_layer = self._grad_norms_per_layer()
            self.optimizer.step()
            aux_losses_dict = {}
            for name, module in self.model.external_modules.items():
                if hasattr(module, 'compute_aux_loss'):
                    aux_val = module.compute_aux_loss()
                    if aux_val is not None:
                        aux_losses_dict[name] = aux_val
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                final_hidden = out['final_hidden']
                hidden_norm_mean = final_hidden.norm(dim=-1).mean().item()
                hidden_std_mean = final_hidden.std(dim=-1).mean().item()
                loss_val = loss.item()
                has_nan = bool(torch.isnan(logits).any().item()) or loss_val != loss_val
                has_inf = bool(torch.isinf(logits).any().item()) or loss_val in (float('inf'), float('-inf'))
            records.append(StepRecord(step=step, loss=loss_val, grad_norm_total=grad_norm_total, grad_norm_per_layer=grad_norm_per_layer, hidden_norm_mean=hidden_norm_mean, hidden_std_mean=hidden_std_mean, has_nan=has_nan, has_inf=has_inf, elapsed_seconds=elapsed, external_aux_loss=aux_losses_dict if aux_losses_dict else None))
            if has_nan or has_inf:
                break
        total_seconds = time.perf_counter() - t_start
        failures = self._detect_training_failures(records)
        return TrainingRunResult(label=label, steps=records, initial_loss=records[0].loss if records else float('nan'), final_loss=records[-1].loss if records else float('nan'), loss_delta=records[-1].loss - records[0].loss if len(records) > 1 else 0.0, total_seconds=total_seconds, failures_detected=failures)

    @staticmethod
    def _detect_training_failures(records: List[StepRecord]) -> List[Dict[str, Any]]:
        failures = []
        th = TRAIN_FAILURE_THRESHOLDS
        for r in records:
            if r.has_nan or r.has_inf:
                failures.append({'type': 'numerical_instability', 'step': r.step, 'detail': 'logits contain NaN/Inf during training'})
            if r.grad_norm_total > th['grad_norm_explosion_above']:
                failures.append({'type': 'gradient_explosion', 'step': r.step, 'detail': f'grad_norm_total ({r.grad_norm_total:.2e}) is above the threshold {th['grad_norm_explosion_above']:.2e}'})
            if r.loss > th['loss_explosion_above']:
                failures.append({'type': 'loss_explosion', 'step': r.step, 'detail': f'loss ({r.loss:.2e}) is above the threshold {th['loss_explosion_above']:.2e}'})
            if r.hidden_std_mean < th['collapse_std_below']:
                failures.append({'type': 'representation_collapse', 'step': r.step, 'detail': f'hidden state std ({r.hidden_std_mean:.2e}) is below the threshold {th['collapse_std_below']:.2e}'})
        return failures

def run_training_comparison(cfg: ArchConfig, dataset: TextDataset, num_steps: int, batch_size: int, seq_len: int, lr: float, seed: int=42, device: str='cpu', external_modules_factory=None, injection_points: Optional[List[dict]]=None) -> Dict[str, Any]:
    baseline_session = TrainingSession(cfg, dataset, seed=seed, lr=lr, device=device, external_modules=None, injection_points=None)
    baseline_result = baseline_session.run_steps(num_steps, batch_size, seq_len, label='baseline')
    with_module_result = None
    if external_modules_factory is not None:
        external_modules = external_modules_factory()
        with_module_session = TrainingSession(cfg, dataset, seed=seed, lr=lr, device=device, external_modules=external_modules, injection_points=injection_points)
        with_module_result = with_module_session.run_steps(num_steps, batch_size, seq_len, label='with_module')
    return {'baseline': baseline_result, 'with_module': with_module_result}