from __future__ import annotations
import argparse
import sys
from arc_loader import load_arch_config, ArchConfigError
from simulator import Simulator, print_report_summary

def main():
    parser = argparse.ArgumentParser(description='Transformer + External Module Simulator (forward-pass OR real training loop)')
    parser.add_argument('--arc', type=str, required=True, help='Path to arc.json')
    parser.add_argument('--module', type=str, default=None, help='Module key from MODULE_REGISTRY in module.py, e.g. KAC_v0.1 / KAC_v0.2')
    parser.add_argument('--no-module', action='store_true', help='Run baseline only, without an external module')
    parser.add_argument('--after-layer', type=int, default=0, help='Injection point: -1 = before the first layer, i = after block i')
    parser.add_argument('--mode', type=str, default='parallel', choices=['parallel', 'sequential'], help='Injection mode')
    parser.add_argument('--module-name', type=str, default='KAC', help='Module instance name (used as the state key & log label)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=None, help='Default: min(cfg.seq_len, 64)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default='report.json')
    parser.add_argument('--capture-attn-probs', action='store_true', help='Enable per-layer attention probability matrix drift (heavier on memory, default OFF)')
    parser.add_argument('--per-head-attn-drift', action='store_true', help='When --capture-attn-probs is on, also store per-head drift (not just the layer average)')
    parser.add_argument('--no-hidden-drift', action='store_true', help='Disable per-layer hidden-state drift & sequential drift (default ON, relatively cheap in memory)')
    parser.add_argument('--train', type=str, default='no', choices=['yes', 'no'], help='yes = run a REAL training loop (backward+optimizer.step()) on a small dataset, instead of just a forward pass')
    parser.add_argument('--train-steps', type=int, default=50, help='Number of training steps (used if --train yes)')
    parser.add_argument('--lr', type=float, default=None, help='Override the learning rate (default: cfg.optimizer_lr from arc.json)')
    parser.add_argument('--data', type=str, default=None, help='Path to a .txt file for the training dataset. Default: the built-in sample text.')
    parser.add_argument('--max-chars', type=int, default=20000, help='Truncate the training dataset to this many characters so training stays fast')
    args = parser.parse_args()
    try:
        cfg = load_arch_config(args.arc)
    except ArchConfigError as e:
        print(f'[FATAL] arc.json is invalid:\n{e}', file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f'[FATAL] File not found: {args.arc}', file=sys.stderr)
        sys.exit(1)
    has_module = bool(args.module) and (not args.no_module)
    try:
        if args.train == 'yes':
            run_training_mode(args, cfg, has_module)
        else:
            run_simulation_mode(args, cfg, has_module)
    except (ValueError, KeyError) as e:
        print(f'[FATAL] Invalid module/injection configuration:\n{e}', file=sys.stderr)
        sys.exit(1)

def run_simulation_mode(args, cfg, has_module: bool) -> None:
    sim = Simulator(cfg, seed=args.seed, device=args.device)
    if has_module:
        sim.attach_module(args.module, injection_points=[{'after_layer': args.after_layer, 'mode': args.mode, 'name': args.module_name}])
    report = sim.run(batch_size=args.batch_size, seq_len=args.seq_len, capture_attn_probs=args.capture_attn_probs, capture_hidden_states=not args.no_hidden_drift, per_head_attn_drift=args.per_head_attn_drift)
    print_report_summary(report)
    Simulator.save_report(report, args.out)
    print(f'\n[saved] full report (including per-layer trace) -> {args.out}')

def run_training_mode(args, cfg, has_module: bool) -> None:
    import json
    from dataset import load_dataset
    from module import build_module
    from trainer import run_training_comparison, TrainingRunResult
    dataset = load_dataset(args.data, seed=args.seed, max_chars=args.max_chars)
    print(f'[dataset] {len(dataset.token_ids)} tokens, vocab_size={dataset.tokenizer.vocab_size} (character-level)')
    seq_len = args.seq_len or min(cfg.seq_len, 64, len(dataset.token_ids) - 1)
    lr = args.lr if args.lr is not None else cfg.optimizer_lr
    external_modules_factory = None
    injection_points = None
    if has_module:
        injection_points = [{'after_layer': args.after_layer, 'mode': args.mode, 'name': args.module_name}]

        def factory():
            return {args.module_name: build_module(args.module, hidden_dim=cfg.hidden_dim)}
        external_modules_factory = factory
    print(f'[train] steps={args.train_steps} batch_size={args.batch_size} seq_len={seq_len} lr={lr}')
    print(f'[train] module={('NONE (baseline only)' if not has_module else args.module)} {('' if not has_module else f'(after_layer={args.after_layer}, mode={args.mode})')}')
    results = run_training_comparison(cfg=cfg, dataset=dataset, num_steps=args.train_steps, batch_size=args.batch_size, seq_len=seq_len, lr=lr, seed=args.seed, device=args.device, external_modules_factory=external_modules_factory, injection_points=injection_points)
    print_training_summary(results)
    out_dict = {'arch_id': cfg.arch_id, 'train_steps': args.train_steps, 'batch_size': args.batch_size, 'seq_len': seq_len, 'lr': lr, 'dataset_tokens': len(dataset.token_ids), 'vocab_size': dataset.tokenizer.vocab_size, 'baseline': results['baseline'].to_dict(), 'with_module': results['with_module'].to_dict() if results['with_module'] else None}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2)
    print(f'\n[saved] full training report (loss/grad/hidden per step) -> {args.out}')

def print_training_summary(results: dict) -> None:
    print('=' * 70)
    print('TRAINING REPORT (real loop: backward + optimizer.step())')
    print('=' * 70)
    for label in ('baseline', 'with_module'):
        result = results.get(label)
        if result is None:
            continue
        print(f'[{label.upper()}]')
        print(f'  initial_loss   = {result.initial_loss:.6f}')
        print(f'  final_loss     = {result.final_loss:.6f}')
        print(f'  loss_delta     = {result.loss_delta:.6f}')
        print(f'  total_seconds  = {result.total_seconds:.4f}')
        print(f'  n_steps_completed = {len(result.steps)}')
        if result.failures_detected:
            print(f'  failures_detected = {len(result.failures_detected)}')
            for f in result.failures_detected[:5]:
                print(f'    - step={f['step']:<4d} type={f['type']:<24s} {f['detail']}')
        print('-' * 70)
    base = results.get('baseline')
    mod = results.get('with_module')
    if base and mod:
        print('[COMPARISON]')
        print(f'  final_loss_delta (with_module - baseline) = {mod.final_loss - base.final_loss:.6f}')
        print(f'  loss_delta_delta (whether the module speeds up/slows down the loss decrease) = {mod.loss_delta - base.loss_delta:.6f}')
        if base.steps and mod.steps:
            avg_grad_base = sum((s.grad_norm_total for s in base.steps)) / len(base.steps)
            avg_grad_mod = sum((s.grad_norm_total for s in mod.steps)) / len(mod.steps)
            print(f'  avg_grad_norm_total baseline   = {avg_grad_base:.6f}')
            print(f'  avg_grad_norm_total with_module = {avg_grad_mod:.6f}')
    print('=' * 70)
if __name__ == '__main__':
    main()