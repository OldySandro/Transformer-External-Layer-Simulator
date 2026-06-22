# Transformer External Layer Simulator

A research simulator for evaluating the influence, behavior, and stability of external layers within Transformer architectures.

This tool allows researchers to inject custom modules into a Transformer stack and compare the results against a baseline Transformer to measure the actual impact of the external module.

## Features

- Real Transformer simulation
- Configurable Transformer architecture
- Custom external module injection
- Baseline vs module comparison
- Hidden-state drift analysis
- Attention drift analysis
- Output distribution analysis
- Stability evaluation
- Lightweight training evaluation

## Usage

### Baseline

```bash
python run.py --arc arc.json --no-module --out baseline.json
```

### External Module

```bash
python run.py --arc arc.json --module KAC_v0.2 --after-layer 0 --mode parallel --out report.json
```

### Sequential Mode

```bash
python run.py --arc arc.json --module KAC_v0.2 --after-layer 0 --mode sequential --out report.json
```

### Training Mode

```bash
python run.py --arc arc.json --module KAC_v0.2 --after-layer 0 --train yes --train-steps 100 --out train_report.json
```

## External Module Injection

- `--after-layer -1` → before the first Transformer layer
- `--after-layer N` → after layer N

Supported modes:

- `parallel`
- `sequential`

## Research Scope

- Knowledge Layers
- Memory Layers
- Reasoning Layers
- Control Layers
- Experimental Transformer Modules

## Goal

Provide a reproducible environment for testing whether an external layer genuinely influences a Transformer architecture and whether that influence remains stable across execution and training.

### Author 
Oldy Sandro
