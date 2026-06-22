from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

class ExternalLayerModule(nn.Module, ABC):
    name: str = 'unnamed_external_module'
    version: str = 'v0.0'

    @abstractmethod
    def forward(self, x: torch.Tensor, state: Optional[Any]=None) -> Tuple[torch.Tensor, Optional[Any], Dict[str, Any]]:
        ...

    def reset_state(self) -> None:
        return None

    def describe(self) -> Dict[str, Any]:
        n_params = sum((p.numel() for p in self.parameters()))
        return {'name': self.name, 'version': self.version, 'param_count': n_params, 'class': self.__class__.__name__}

def validate_module_output(module: ExternalLayerModule, x_in: torch.Tensor, output: torch.Tensor) -> None:
    if output.shape != x_in.shape:
        raise RuntimeError(f"External module '{module.name}' ({module.__class__.__name__}) returned output shape {tuple(output.shape)} but the input shape is {tuple(x_in.shape)}. The ExternalLayerModule contract requires output shape == input shape.")