"""Anima-style rank-8 routed branches; shared core owns router and PG mathematics."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from particle_sliders import winning_formulation

TARGET_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                   "attn.to_qkv_mlp_proj", "attn.to_out")


class Branch(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        spec = winning_formulation().spec
        rank = int(spec["adapter_rank"])
        self.down = nn.Linear(inputs, rank, bias=False)
        self.routed = winning_formulation().bridge()
        self.up = nn.Linear(rank, outputs, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, value, particles):
        return self.up(self.routed(self.down(value.float()), particles))


class ParticleAdapter(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        if any(p.requires_grad for p in transformer.parameters()):
            raise ValueError("freeze the native transformer before attaching particles")
        spec = winning_formulation().spec
        self.particles = nn.Parameter(torch.randn(int(spec["parts"]), int(spec["particle_dim"])))
        modules = dict(transformer.named_modules())
        self.names = tuple(name for name, module in modules.items()
                           if isinstance(module, nn.Linear) and name.endswith(TARGET_SUFFIXES))
        if not self.names:
            raise ValueError("no native Klein attention projections found")
        self.branches = nn.ModuleList([Branch(modules[name].in_features, modules[name].out_features)
                                       for name in self.names])
        self.strength = 0.
        self.handles = []
        try:
            for index, name in enumerate(self.names):
                def hook(module, args, output, index=index):
                    if self.strength == 0:
                        return output  # Byte-exact base bypass, including signed zero.
                    delta = self.branches[index](args[0], self.particles) * self.strength
                    return output + delta.to(output.dtype)
                self.handles.append(modules[name].register_forward_hook(hook))
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("finite slider strength required")
        old = self.strength
        self.strength = value
        try:
            yield
        finally:
            self.strength = old

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
