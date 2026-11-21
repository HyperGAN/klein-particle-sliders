"""Opt-in strength changes on the original Klein finished-image consumer."""

from contextlib import ExitStack
from itertools import pairwise
from math import isfinite
from sys import float_info

from .render import render_image


def resolve(strength, steps, step_gains):
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
        raise ValueError("Klein requires one gain per native scheduler step")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Klein step gains must be finite and nonnegative")
    strengths = [strength * gain for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("Klein effective slider strengths must be in [0,1]")
    return strengths


def render(runtime, adapter, *, step_gains, strength, steps=8, **request):
    values = resolve(strength, steps, step_gains)
    position = 0
    sigmas = None
    scope = ExitStack()

    def before(_model, _args):
        nonlocal position, sigmas
        if position >= steps:
            raise RuntimeError("Klein made extra native denoiser calls")
        if sigmas is None:
            sigmas = runtime.pipe.scheduler.sigmas.detach().cpu().tolist()
            if len(sigmas) != steps + 1 or any(
                not isfinite(start) or not isfinite(end) or start <= end for start, end in pairwise(sigmas)
            ):
                raise RuntimeError("Klein native sigma grid differs from the requested step count")
        scope.enter_context(adapter.scaled(values[position]))
        position += 1

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(runtime.transformer.register_forward_pre_hook(before).remove)
        handles.callback(runtime.transformer.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        image = render_image(runtime, adapter, strength=strength, steps=steps, **request)
    if position != steps:
        raise RuntimeError("Klein finished before applying all scheduled strengths")
    image.info["strength_schedule"] = {"sigmas": sigmas, "effective_strengths": values}
    return image
