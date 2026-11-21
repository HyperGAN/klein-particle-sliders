"""Opt-in strength changes on the original Klein finished-image consumer."""

from contextlib import ExitStack
from itertools import pairwise
from math import isfinite
from sys import float_info

from .adapter import ParticleAdapter
from .native import NativeRuntime


def resolve(strength, steps, step_gains):
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
        raise ValueError("Klein requires one gain per native scheduler step")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Klein step gains must be finite and nonnegative")
    strengths = [strength * gain for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("Klein effective slider strengths must be in [0,1]")
    return strengths


def render(session, *, step_gains, strength=1.0, steps=8, **request):
    """Schedule an already opened strict RAW session and keep its native factory."""
    session._open()
    if session.selection != "raw" or session.binding.render is None:
        raise ValueError("Klein scheduling requires an open RAW render session")
    # The original factory retains the actual native owners in this callback.
    # Use those same objects so its packet and factory source hashes stay intact.
    owners = [cell.cell_contents for cell in session.binding.close.__closure__ or ()]
    runtimes = [owner for owner in owners if isinstance(owner, NativeRuntime)]
    adapters = [owner for owner in owners if isinstance(owner, ParticleAdapter)]
    if len(runtimes) != 1 or len(adapters) != 1 or runtimes[0].identity != session.binding.identity["model"]:
        raise ValueError("expected the original Klein native factory owners")
    runtime, adapter = runtimes[0], adapters[0]
    if type(steps) is not int or not 2 <= steps <= 50:
        raise ValueError("Klein requires its supported native step count")
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("Klein nominal strength must be in [0,1]")
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
        image = session.render(strength=strength, steps=steps, **request)
    if position != steps:
        raise RuntimeError("Klein finished before applying all scheduled strengths")
    image.info["strength_schedule"] = {"sigmas": sigmas, "effective_strengths": values}
    return image
