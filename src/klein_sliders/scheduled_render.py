"""Opt-in strength changes on the original Klein finished-image consumer."""

from contextlib import ExitStack
from itertools import pairwise
from math import isfinite
from sys import float_info

from .activation_gate import LatentGatePlan
from .adapter import ParticleAdapter
from .native import NativeRuntime

_USE_FIXED_DEFAULT = object()


def resolve(strength, steps, step_gains):
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
        raise ValueError("Klein requires one gain per native scheduler step")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Klein step gains must be finite and nonnegative")
    strengths = [strength * gain for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("Klein effective slider strengths must be in [0,1]")
    return strengths


def render(session, *, step_gains=None, gate_plan=None, strength=_USE_FIXED_DEFAULT, steps=8, **request):
    """Apply fixed gains or a calibrated gate on the original strict RAW renderer."""
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
    if strength is _USE_FIXED_DEFAULT:
        if gate_plan is not None:
            raise ValueError("Gated Klein requires the explicit nominal strength used in calibration")
        strength = 1.0
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("Klein nominal strength must be in [0,1]")
    if gate_plan is None:
        values = resolve(strength, steps, step_gains)
    else:
        if step_gains is not None or not isinstance(gate_plan, LatentGatePlan):
            raise ValueError("Choose fixed Klein step gains or one calibrated LatentGatePlan")
        if strength == 0:
            raise ValueError("Gated Klein needs nonzero nominal strength; render OFF at strength zero")
        channels = runtime.transformer.config.in_channels
        gate, direction = gate_plan._render_gate(channels, runtime.device)
        values = []
    decisions = []
    position = 0
    sigmas = None
    scheduler = None
    scope = ExitStack()

    def before(_model, _args, kwargs):
        nonlocal position, sigmas, scheduler
        if position >= steps:
            raise RuntimeError("Klein made extra native denoiser calls")
        if sigmas is None:
            scheduler = runtime.pipe.scheduler
            sigmas = scheduler.sigmas.detach().cpu().tolist()
            if len(sigmas) != steps + 1 or any(
                not isfinite(start) or not isfinite(end) or start <= end for start, end in pairwise(sigmas)
            ):
                raise RuntimeError("Klein native sigma grid differs from the requested step count")
        if gate_plan is not None:
            expected_index = position if position else None
            if (runtime.pipe.scheduler is not scheduler or scheduler.begin_index != 0
                    or scheduler.step_index != expected_index or runtime.pipe.do_classifier_free_guidance):
                raise RuntimeError("Expected one pre-edit Klein CFG=1 forward per Euler interval")
            score = gate_plan._score(kwargs.get("hidden_states"), direction, strength, channels)
            decision = gate.step(score)
            value = strength if decision.fired else 0.0
            values.append(value)
            decisions.append(dict(interval=position, projection=score, probability=decision.probability,
                                  fired=decision.fired, effective_strength=value))
        scope.enter_context(adapter.scaled(values[position]))
        position += 1

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(runtime.transformer.register_forward_pre_hook(before, with_kwargs=True).remove)
        handles.callback(runtime.transformer.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        image = session.render(strength=strength, steps=steps, **request)
    if position != steps:
        raise RuntimeError("Klein finished before applying all scheduled strengths")
    receipt = {"sigmas": sigmas, "effective_strengths": values}
    if gate_plan is not None:
        if scheduler.step_index != steps:
            raise RuntimeError("Klein scheduler did not complete all gated intervals")
        usage = gate.usage
        receipt.update(activation_gate=gate_plan._receipt(), decisions=decisions, nominal_strength=strength,
                       fired=usage.fired, eligible=usage.eligible, gate_rate=usage.rate,
                       sigma_weighted_strength=sum(value * (start - end) for value, (start, end)
                                                   in zip(values, pairwise(sigmas), strict=True)))
    image.info["strength_schedule"] = receipt
    return image
