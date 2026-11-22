"""Opt-in S-PSR on the original Klein velocity consumed by Euler."""

from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
from itertools import pairwise
import json
from math import fsum, isfinite
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType
from zipfile import ZipFile

import torch

from .contracts import digest


SCHEMA = "klein-native8-velocity-replay-v1"
VELOCITY_SHAPE = (1, 1024, 128)
CORE_REVISION = "28745b7d47435f89bb55b86d1761adce6b8f1b06"
_SHARED_LOCK = Lock()
_SHARED = {
    "prompt_steering": "67ee03066e255fe105735820b0edba53757d943b93239b1279de707d23dbae59",
    "strength_schedule": "b759d72a866531a7d4cd23c13f1664ec6c746e812b64217cb1835ef76eebcb4f",
}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read(path, expected, label):
    if not _sha(expected):
        raise ValueError(f"Provide the authenticated Klein {label} SHA256")
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Klein {label} SHA256 differs")
    return data


def _load_shared(core_wheel):
    with ZipFile(core_wheel) as wheel:
        prefix = str(core_wheel)
        sources = {key: wheel.read(f"particle_sliders/{key}.py") for key in _SHARED}
    if any(sha256(source).hexdigest() != _SHARED[key] for key, source in sources.items()):
        raise ValueError("Expected the exact published shared PSR and binary32 dose sources")
    scope = {}
    # The original owner keeps its historical particle_sliders imports.
    with _SHARED_LOCK:
        for key, source in sources.items():
            name = f"{__name__}._shared_{key}"
            module = sys.modules.get(name)
            if module is None:
                module = ModuleType(name)
                module.__file__ = f"{prefix}/particle_sliders/{key}.py"
                sys.modules[name] = module
                try:
                    exec(compile(source, module.__file__, "exec"), module.__dict__)
                except Exception:
                    del sys.modules[name]
                    raise
            scope[key] = module
    return scope["prompt_steering"].PromptSteering, scope["strength_schedule"].match_binary32_interval_dose


def _finite_nonnegative(value):
    return type(value) in (int, float) and 0 <= value <= torch.finfo(torch.float32).max


class VelocitySteeringPlan:
    """Bind one archived fit to its authenticated stock RAW owner and native grid.

    The binding manifest identifies the fit file, its preregistration, the
    original RAW file and exact native model identity by SHA256. It is created
    before rendering or choosing a calibration alpha.
    """

    def __init__(self, *, core_wheel, binding_path, binding_sha256, fit_path):
        binding = json.loads(_read(binding_path, binding_sha256, "binding"))
        if (not isinstance(binding, dict) or binding.get("schema") != SCHEMA
                or binding.get("weights") != "raw" or "ema_decay" not in binding or binding["ema_decay"] is not None):
            raise ValueError("Expected an authenticated Klein RAW velocity replay binding")
        for key in ("fit_sha256", "fit_plan_sha256", "raw_sha256", "model_identity_sha256"):
            if not _sha(binding.get(key)):
                raise ValueError(f"Klein velocity replay binding lacks {key}")
        grid = binding.get("sigmas")
        if (binding.get("velocity_shape") != list(VELOCITY_SHAPE) or not isinstance(grid, list)
                or len(grid) != 9 or grid[-1] != 0 or any(type(v) not in (float, int) or not isfinite(v) for v in grid)
                or any(a <= b for a, b in pairwise(grid))):
            raise ValueError("Expected the measured eight-step Klein velocity geometry and grid")
        fit = torch.load(BytesIO(_read(fit_path, binding["fit_sha256"], "probe fit")), map_location="cpu", weights_only=True)
        if (not isinstance(fit, Mapping) or fit.get("weights") != "raw" or "ema_decay" not in fit or fit["ema_decay"] is not None
                or fit.get("native_raw_strength") != 0 or fit.get("velocity_shape") != list(VELOCITY_SHAPE)
                or fit.get("sigmas") != grid or fit.get("core_module_sha256") != _SHARED["prompt_steering"]
                or fit.get("raw_checkpoint", {}).get("sha256") != binding["raw_sha256"]
                or fit.get("plan", {}).get("sha256") != binding["fit_plan_sha256"]):
            raise ValueError("Probe fit differs from the authenticated Klein RAW and velocity binding")
        state = fit.get("probe")
        shapes = {"probe.weight": (1, 128), "probe.bias": (1,), "direction.weight": (1, 128)}
        if (not isinstance(state, Mapping) or state.keys() != shapes.keys()
                or any(not isinstance(state[k], torch.Tensor) or state[k].dtype != torch.float32
                       or state[k].shape != shape or not state[k].isfinite().all() for k, shape in shapes.items())):
            raise ValueError("Expected one finite fitted shared FP32 velocity direction and affine probe")
        self._probe_type, self._dose = _load_shared(core_wheel)
        self._state = {name: value.detach().clone() for name, value in state.items()}
        self._binding = {
            "binding_sha256": binding_sha256, "fit_sha256": binding["fit_sha256"],
            "raw_sha256": binding["raw_sha256"], "model_identity_sha256": binding["model_identity_sha256"],
        }
        self._sigmas = tuple(grid)

    def _verify_owner(self, adapter, raw_path, model_identity):
        if digest(model_identity) != self._binding["model_identity_sha256"] or adapter.strength != 0:
            raise ValueError("Use the fitted Klein native model and frozen RAW scale zero")
        payload = torch.load(BytesIO(_read(raw_path, self._binding["raw_sha256"], "RAW adapter")),
                             map_location="cpu", weights_only=True)
        if (not isinstance(payload, Mapping) or payload.get("weights") != "raw"
                or "ema_decay" not in payload or payload["ema_decay"] is not None):
            raise ValueError("The bound legacy Klein adapter must be RAW with EMA disabled")
        saved = payload.get("network")
        attached = adapter.state_dict()
        if (not isinstance(saved, Mapping) or saved.keys() != attached.keys()
                or any(not isinstance(value, torch.Tensor) or value.dtype != attached[key].dtype
                       or not torch.equal(value, attached[key].detach().cpu()) for key, value in saved.items())):
            raise ValueError("The live Klein adapter differs from the fitted legacy RAW packet")

    def _probe(self, device):
        with torch.random.fork_rng(devices=[]):
            probe = self._probe_type(VELOCITY_SHAPE[-1], device="cpu")
        probe.load_state_dict(self._state, strict=True)
        return probe.eval().requires_grad_(False).to(device)


def _input_hash(tensor):
    return sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _fraction(value):
    return [value.numerator, value.denominator]


def _constant(plan, from_adaptive, bound):
    if not isinstance(from_adaptive, Mapping) or from_adaptive.get("mode") != "adaptive":
        raise ValueError("Supply this request's adaptive Klein trace for constant playback")
    if any(from_adaptive.get(key) != value for key, value in bound.items()):
        raise ValueError("The matching Klein trace belongs to a different fit, request or alpha")
    values = from_adaptive.get("steps")
    intervals = len(plan._sigmas) - 1
    if (not isinstance(values, list) or len(values) != intervals
            or any(not isinstance(step, Mapping) or step.get("interval") != index
                   or not _finite_nonnegative(step.get("applied_coefficient")) for index, step in enumerate(values))):
        raise ValueError("Expected a measured adaptive coefficient on every Klein Euler interval")
    match = plan._dose([row["applied_coefficient"] for row in values], plan._sigmas)
    if (from_adaptive.get("exact_coefficient_dose") != _fraction(match.adaptive_dose)
            or from_adaptive.get("nearest_even_constant") != match.constant_strength
            or from_adaptive.get("exact_nearest_constant_dose") != _fraction(match.constant_dose)):
        raise ValueError("The measured Klein adaptive trace and matching receipt differ")
    return match.constant_strength


def render_native_velocity(render, *, plan, transformer, pipeline, adapter, raw_path, model_identity,
                           request, alpha, mode="adaptive", from_adaptive=None, check_live):
    """Run an owned stock Klein request, returning its result and exact velocity receipt.

    The caller binds a specific seed plus hashes of the actual packed initial
    latent, prompt embedding, and text coordinates. All three arms share the same RAW scale zero.
    """
    if not isinstance(plan, VelocitySteeringPlan):
        raise ValueError("Provide a fitted Klein VelocitySteeringPlan")
    if (not isinstance(request, Mapping) or set(request) != {"seed", "initial_sha256", "conditioning_sha256", "text_ids_sha256"}
            or type(request["seed"]) is not int or not 0 <= request["seed"] < 2**63
            or not all(_sha(request[key]) for key in ("initial_sha256", "conditioning_sha256", "text_ids_sha256"))):
        raise ValueError("Bind the native request seed, start and prompt embedding")
    if not _finite_nonnegative(alpha) or mode not in ("off", "adaptive", "same-atom-constant"):
        raise ValueError("Supply an explicit Klein alpha and a supported velocity arm")
    sigmas = plan._sigmas
    bound = dict(plan._binding, request=dict(request), requested_alpha=float(alpha), sigmas=list(sigmas))
    if mode == "same-atom-constant":
        constant = _constant(plan, from_adaptive, bound)
    elif from_adaptive is not None:
        raise ValueError("Only the same-atom constant uses an adaptive trace")
    else:
        constant = 0.0
    if (adapter.training or transformer.training
            or any(parameter.requires_grad for owner in (adapter, transformer) for parameter in owner.parameters())):
        raise ValueError("The stock Klein base and RAW adapter must be frozen for velocity replay")
    check_live()
    plan._verify_owner(adapter, raw_path, model_identity)

    scheduler = probe = None
    frames = []

    def after(_model, _args, kwargs, output):
        nonlocal scheduler, probe
        check_live()
        pipe = pipeline()
        index = len(frames)
        if index == 0:
            scheduler = pipe.scheduler
            if tuple(scheduler.sigmas.detach().cpu().tolist()) != sigmas:
                raise RuntimeError("Actual Klein Euler grid differs from the fitted velocity grid")
            start = kwargs.get("hidden_states")
            condition, ids = kwargs.get("encoder_hidden_states"), kwargs.get("txt_ids")
            if (not isinstance(start, torch.Tensor) or start.shape != VELOCITY_SHAPE or start.dtype != torch.bfloat16
                    or not isinstance(condition, torch.Tensor) or condition.shape != (1, 128, 7680)
                    or condition.dtype != torch.bfloat16 or _input_hash(start) != request["initial_sha256"]
                    or _input_hash(condition) != request["conditioning_sha256"]
                    or not isinstance(ids, torch.Tensor) or ids.shape != (1, 128, 4)
                    or ids.dtype != torch.int64 or _input_hash(ids) != request["text_ids_sha256"]):
                raise RuntimeError("The actual first Klein latent or prompt embedding differs from the request")
        if (index >= len(sigmas) - 1 or pipe.transformer is not transformer or pipe.scheduler is not scheduler
                or pipe.do_classifier_free_guidance or scheduler.begin_index != 0
                or scheduler.step_index != (index if index else None) or adapter.strength != 0):
            raise RuntimeError("Expected one stock Klein CFG1 velocity at fixed RAW scale zero per Euler interval")
        if not isinstance(output, tuple) or len(output) != 1:
            raise RuntimeError("Expected the stock Klein one-tensor velocity tuple")
        velocity = output[0]
        if (not isinstance(velocity, torch.Tensor) or velocity.shape != VELOCITY_SHAPE
                or velocity.dtype != torch.bfloat16 or not velocity.isfinite().all()):
            raise RuntimeError("Expected finite packed Klein BF16 output velocity")
        if probe is None:
            probe = plan._probe(velocity.device)
        with torch.no_grad(), torch.autocast(device_type=velocity.device.type, enabled=False):
            coefficient = probe.coefficients(velocity.mean(dim=(0, 1), dtype=torch.float32))
            applied = float(coefficient * alpha) if mode == "adaptive" else constant
            if not isfinite(applied):
                raise RuntimeError("Nonfinite applied Klein S-PSR coefficient")
            edited = (velocity.float() + applied * probe.direction.weight[0]).to(velocity.dtype) if applied else velocity
            realized = (edited.float() - velocity.float()).norm(dim=-1).mean().item()
        if not edited.isfinite().all() or not isfinite(realized):
            raise RuntimeError("Nonfinite post-rounding Klein velocity edit")
        frames.append(dict(interval=index, applied_coefficient=applied, lambda_coefficient=float(coefficient),
                           realized_mean_BF16_edit_norm=realized))
        return (edited,)

    with transformer.register_forward_hook(after, with_kwargs=True):
        result = render()
    if scheduler is None or len(frames) != len(sigmas) - 1 or scheduler.step_index != len(frames) or adapter.strength != 0:
        raise RuntimeError("The original Klein owner did not consume the complete measured scale-zero Euler grid")
    dose = plan._dose([frame["applied_coefficient"] for frame in frames], sigmas)
    field_dose = fsum(float(Fraction(a) - Fraction(b)) * frame["realized_mean_BF16_edit_norm"]
                      for (a, b), frame in zip(pairwise(sigmas), frames, strict=True))
    return result, dict(
        bound, mode=mode, raw_strength=0.0, steps=frames, weights="raw", ema_decay=None,
        shared_core_revision=CORE_REVISION, exact_coefficient_dose=_fraction(dose.adaptive_dose),
        nearest_even_constant=dose.constant_strength, exact_nearest_constant_dose=_fraction(dose.constant_dose),
        integrated_realized_mean_BF16_edit_norm=field_dose)
