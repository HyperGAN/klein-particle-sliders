"""Verified shared steering for the strict Klein runtime's original core pin."""

from dataclasses import asdict
from hashlib import sha256
import sys
from threading import Lock
from types import ModuleType
from zipfile import ZipFile

import torch


CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_MODULE = "particle_sliders/adaptive_steering.py"
CORE_MODULE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
_MODULE_NAME = f"{__name__}._core_{CORE_REVISION}"
_LOAD_LOCK = Lock()


def _shared_steering(core_wheel):
    # Historical RAW receipts bind the original core through initialization.
    # Load only this Git-verified standalone module into its own namespace.
    try:
        with ZipFile(core_wheel) as archive:
            source = archive.read(CORE_MODULE)
    except KeyError as error:
        raise ValueError("Klein requires the shared steering module from core PR #269") from error
    if sha256(source).hexdigest() != CORE_MODULE_SHA256:
        raise ValueError("Klein requires the exact shared steering module from core PR #269")
    with _LOAD_LOCK:
        if _MODULE_NAME not in sys.modules:
            module = ModuleType(_MODULE_NAME)
            sys.modules[_MODULE_NAME] = module
            try:
                exec(compile(source, f"{core_wheel}/{CORE_MODULE}", "exec"), module.__dict__)
            except BaseException:
                del sys.modules[_MODULE_NAME]
                raise
        return sys.modules[_MODULE_NAME]


class LatentGatePlan:
    """Explicit settings and provenance for one live Klein latent readout."""

    def __init__(self, *, core_wheel, method, parameter, calibration_sha256,
                 direction=None, seed=None, clabs=None):
        if (not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(character not in "0123456789abcdef" for character in calibration_sha256)):
            raise ValueError("Provide the SHA-256 of the Klein gate calibration record")
        self._steering = _shared_steering(core_wheel)
        self._clabs = self._steering.CLABSParameters(**clabs) if clabs is not None else None
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random gate does not use a Klein latent direction")
            self._direction = None
        else:
            if direction is None:
                raise ValueError("Klein CFP and CLABS require a calibrated packed-channel direction")
            probe = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (probe.ndim != 1 or not torch.isfinite(probe).all()
                    or not torch.isclose(probe.norm(), probe.new_tensor(1.0))):
                raise ValueError("Klein requires a finite unit direction in packed latent channel space")
            self._direction = probe
        self._method, self._parameter, self._seed = method, float(parameter), seed
        self._calibration_sha256 = calibration_sha256
        self._new_gate()

    def _new_gate(self):
        return self._steering.ActivationGate(self._method, self._parameter, seed=self._seed, clabs=self._clabs)

    def _receipt(self):
        return {
            "core_revision": CORE_REVISION,
            "signal": "mean pre-edit packed Klein latent channels in FP32",
            "calibration_sha256": self._calibration_sha256,
            "method": self._method,
            "parameter": self._parameter,
            "seed": self._seed,
            "clabs": asdict(self._clabs) if self._clabs is not None else None,
            "direction": self._direction.tolist() if self._direction is not None else None,
        }

    def _render_gate(self, channels, device):
        gate = self._new_gate()
        if self._direction is None:
            return gate, None
        if self._direction.shape != (channels,):
            raise ValueError("Direction width must match the opened Klein native model")
        return gate, self._direction.to(device=device)

    def _score(self, latent, direction, strength, channels):
        if (not isinstance(latent, torch.Tensor) or latent.ndim != 3 or latent.shape[0] != 1
                or latent.shape[1] == 0 or latent.shape[-1] != channels):
            raise RuntimeError("Expected one native packed Klein latent before adapter intervention")
        if direction is None:
            return None
        pooled = latent.mean(dim=(0, 1), dtype=torch.float32)
        return float(self._steering.activation_projection(pooled, direction, strength=strength))
