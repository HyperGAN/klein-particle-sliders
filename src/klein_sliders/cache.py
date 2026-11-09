"""Immutable native teacher rows, stored on CPU and fetched one row at a time."""
from copy import deepcopy
from pathlib import Path
import json
import torch
from particle_sliders.pg09_backends import PreparedResponse, ResponseRow
from particle_sliders.pg09_images import image_condition_features, _condition_identity
from .contracts import PROTOCOL, digest, file_hash


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    return value


def write_row(root, index, prepared):
    row = prepared.row
    stored = dict(row=dict(neutral=cpu(row.neutral), positive=cpu(row.positive),
                           negative=cpu(row.negative), latent=cpu(row.latent), timestep=cpu(row.timestep)),
                  target=cpu(prepared.target), neutral=cpu(prepared.neutral))
    name = f"row-{index:05d}.pt"
    path = root / name
    if path.exists():
        raise FileExistsError(path)
    torch.save(stored, path)
    frozen = PreparedResponse(ResponseRow(**stored["row"]), stored["target"], stored["neutral"])
    return dict(path=name, sha256=file_hash(path),
                context_sha256=_condition_identity(frozen.row),
                condition=image_condition_features([frozen])[0].tolist())


class CachedResponses:
    def __init__(self, directory, runtime, index_sha256):
        self.root = Path(directory).resolve()
        self.runtime = runtime
        self.index_sha256 = index_sha256
        if file_hash(self.root / "index.json") != index_sha256:
            raise ValueError("cache index checksum differs")
        self.index = json.loads((self.root / "index.json").read_text())
        if (self.index.get("protocol") != PROTOCOL or self.index.get("model") != runtime.identity
                or self.index.get("precision") != str(runtime.dtype)):
            raise ValueError("cache protocol, base identity or precision differs")
        self.records = deepcopy(self.index["rows"])
        if not 2 <= len(self.records) <= 4096:
            raise ValueError("requires 2..4096 frozen response rows")
        self.response_shape = tuple(self.index["response_shape"])
        if len(self.response_shape) != 4 or self.response_shape[0] != 1:
            raise ValueError("requires a genuine singleton-batch native spatial field")
        for record in self.records:
            self._path(record)
        self.features = torch.tensor([r["condition"] for r in self.records], dtype=torch.float32)
        if self.features.ndim != 2 or not torch.isfinite(self.features).all():
            raise ValueError("invalid frozen critic conditions")
        contexts = {}
        for record in self.records:
            key = tuple(record["condition"])
            identity = record["context_sha256"]
            if key in contexts and contexts[key] != identity:
                raise ValueError("distinct native contexts have colliding critic conditions")
            contexts[key] = identity
        self._identity = deepcopy(self.index)

    def __len__(self):
        return len(self.records)

    def identity(self):
        if (file_hash(self.root / "index.json") != self.index_sha256
                or self.index != self._identity or self.records != self._identity["rows"]
                or self.runtime.identity != self._identity["model"]):
            raise ValueError("frozen cache identity changed")
        return deepcopy(self._identity)

    def _path(self, record):
        relative = Path(record["path"])
        path = (self.root / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.root)
                or file_hash(path) != record["sha256"]):
            raise ValueError("cache row path/checksum differs")
        return path

    def __getitem__(self, index):
        self.identity()
        record = self.records[index]
        path = self._path(record)
        # Check the same bytes that will be deserialized, without pickle globals.
        import io, hashlib
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("cache row changed while reading")
        stored = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
        if not isinstance(stored, dict) or set(stored) != {"row", "target", "neutral"}:
            raise ValueError("malformed teacher cache row")
        row = ResponseRow(**stored["row"])
        for value in (row.latent, stored["target"], stored["neutral"]):
            if tuple(value.shape) != self.response_shape or not torch.isfinite(value).all():
                raise ValueError("invalid native teacher field")
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.runtime.device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            return value
        return PreparedResponse(ResponseRow(**move(stored["row"])),
                                stored["target"].to(self.runtime.device), stored["neutral"].to(self.runtime.device))

    def conditions(self, indices):
        self.identity()
        # Derive from the content-bound index rather than a caller-mutable tensor.
        return torch.tensor([self.records[int(i)]["condition"] for i in indices],
                            dtype=torch.float32, device=self.runtime.device)


@torch.no_grad()
def prepare_cache(runtime, adapter, pairs, directory, *, seeds=(7, 19), size=512, steps=8):
    """Positive teacher velocities on a matched neutral native Euler trajectory."""
    import numpy as np
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    from .native import KleinResponseAdapter
    if runtime.pipe is None or size < 64 or size > 1024 or size % 16 or not 2 <= steps <= 50:
        raise ValueError("cache needs full pipeline, size 64..1024 divisible by 16, and 2..50 steps")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    backend = KleinResponseAdapter(runtime, adapter)
    pipe = runtime.pipe
    records = []
    for pair in pairs:
        neutral, positive = runtime.encode(pair["neutral"]), runtime.encode(pair["positive"])
        for seed in seeds:
            generator = torch.Generator(device=runtime.device).manual_seed(seed)
            packed, ids = pipe.prepare_latents(1, runtime.transformer.config.in_channels // 4,
                size, size, runtime.dtype, runtime.device, generator)
            side = size // 16
            scheduler = deepcopy(pipe.scheduler)
            sigmas = np.linspace(1., 1 / steps, steps)
            if getattr(scheduler.config, "use_flow_sigmas", False):
                sigmas = None
            times, _ = retrieve_timesteps(scheduler, steps, runtime.device, sigmas=sigmas,
                mu=compute_empirical_mu(image_seq_len=packed.shape[1], num_steps=steps))
            for timestep in times:
                field = pipe._unpack_latents_with_ids(packed, ids, side, side)
                row = ResponseRow(neutral, positive, neutral, field, timestep=timestep)
                prepared = backend.prepare(row)
                records.append(write_row(root, len(records), prepared))
                packed = scheduler.step(pipe._pack_latents(prepared.neutral.to(runtime.dtype)),
                                        timestep, packed, return_dict=False)[0]
    manifest = dict(protocol=PROTOCOL, model=runtime.identity, precision=str(runtime.dtype),
        response_shape=list(prepared.target.shape), pairs_sha256=digest(pairs), seeds=list(seeds),
        size=size, scheduler_steps=steps, teacher="frozen positive velocity on neutral trajectory; CFG=1",
        split="train", condition_policy="shared full-conditioning-rademacher-v2", rows=records)
    (root / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return file_hash(root / "index.json")
