"""Product conditioning projection, extracted without numerical changes.

Source: HyperGAN/particle-sliders core commit
3e8052250d593a55d9f071cebae503e764342d05, pg09_images.py's conditioning block.
MIT; see LICENSE/NOTICE. No image-model binding or GAN implementation copied.
"""
import torch

CONDITION_FEATURE_POLICY = dict(
    version='native-full-conditioning-rademacher-v2', projection_rank=32,
    seed=20261017, chunk_elements=16384, collision_policy='reject differing contexts with identical features')


def _condition_projection(condition):
    """Fixed key/coordinate-specific sketch of all native tensor coordinates.

    Bounded temporary memory; no training RNG, learned encoder or row IDs.
    Tensor masks, pooled vectors and positional/time IDs participate too.
    """
    import hashlib
    from math import sqrt, log1p
    rank = CONDITION_FEATURE_POLICY['projection_rank']
    chunk = CONDITION_FEATURE_POLICY['chunk_elements']
    projection = torch.zeros(rank, dtype=torch.float64)
    count = 0
    rms = 0.
    for key, value in sorted(condition.items()):
        if value is None:
            continue
        if not isinstance(value, (torch.Tensor, int, float, bool)):
            raise ValueError('native condition features require tensor/numeric entries')
        flat = torch.as_tensor(value).detach().float().cpu().contiguous().flatten()
        if not len(flat) or not torch.isfinite(flat).all():
            raise ValueError('native condition features require finite nonempty tensors')
        seed = int.from_bytes(hashlib.sha256(
            f"{CONDITION_FEATURE_POLICY['version']}:{CONDITION_FEATURE_POLICY['seed']}:{key}".encode()
        ).digest()[:8], 'little')
        generator = torch.Generator().manual_seed(seed)
        for start in range(0, len(flat), chunk):
            values = flat[start:start + chunk].double()
            signs = torch.randint(0, 2, (len(values), rank), dtype=torch.int8,
                                  generator=generator).double().mul_(2).sub_(1)
            projection += values @ signs / sqrt(len(flat))
        count += len(flat)
        rms += float(flat.double().square().mean().sqrt())
    if not count:
        raise ValueError('native conditioning contains no numeric coordinates')
    return torch.cat([projection.float(), torch.tensor([log1p(count), rms])])


def _condition_identity(row):
    """Exact context identity for the dataset collision gate; excludes sampled latent."""
    from dataclasses import asdict
    import hashlib
    import json
    def identity(value):
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu().contiguous()
            return dict(shape=list(v.shape), dtype=str(v.dtype),
                        sha256=hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if isinstance(value, dict):
            return {k: identity(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [identity(v) for v in value]
        return value
    context = dict(neutral=row.neutral, positive=row.positive, negative=row.negative,
                   timestep=row.timestep, schedule=asdict(row.schedule) if row.schedule else None,
                   scale=row.scale, guidance=row.guidance, strength=row.strength)
    return hashlib.sha256(json.dumps(identity(context), sort_keys=True).encode()).hexdigest()


def image_condition_features(prepared):
    """Time/sign/CFG plus full native conditioning sketches and exact collision gate.

    Evaluate once over the complete fixed-row dataset before constructing the
    game. For a lazy cache, persist the resulting conditions and their policy/
    tensor hashes in the provider index; do not recompute from row IDs.
    """
    features = []
    contexts = {}
    for item in prepared:
        row = item.row
        times = ([float(row.timestep)] if row.schedule is None else list(row.schedule.timesteps))
        coordinates = torch.tensor([sum(times) / len(times) / 1000., row.scale, row.guidance, row.strength])
        values = torch.cat([coordinates] + [_condition_projection(c)
                           for c in (row.neutral, row.positive, row.negative)])
        if not torch.isfinite(values).all():
            raise ValueError('native conditioning features overflow float32')
        key = values.numpy().tobytes()
        context = _condition_identity(row)
        if key in contexts and contexts[key] != context:
            raise ValueError('native conditioning collision: distinct contexts have identical critic features')
        contexts[key] = context
        features.append(values)
    if not features:
        raise ValueError('nonempty fixed response dataset required')
    return torch.stack(features).to(prepared[0].target.device)
