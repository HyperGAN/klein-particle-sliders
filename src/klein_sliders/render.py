"""Native finished-image inference from RAW exports; no teacher cache or PG game.

Definitions and request validation use stdlib only. Torch/the native factory are
imported only for an explicit valid render request. This is not qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path

RENDER_PROTOCOL = "klein-base4b-native-pipeline-matched-images-v1"
SIZES = (256, 512)


def validate_render_request(*, prompt, height, width, steps, seed, strength,
                            images=None, output_type="pil"):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
        raise ValueError("render requires a nonempty prompt of at most16384 characters")
    if type(height) is not int or type(width) is not int or height not in SIZES or width not in SIZES:
        raise ValueError("initial native render supports explicit256/512 dimensions;1024 remains unqualified")
    if type(steps) is not int or not 2 <= steps <= 50:
        raise ValueError("native render requires2..50 scheduler steps")
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError("explicit nonnegative native seed required")
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("native unipolar slider strength must be finite in[0,1]")
    if images is not None or output_type != "pil":
        raise ValueError("initial finished render is text-only/PIL; reference editing is unsupported")


@contextmanager
def native_pipeline_state(pipe):
    """OFF and ON each begin from the same native scheduler, even after error."""
    scheduler = pipe.scheduler
    saved = {key: (key in vars(pipe), vars(pipe).get(key)) for key in
             ("_guidance_scale", "_attention_kwargs", "_current_timestep", "_interrupt", "_num_timesteps")}
    pipe.scheduler = deepcopy(scheduler)
    try:
        yield
    finally:
        pipe.scheduler = scheduler
        for key, (present, value) in saved.items():
            if present:
                setattr(pipe, key, value)
            else:
                vars(pipe).pop(key, None)


def render_image(runtime, adapter, *, prompt, height=512, width=512, steps=8,
                 seed=7, strength=1., images=None, output_type="pil"):
    """Call the exact native public pipeline with the actual attached RAW adapter.

    The common inference session validates/binds the packet and owns network/RNG
    observation. The pipeline owns native scheduling, packing, BN inverse, VAE
    decoding and image postprocessing. No response tensor is called an image.
    """
    validate_render_request(prompt=prompt, height=height, width=width, steps=steps,
                            seed=seed, strength=strength, images=images, output_type=output_type)
    import torch
    pipe = runtime.load_renderer()
    if pipe.transformer is not runtime.transformer:
        raise RuntimeError("finished rendering must reuse the actual bound native transformer")
    generator = torch.Generator(device=runtime.device).manual_seed(seed)
    with native_pipeline_state(pipe), adapter.scaled(strength):
        result = pipe(image=None, prompt=prompt, height=height, width=width,
            num_inference_steps=steps, guidance_scale=1., num_images_per_prompt=1,
            generator=generator, output_type="pil", return_dict=True,
            max_sequence_length=128, text_encoder_out_layers=(9, 18, 27))
    if (not isinstance(result.images, list) or len(result.images) != 1
            or result.images[0].size != (width, height)):
        raise ValueError("native pipeline must finish exactly one requested-size image")
    from PIL.Image import Image
    if not isinstance(result.images[0], Image):
        raise TypeError("native finished render must return a real PIL image")
    return result.images[0]


def _record(path, raw=None):
    path = Path(path)
    raw = path.read_bytes() if raw is None else raw
    return dict(path=str(path.absolute()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _write(path, value):
    import os
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def render_pair(*, artifact, model_dir, output, prompt, size=512, steps=8, seed=7,
                strength=1., device="cpu", precision="bfloat16"):
    """One cache-independent RAW inference session, two genuine finished PNGs.

    An existing output directory is never resumed/replaced. Failure preserves
    any completed OFF/ON plus truthful metadata; this function never retries.
    """
    validate_render_request(prompt=prompt, height=size, width=size, steps=steps,
                            seed=seed, strength=strength)
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a fresh native render output directory")
    if device not in ("cpu", "cuda:0", "cuda:1") or precision not in ("float32", "bfloat16"):
        raise ValueError("explicit single native device and FP32/BF16 required")
    if not Path(model_dir).is_dir():
        raise ValueError("local staged model_dir required; no download fallback")
    raw = Path(artifact).read_bytes()
    artifact_record = _record(artifact, raw)
    import torch
    from particle_sliders.inference import open_inference
    from .lifecycle import NATIVE_BACKEND
    # Exactly these recorded bytes are deserialized; core validates the whole
    # packet/spec/factory/native identity before strictly loading its network.
    packet = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    del raw
    if not isinstance(packet, dict) or packet.get("selection") != "raw":
        raise ValueError("finished native rendering requires an original RAW-only export")
    native = dict(prompt=prompt, height=size, width=size, steps=steps)
    options = dict(model_dir=str(Path(model_dir).absolute()), device=device, base_precision=precision)
    with open_inference(packet, backend=NATIVE_BACKEND, operation="render", strength=strength,
                        options=options, output_type="pil", native_request=native) as session:
        if session.selection != "raw":
            raise ValueError("RAW current weights required")
        output.mkdir(parents=True, exist_ok=False)
        receipt = dict(protocol=RENDER_PROTOCOL, status="running", artifact=artifact_record,
            original_artifact_internal_sha256=packet["sha256"], origin=packet["origin"],
            specification=packet["specification"], selection="raw", ema_decay=None,
            request=dict(native, seed=seed, ON_strength=strength), options=options,
            matched_OFF_strength=0., pipeline_guidance_scale=1.,
            prompt_profile=dict(max_sequence_length=128, text_encoder_out_layers=[9,18,27]),
            images=[], native_qualification=None, training_updates=0, teacher_cache_used=False)
        _write(output / "render-receipt.json", receipt)
        try:
            for name, alpha in (("OFF", 0.), ("ON", strength)):
                image = session.render(strength=alpha, seed=seed, output_type="pil", **native)
                path = output / (name + ".png")
                with path.open("xb") as stream:
                    image.save(stream, format="PNG")
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
                receipt["images"].append(dict(label=name, strength=alpha, output=_record(path),
                    width=image.width, height=image.height, mode=image.mode))
                _write(output / "render-receipt.json", receipt)
            receipt["status"] = "complete"
        except BaseException as error:
            receipt.update(status="failed", error=repr(error), completed_outputs_retained=True)
            raise
        finally:
            _write(output / "render-receipt.json", receipt)
    return receipt
