"""Allocation-free product configuration and staged-source identity."""
from pathlib import Path
import hashlib
import json
import math

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
MODEL_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
CORE_REVISION = "3e8052250d593a55d9f071cebae503e764342d05"
DIFFUSERS_REVISION = "35087a644038eb531f48581708d4763044cb4852"
NATIVE_SOURCE_HASHES = {
    "pipeline_flux2_klein.py": "8afefd06451d2bed0f34b59e930e60383c6a47fc8cebbec74ea30bfcdc7b79a9",
    "transformer_flux2.py": "88cbc0569409698efbdaa5cd25403b78fb6607b9cf1b15855bc392b2ad0c1607",
}
PROTOCOL = "flux2-klein-base4b-native-velocity-v1"
EXPECTED = dict(in_channels=128, joint_attention_dim=7680, num_layers=5,
                num_single_layers=20, guidance_embeds=False, patch_size=1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_architecture(config):
    if any(config.get(key) != value for key, value in EXPECTED.items()):
        raise ValueError("requires exact undistilled FLUX.2-klein-base-4B architecture")


def verify_model(directory):
    root = Path(directory).resolve()
    lock = json.loads((root / "klein-model.lock.json").read_text())
    if lock.get("model") != MODEL or lock.get("revision") != MODEL_REVISION:
        raise ValueError("model identity differs from the pinned Klein base 4B")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("staged model requires a nonempty file manifest")
    actual = {str(p.relative_to(root)) for folder in ("transformer", "scheduler", "vae", "text_encoder", "tokenizer")
              for p in (root / folder).rglob("*") if p.is_file()}
    actual.add("model_index.json")
    if actual != set(files):
        raise ValueError("staged model file inventory changed")
    for name, expected in files.items():
        relative = Path(name)
        path = (root / relative).resolve()
        # Hugging Face cache symlinks are legal; relative names must be contained.
        if relative.is_absolute() or ".." in relative.parts or file_hash(path) != expected:
            raise ValueError("staged model checksum differs: " + name)
    validate_architecture(json.loads((root / "transformer/config.json").read_text()))
    if json.loads((root / "model_index.json").read_text()).get("_class_name") != "Flux2KleinPipeline":
        raise ValueError("requires the native Flux2KleinPipeline")
    return dict(model=MODEL, revision=MODEL_REVISION, files_sha256=digest(files),
                diffusers_revision=DIFFUSERS_REVISION)


def load_pairs(path):
    pairs = json.loads(Path(path).read_text())
    if not isinstance(pairs, list) or not 2 <= len(pairs) <= 128:
        raise ValueError("supply 2..128 prompt pairs")
    for pair in pairs:
        if (not isinstance(pair, dict) or set(pair) != {"neutral", "positive"}
                or any(not isinstance(v, str) or not v.strip() for v in pair.values())
                or pair["neutral"] == pair["positive"]):
            raise ValueError("each pair requires different nonempty neutral and positive prompts")
    return pairs


def validate_options(options):
    allowed = {"model_dir", "cache_dir", "cache_index_sha256", "device", "base_precision",
               "steps", "batch", "seed", "ema_decay", "critic_width", "response_scale", "sigma"}
    if set(options) - allowed:
        raise ValueError("unknown Klein options: " + str(sorted(set(options) - allowed)))
    for key in ("model_dir", "cache_dir"):
        if not isinstance(options.get(key), str) or not Path(options[key]).is_dir():
            raise ValueError(key + " must be an existing local directory")
    device = options.get("device", "cpu")
    if device != "cpu" and device not in ("cuda:0", "cuda:1"):
        raise ValueError("use cpu, cuda:0 or cuda:1 explicitly")
    if options.get("base_precision", "bfloat16") not in ("float32", "bfloat16"):
        raise ValueError("only FP32/BF16 native bases are supported")
    expected = options.get("cache_index_sha256")
    if not isinstance(expected, str) or file_hash(Path(options["cache_dir"]) / "index.json") != expected:
        raise ValueError("cache_index_sha256 must bind the exact frozen cache")
    for key in ("sigma", "response_scale"):
        value = options.get(key, 1.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
    if type(options.get("critic_width", 32)) is not int or not 4 <= options.get("critic_width", 32) <= 128:
        raise ValueError("critic_width must be an integer in [4,128]")
