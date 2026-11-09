"""Explicit commands; the default plan never imports torch or allocates a device."""
import argparse
import json
from pathlib import Path
from .contracts import (MODEL, MODEL_REVISION, CORE_REVISION, DIFFUSERS_REVISION,
                        PROTOCOL, file_hash, load_pairs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native FLUX.2 Klein base 4B Particle Sliders")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="allocation-free training plan")
    plan.add_argument("--config", default="configs/diffuse-light.json")
    stage = sub.add_parser("stage-model", help="download the pinned pretrained snapshot; no GPU")
    stage.add_argument("--directory", required=True)
    cache = sub.add_parser("cache", help="prepare frozen native teacher velocities; scheduler admission first")
    cache.add_argument("--model-dir", required=True)
    cache.add_argument("--pairs", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--device", choices=["cpu", "cuda:0", "cuda:1"], required=True)
    cache.add_argument("--size", type=int, default=512)
    cache.add_argument("--steps", type=int, default=8)
    cache.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    train = sub.add_parser("train", help="train/resume through shared core; scheduler admission first")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--stop-after", type=int)
    args = parser.parse_args(argv)

    if args.command == "plan":
        config = json.loads(Path(args.config).read_text())
        options = config["options"]
        print(json.dumps(dict(model=MODEL, model_revision=MODEL_REVISION, core_revision=CORE_REVISION,
            diffusers_revision=DIFFUSERS_REVISION, protocol=PROTOCOL, config=config,
            staged_model=Path(options["model_dir"], "klein-model.lock.json").is_file(),
            frozen_cache=Path(options["cache_dir"], "index.json").is_file(),
            qualification="CPU source conformance only; no pretrained/GPU quality claim",
            next="Stage model, prepare cache, bind index SHA256, coordinate bounded GPU qualification with scheduler"),
            indent=2))
        return

    if args.command == "stage-model":
        from huggingface_hub import snapshot_download
        from .contracts import validate_architecture
        directory = Path(args.directory)
        if directory.exists():
            raise FileExistsError("use a fresh directory for a pinned snapshot")
        snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=directory,
            allow_patterns=["model_index.json", "transformer/*", "scheduler/*", "vae/*",
                            "text_encoder/*", "tokenizer/*"])
        validate_architecture(json.loads((directory / "transformer/config.json").read_text()))
        files = {str(p.relative_to(directory)): file_hash(p)
                 for folder in ("transformer", "scheduler", "vae", "text_encoder", "tokenizer")
                 for p in (directory / folder).rglob("*") if p.is_file()}
        files["model_index.json"] = file_hash(directory / "model_index.json")
        lock = dict(model=MODEL, revision=MODEL_REVISION, files=files)
        (directory / "klein-model.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
        print(json.dumps(dict(directory=str(directory), files=len(files), model_revision=MODEL_REVISION)))
        return

    if args.command == "cache":
        pairs = load_pairs(args.pairs)
        if (not 1 <= len(args.seeds) <= 16 or len(set(args.seeds)) != len(args.seeds)
                or any(not 0 <= seed < 2**63 - 100 for seed in args.seeds)
                or len(pairs) * len(args.seeds) * args.steps > 4096):
            raise ValueError("supply 1..16 unique nonnegative seeds and at most 4096 rows")
        from .native import NativeRuntime
        from .adapter import ParticleAdapter
        from .cache import prepare_cache
        runtime = NativeRuntime(args.model_dir, device=args.device, encoder=True)
        adapter = None
        try:
            adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
            index_hash = prepare_cache(runtime, adapter, pairs, args.output,
                                       seeds=args.seeds, size=args.size, steps=args.steps)
            print(json.dumps(dict(cache=args.output, cache_index_sha256=index_hash)))
        finally:
            if adapter is not None:
                adapter.close()
            runtime.close()
        return

    from particle_sliders.lifecycle import SessionConfig, read_packet
    from .lifecycle import BACKEND
    config = SessionConfig.from_dict(json.loads(Path(args.config).read_text()))
    root = Path(args.output)
    if root.exists():
        if not args.resume or not (root / "session.json").is_file():
            raise FileExistsError("existing output requires explicit resume")
        if json.loads((root / "session.json").read_text()) != config.as_dict():
            raise ValueError("resume configuration must match the original session exactly")
    else:
        root.mkdir(parents=True)
        (root / "session.json").write_text(json.dumps(config.as_dict(), indent=2) + "\n")
    with BACKEND.open(config) as session:
        if args.resume:
            session.restore(read_packet(args.resume))
        def record(metrics):
            with (root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if metrics["step"] % 10 == 0:
                session.checkpoint(root / "training.pt")
        session.train(stop_after=args.stop_after, on_step=record)
        session.checkpoint(root / "training.pt")
        session.export_inference(root / "native-inference.pt", selection="ema")
    print(json.dumps(dict(output=str(root), checkpoint="training.pt", export="native-inference.pt")))


if __name__ == "__main__":
    main()
