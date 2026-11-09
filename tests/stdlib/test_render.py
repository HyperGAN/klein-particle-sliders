"""Source/CLI driver checks using explicit fake components; no native qualification."""
import ast
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from klein_sliders.cli import main
from klein_sliders.render import (native_pipeline_state, render_image, render_pair,
                                  validate_render_request)
ROOT = Path(__file__).resolve().parents[2]


class Image:
    def __init__(self, size=(256,256)):
        self.size=size;self.width,self.height=size;self.mode="RGB"
    def save(self, stream, format):
        assert format=="PNG"
        stream.write(b"EXPLICIT MOCK OUTPUT, NOT A REAL NATIVE PNG")


class Module:
    def requires_grad_(self, value):
        self.frozen=value is False;return self
    def eval(self):self.training=False;return self


class Pipeline:
    def __init__(self, transformer):
        self.transformer=transformer;self.text_encoder=Module();self.vae=Module()
        self.scheduler=SimpleNamespace(marker="initial");self.calls=[]
    def to(self, device):self.device=device;return self
    def __call__(self, **kwargs):
        self.calls.append(kwargs);self.scheduler.marker="mutated";self._num_timesteps=kwargs["num_inference_steps"]
        return SimpleNamespace(images=[Image((kwargs["width"],kwargs["height"]))])


class Adapter:
    strength=.75
    @contextlib.contextmanager
    def scaled(self,value):
        old=self.strength;self.strength=value
        try:yield
        finally:self.strength=old


class Generator:
    def __init__(self,device):self.device=device
    def manual_seed(self,seed):self.seed=seed;return self


def load_method():
    path=ROOT/"src/klein_sliders/native.py";tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="NativeRuntime")
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="load_renderer")
    scope=dict(Path=Path,__file__=str(path),file_hash=lambda unused:"fixed Source SHA")
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),scope)
    return scope["load_renderer"]


class NativeRenderContracts(unittest.TestCase):
    def request(self,**extra):
        value=dict(prompt="A watering can",height=256,width=256,steps=8,seed=7,strength=1.)
        value.update(extra);return value
    def test_finite_render_scope_and_early_refusals(self):
        validate_render_request(**self.request())
        for extra in [dict(height=1024),dict(width=255),dict(steps=1),dict(seed=-1),
                      dict(seed=True),dict(strength=float("nan")),dict(strength=-.1),
                      dict(strength=2),dict(prompt=" "),dict(images=[]),dict(output_type="latent")]:
            with self.subTest(extra=extra),self.assertRaises(ValueError):validate_render_request(**self.request(**extra))
        with patch.dict(sys.modules,{"torch":None,"klein_sliders.lifecycle":None}):
            with self.assertRaises(ValueError):render_pair(artifact="missing",model_dir="missing",output="missing",**dict(prompt="valid",size=1024))
    def test_one_actual_transformer_reused_in_public_loader(self):
        transformer=Module();pipe=Pipeline(transformer);calls=[]
        def from_pretrained(*args,**kw):calls.append((args,kw));return pipe
        runtime=SimpleNamespace(pipe=None,helper=SimpleNamespace(from_pretrained=from_pretrained),
            transformer=transformer,identity={"finished_renderer_source_sha256":"fixed Source SHA"},
            model_directory="exact local stage",dtype="BF16",device="cuda:0")
        method=load_method();self.assertIs(method(runtime),pipe);self.assertIs(method(runtime),pipe)
        self.assertEqual(len(calls),1);self.assertIs(calls[0][1]["transformer"],transformer)
        self.assertIs(calls[0][1]["local_files_only"],True);self.assertIs(calls[0][1]["is_distilled"],False)
        self.assertEqual(pipe.device,"cuda:0");self.assertTrue(all(m.frozen and not m.training for m in [transformer,pipe.text_encoder,pipe.vae]))
        runtime.identity["finished_renderer_source_sha256"]="changed"
        with self.assertRaises(RuntimeError):method(runtime)
        runtime.identity["finished_renderer_source_sha256"]="fixed Source SHA";runtime.pipe=Pipeline(Module())
        with self.assertRaises(RuntimeError):method(runtime)
    def test_scheduler_identity_and_state_restored_after_error(self):
        pipe=Pipeline(Module());original=pipe.scheduler;pipe._guidance_scale=4.
        with self.assertRaisesRegex(RuntimeError,"synthetic"):
            with native_pipeline_state(pipe):
                self.assertIsNot(pipe.scheduler,original);pipe.scheduler.marker="mutated";pipe._guidance_scale=1.;pipe._num_timesteps=8;raise RuntimeError("synthetic")
        self.assertIs(pipe.scheduler,original);self.assertEqual(original.marker,"initial");self.assertEqual(pipe._guidance_scale,4.);self.assertNotIn("_num_timesteps",vars(pipe))
    def test_actual_driver_preserves_native_call_profile_and_strength(self):
        transformer=Module();pipe=Pipeline(transformer);runtime=SimpleNamespace(transformer=transformer,device="cuda:0",load_renderer=lambda:pipe);adapter=Adapter();torch=ModuleType("torch");torch.Generator=Generator;pil=ModuleType("PIL.Image");pil.Image=Image
        with patch.dict(sys.modules,{"torch":torch,"PIL.Image":pil}):
            for strength in [0.,.25]:
                image=render_image(runtime,adapter,**self.request(strength=strength))
                self.assertEqual(image.size,(256,256));self.assertEqual(adapter.strength,.75)
        self.assertEqual(len(pipe.calls),2)
        for call in pipe.calls:
            self.assertEqual(call["guidance_scale"],1.);self.assertEqual(call["max_sequence_length"],128)
            self.assertEqual(call["text_encoder_out_layers"],(9,18,27));self.assertIsNone(call["image"])
            self.assertEqual(call["generator"].seed,7);self.assertEqual(call["generator"].device,"cuda:0")
        self.assertIsNot(pipe.calls[0]["generator"],pipe.calls[1]["generator"])
        self.assertEqual(pipe.scheduler.marker,"initial")
    def test_actual_driver_cleanup_when_public_pipeline_raises(self):
        class Broken(Pipeline):
            def __call__(self,**kwargs):self.scheduler.marker="bad";raise RuntimeError("synthetic native call failure")
        transformer=Module();pipe=Broken(transformer);adapter=Adapter();runtime=SimpleNamespace(transformer=transformer,device="cpu",load_renderer=lambda:pipe);torch=ModuleType("torch");torch.Generator=Generator
        with patch.dict(sys.modules,{"torch":torch}),self.assertRaisesRegex(RuntimeError,"native call failure"):
            render_image(runtime,adapter,**self.request())
        self.assertEqual(adapter.strength,.75);self.assertEqual(pipe.scheduler.marker,"initial")
    def test_pair_uses_recorded_RAW_bytes_one_session_and_preserves_partial(self):
        requests=[];opens=[];loads=[];closed=[]
        class Session:
            selection="raw"
            def __enter__(self):return self
            def __exit__(self,*unused):closed.append(True)
            def render(self,**kw):
                requests.append(kw)
                if kw["strength"]==.5:raise RuntimeError("synthetic ON failure")
                return Image()
        def open_inference(packet,**kw):opens.append((packet,kw));return Session()
        shared=ModuleType("particle_sliders.inference");shared.open_inference=open_inference;product=ModuleType("klein_sliders.lifecycle");product.NATIVE_BACKEND=object();torch=ModuleType("torch")
        def load(stream,**kw):loads.append((stream.read(),kw));return dict(selection="raw",sha256="internal",origin={"cohort":"synthetic"},specification={"synthetic":True})
        torch.load=load
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);artifact=root/"synthetic.pt";artifact.write_bytes(b"EXPLICIT SYNTHETIC PACKET BYTES");out=root/"pair"
            with patch.dict(sys.modules,{"torch":torch,"particle_sliders.inference":shared,"klein_sliders.lifecycle":product}):
                with self.assertRaisesRegex(RuntimeError,"synthetic ON failure"):
                    render_pair(artifact=artifact,model_dir=root,output=out,prompt="A watering can",size=256,strength=.5)
            self.assertEqual(len(opens),1);self.assertEqual([r["strength"] for r in requests],[0.,.5]);self.assertTrue(closed)
            receipt=json.loads((out/"render-receipt.json").read_text());self.assertEqual(receipt["status"],"failed");self.assertEqual(len(receipt["images"]),1);self.assertIsNone(receipt["native_qualification"]);self.assertFalse(receipt["teacher_cache_used"])
            self.assertEqual(loads[0][0],artifact.read_bytes());self.assertIs(loads[0][1]["weights_only"],True)
            with self.assertRaises(FileExistsError):render_pair(artifact=artifact,model_dir=root,output=out,prompt="valid")
            with patch.dict(sys.modules,{"torch":torch,"particle_sliders.inference":shared,"klein_sliders.lifecycle":product}):
                complete=render_pair(artifact=artifact,model_dir=root,output=root/"complete",prompt="A watering can",size=256,strength=.25)
            self.assertEqual(complete["status"],"complete");self.assertEqual([r["label"] for r in complete["images"]],["OFF","ON"])
            self.assertEqual([r["strength"] for r in complete["images"]],[0.,.25]);self.assertEqual(len(opens),2)
    def test_EMA_packet_refused_before_native_factory(self):
        torch=ModuleType("torch");torch.load=lambda *a,**k:{"selection":"ema"};shared=ModuleType("particle_sliders.inference");shared.open_inference=lambda *a,**k:(_ for _ in ()).throw(AssertionError("no native open"));product=ModuleType("klein_sliders.lifecycle");product.NATIVE_BACKEND=object()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/"synthetic.pt";p.write_bytes(b"synthetic")
            with patch.dict(sys.modules,{"torch":torch,"particle_sliders.inference":shared,"klein_sliders.lifecycle":product}),self.assertRaisesRegex(ValueError,"RAW-only"):
                render_pair(artifact=p,model_dir=root,output=root/"out",prompt="valid")
    def test_render_CLI_dispatch_is_explicit_and_cache_free(self):
        from klein_sliders import render
        with patch.object(render,"render_pair",return_value={"status":"MOCK"}) as call,contextlib.redirect_stdout(io.StringIO()):
            main(["render","--model-dir","stage","--export","raw.pt","--output","fresh","--prompt","A watering can","--device","cuda:0","--size","256","--seed","19","--strength","0.25"])
        self.assertEqual(call.call_args.kwargs,dict(artifact="raw.pt",model_dir="stage",output="fresh",prompt="A watering can",seed=19,size=256,steps=8,strength=.25,device="cuda:0",precision="bfloat16"))

if __name__=="__main__":unittest.main()
