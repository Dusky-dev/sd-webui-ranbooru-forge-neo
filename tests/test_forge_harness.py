"""Import the real scripts/ranbooru.py under Gradio with stand-ins for Forge
Neo's modules, build the UI, call the event handlers and run the generation
hooks end to end (no network: booru replies are recorded shapes).

The processing stand-in copies the Forge Neo rules that broke Ranbooru before:
prompt/negative-prompt list lengths (setup_prompts) and the img2img
init-image count ("bad number of images passed").

Needs gradio 4.40 (Forge Neo's pinned version). Without it the tests are
skipped, unless RANBOORU_REQUIRE_GRADIO=1 (set in CI) turns that into a
failure.
"""
import dataclasses
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import gradio as gr
    from forge_ui_stub import InputAccordion
except Exception as exc:  # pragma: no cover - depends on the environment
    gr = None
    GRADIO_ERROR = exc
else:
    GRADIO_ERROR = None

REPO = Path(__file__).resolve().parents[1]
REQUIRE = os.environ.get('RANBOORU_REQUIRE_GRADIO') == '1'


# --- Forge Neo stand-ins ------------------------------------------------------

class State:
    interrupted = False
    stopping_generation = False
    textinfo = ''


def build_forge_modules(extension_root, calls):
    modules = types.ModuleType('modules')

    scripts = types.ModuleType('modules.scripts')

    class Script:
        def elem_id(self, item_id):
            return item_id

    scripts.Script = Script
    scripts.AlwaysVisible = object()
    scripts.basedir = lambda: str(extension_root)

    @dataclasses.dataclass
    class StableDiffusionProcessing:
        prompt: object = ''
        negative_prompt: object = ''
        styles: list = None
        seed: object = -1
        subseed: int = -1
        subseed_strength: float = 0
        sampler_name: str = 'Euler'
        scheduler: str = 'Automatic'
        batch_size: int = 1
        n_iter: int = 1
        steps: int = 20
        cfg_scale: float = 7.0
        distilled_cfg_scale: float = 3.5
        width: int = 512
        height: int = 512
        override_settings: dict = None
        sd_model: object = None
        outpath_samples: str = None
        outpath_grids: str = None

        def setup_prompts(self):
            # Forge Neo modules/processing.py setup_prompts
            if isinstance(self.prompt, list):
                self.all_prompts = self.prompt
            elif isinstance(self.negative_prompt, list):
                self.all_prompts = [self.prompt] * len(self.negative_prompt)
            else:
                self.all_prompts = self.batch_size * self.n_iter * [self.prompt]
            if isinstance(self.negative_prompt, list):
                self.all_negative_prompts = self.negative_prompt
            else:
                self.all_negative_prompts = [self.negative_prompt] * len(self.all_prompts)
            if len(self.all_prompts) != len(self.all_negative_prompts):
                raise RuntimeError(
                    f'Received a different number of prompts ({len(self.all_prompts)}) '
                    f'and negative prompts ({len(self.all_negative_prompts)})')

    @dataclasses.dataclass
    class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
        init_images: list = None
        denoising_strength: float = 0.75

    class Processed:
        def __init__(self, p, images, infotexts, first):
            self.images, self.infotexts = images, infotexts
            self.index_of_first_image = first
            self.all_seeds = p.all_seeds

    def process_images(p):
        calls.append(p)
        p.setup_prompts()
        if isinstance(p.seed, list):
            p.all_seeds = p.seed
        else:
            p.all_seeds = [int(p.seed) + x for x in range(len(p.all_prompts))]
        if isinstance(p, StableDiffusionProcessingImg2Img):
            # Forge Neo StableDiffusionProcessingImg2Img.init
            count = len(p.init_images)
            if count == 1:
                pass
            elif count <= p.batch_size:
                p.batch_size = count
            else:
                raise RuntimeError(
                    f'bad number of images passed: {count}; expecting {p.batch_size} or less')
        total = len(p.all_prompts)
        images = [f'image{i}' for i in range(total)]
        infotexts = [f'info:{prompt}' for prompt in p.all_prompts]
        if total > 1:
            return Processed(p, ['grid'] + images, ['grid-info'] + infotexts, 1)
        return Processed(p, images, infotexts, 0)

    processing = types.ModuleType('modules.processing')
    processing.process_images = process_images
    processing.StableDiffusionProcessing = StableDiffusionProcessing
    processing.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2Img

    shared = types.ModuleType('modules.shared')
    shared.opts = types.SimpleNamespace(
        outdir_samples='', outdir_img2img_samples='out', outdir_grids='',
        outdir_img2img_grids='grids')
    shared.sd_model = 'model'
    shared.state = State()

    ui_components = types.ModuleType('modules.ui_components')
    ui_components.InputAccordion = InputAccordion

    modules.scripts, modules.processing = scripts, processing
    modules.shared, modules.ui_components = shared, ui_components
    return {
        'modules': modules, 'modules.scripts': scripts,
        'modules.processing': processing, 'modules.shared': shared,
        'modules.ui_components': ui_components,
    }


# --- Recorded booru replies (Safebooru shapes) -------------------------------

class Reply:
    def __init__(self, text='', data=None, content=b'', status=200, ctype='application/json'):
        self.text, self.data, self.content = text, data, content
        self.status_code = status
        self.headers = {'Content-Type': ctype}

    def json(self):
        if self.data is None:
            raise ValueError('no JSON')
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def png_bytes():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (64, 48), (200, 10, 10)).save(buffer, format='PNG')
    return buffer.getvalue()


def fake_get(url, **kwargs):
    if '/images/' in url:
        return Reply(content=png_bytes(), ctype='image/png')
    if 'limit=0' in url:
        return Reply(text='<?xml version="1.0"?><posts count="5000" offset="0"></posts>',
                     ctype='text/xml')
    posts = [
        {'id': 100 + i, 'image': f'{i}.png', 'directory': '12', 'score': i,
         'file_url': f'https://safebooru.org/images/12/{i}.png',
         'tags': f'1girl solo smile blue_hair post_{i} sitting'}
        for i in range(6)
    ]
    return Reply(text=json.dumps(posts), data=posts)


@unittest.skipIf(gr is None and not REQUIRE, f'gradio not installed ({GRADIO_ERROR})')
class ForgeHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if gr is None:
            raise AssertionError(f'gradio is required here: {GRADIO_ERROR}')
        cls.tmp = tempfile.mkdtemp(prefix='ranbooru-harness-')
        for name in ('pose_tag_groups.json', 'pose_name_tags.json'):
            shutil.copy(REPO / name, cls.tmp)
        cls.calls = []
        cls.saved_modules = {name: sys.modules.get(name) for name in (
            'modules', 'modules.scripts', 'modules.processing', 'modules.shared',
            'modules.ui_components')}
        sys.modules.update(build_forge_modules(cls.tmp, cls.calls))
        spec = importlib.util.spec_from_file_location(
            'ranbooru_harness', REPO / 'scripts' / 'ranbooru.py')
        cls.rb = importlib.util.module_from_spec(spec)
        with mock.patch('sys.stdout', new=io.StringIO()):
            spec.loader.exec_module(cls.rb)
        cls.state_file = cls.rb.RANBOORU_STATE_FILE
        assert cls.state_file.startswith(cls.tmp), cls.state_file

        cls.ui = {}
        for is_img2img in (False, True):
            with gr.Blocks() as demo:
                script = cls.rb.Script()
                controls = script.ui(is_img2img)
            cls.ui[is_img2img] = (demo, controls)

    @classmethod
    def tearDownClass(cls):
        for name, module in cls.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.calls.clear()
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    # -- helpers --

    def functions(self, demo):
        fns = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
        return list(fns)

    def handlers_for(self, demo, component):
        return [fn for fn in self.functions(demo)
                if any(target[0] == component._id for target in fn.targets)]

    def values(self, **overrides):
        _, controls = self.ui[False]
        names = self.param_names('before_process')
        values = {name: getattr(control, 'value', None) for name, control in zip(names, controls)}
        values.update(enabled=True, booru='safebooru', tags='1girl', max_pages=5,
                      use_cache=False, remove_tags='', api_key='', user_id='', post_id='',
                      use_img2img=False, use_ip=False, same_prompt=False,
                      only_poses=False, strip_clothing=False, shuffle_tags=False)
        values.update(overrides)
        return [values[name] for name in names]

    def param_names(self, hook):
        import inspect
        params = list(inspect.signature(getattr(self.rb.Script, hook)).parameters)
        return params[2:] if hook == 'before_process' else params[3:]

    def processing(self, batch=1, n_iter=1):
        p = self.rb.StableDiffusionProcessingImg2Img.__mro__[1](
            prompt='masterpiece', negative_prompt='lowres', batch_size=batch,
            n_iter=n_iter, seed=1000)
        p.scripts = types.SimpleNamespace(alwayson_scripts=[])
        p.script_args = ()
        return p

    def generate(self, batch=1, n_iter=1, **overrides):
        """before_process, the txt2img run, then postprocess, like Forge."""
        self.calls.clear()
        script = self.rb.Script()
        p = self.processing(batch, n_iter)
        with mock.patch.object(self.rb.requests, 'get', side_effect=fake_get), \
                mock.patch('sys.stdout', new=io.StringIO()):
            script.before_process(p, *self.values(**overrides))
            processed = self.rb.process_images(p)  # the txt2img run
            script.postprocess(p, processed, *self.values(**overrides))
        return p, processed

    # -- UI --

    def test_controls_match_both_hooks(self):
        for is_img2img, (_, controls) in self.ui.items():
            self.assertEqual(len(controls), len(self.param_names('before_process')))
            self.assertEqual(len(controls), len(self.param_names('postprocess')))

    def test_booru_dropdown_handlers_run(self):
        demo, controls = self.ui[False]
        booru = controls[self.param_names('before_process').index('booru')]
        handlers = self.handlers_for(demo, booru)
        self.assertGreaterEqual(len(handlers), 3)
        for value in ('gelbooru', 'safebooru', 'rule34', 'e621', 'danbooru'):
            for handler in handlers:
                result = handler.fn(value)
                updates = result if isinstance(result, (tuple, list)) else [result]
                for update in updates:
                    self.assertIsInstance(update, dict, (handler.fn, value))

    def test_browser_restore_with_old_settings(self):
        demo, _ = self.ui[False]
        restore = [fn for fn in self.functions(demo)
                   if getattr(fn.fn, '__name__', '') == 'restore_browser_page']
        self.assertEqual(len(restore), 1)
        old = {'tabs': {'txt2img': {'booru': 'rule34', 'enabled': True, 'max_pages': 7,
                                    'lora_enabled': True, 'chaos_mode': 'Chaos',
                                    'use_deepbooru': True, 'mature_rating': 'Explicit'}}}
        with open(self.state_file, 'w', encoding='utf-8') as handle:
            json.dump(old, handle)
        with mock.patch('sys.stdout', new=io.StringIO()):
            results = restore[0].fn()
        self.assertEqual(len(results), len(restore[0].outputs))

    # -- generation --

    def test_txt2img_batches(self):
        for batch, n_iter, same in ((1, 1, False), (2, 1, False), (2, 2, False), (2, 2, True)):
            p, processed = self.generate(batch, n_iter, same_prompt=same)
            prompts = p.prompt if isinstance(p.prompt, list) else [p.prompt]
            self.assertIn(len(prompts), (1, batch * n_iter))
            self.assertTrue(all('masterpiece' in prompt for prompt in prompts))

    def test_skip_tags_already_in_prompt(self):
        script = self.rb.Script()
        for skip, expect_removed in ((False, False), (True, True)):
            self.calls.clear()
            p = self.processing(1, 1)
            p.prompt = '(blue hair:1.2), <lora:style:0.8>, Smile'
            with mock.patch.object(self.rb.requests, 'get', side_effect=fake_get), \
                    mock.patch('sys.stdout', new=io.StringIO()):
                script.before_process(p, *self.values(skip_prompt_tags=skip))
            fetched = p.prompt.split('<lora:style:0.8>, Smile', 1)[1]
            self.assertEqual('blue_hair' not in fetched and 'smile' not in fetched,
                             expect_removed, fetched)
            self.assertIn('sitting', fetched)

    def test_img2img_every_batch_size_and_count(self):
        for batch, n_iter, last in ((1, 1, False), (2, 1, False), (1, 2, False),
                                    (2, 2, False), (2, 3, True)):
            with self.subTest(batch=batch, n_iter=n_iter, use_last_img=last):
                p, processed = self.generate(batch, n_iter, use_img2img=True, use_last_img=last)
                self.assertEqual(len(processed.images), len(processed.infotexts))
                txt2img = [c for c in self.calls if not hasattr(c, 'init_images')]
                img2img = [c for c in self.calls if hasattr(c, 'init_images')]
                # The txt2img run is the 1-step throwaway pass.
                self.assertEqual([c.steps for c in txt2img], [1])
                # One img2img pass per batch, each with its own slice.
                self.assertEqual(len(img2img), n_iter)
                self.assertTrue(all(c.steps == 20 and c.n_iter == 1 for c in img2img))
                self.assertTrue(all(len(c.init_images) == batch for c in img2img))
                self.assertEqual([s for c in img2img for s in c.all_seeds],
                                 txt2img[0].all_seeds)
                self.assertEqual([pr for c in img2img for pr in c.all_prompts],
                                 txt2img[0].all_prompts)
                # Results first (grid only for a single run), then the sources.
                generated = batch * n_iter + (1 if n_iter == 1 and batch > 1 else 0)
                sources = 1 if last else batch * n_iter
                self.assertEqual(len(processed.images), generated + sources)
                self.assertEqual(processed.index_of_first_image,
                                 1 if n_iter == 1 and batch > 1 else 0)
                self.assertTrue(all(not isinstance(i, str) for i in processed.images[generated:]))

    def test_interrupt_stops_between_passes(self):
        state = self.rb.shared.state
        original = self.rb.process_images

        def interrupting(p):
            result = original(p)
            if hasattr(p, 'init_images'):
                state.interrupted = True
            return result

        try:
            with mock.patch.object(self.rb, 'process_images', side_effect=interrupting):
                self.generate(1, 3, use_img2img=True)
        finally:
            state.interrupted = False
        self.assertEqual(len([c for c in self.calls if hasattr(c, 'init_images')]), 1)

    # -- Use cache --

    def test_use_cache_is_ranbooru_only_and_keeps_no_secrets(self):
        import requests
        import requests_cache
        from requests.adapters import BaseAdapter

        replies = []

        class FakeAdapter(BaseAdapter):
            def send(self, request, **kwargs):
                body = replies.pop(0)
                response = requests.Response()
                response.status_code = 200
                response._content = body.encode()
                from urllib3 import HTTPResponse
                response.raw = HTTPResponse(body=io.BytesIO(body.encode()), status=200,
                                            preload_content=False, request_url=request.url)
                response.headers['Content-Type'] = 'application/json'
                response.url = request.url
                response.request = request
                return response

            def close(self):
                pass

        http = self.rb.booru_http
        http.set_cache(True)
        try:
            self.assertFalse(requests_cache.patcher.is_installed())
            self.assertTrue(self.rb.RANBOORU_CACHE_FILE.startswith(self.tmp))
            http.session.mount('https://', FakeAdapter())
            http.session.cache.clear()
            url = 'https://api.rule34.xxx/index.php?page=dapi&json=1&tags=solo'

            # A refusal is not cached: adding the login gets a fresh reply.
            replies[:] = ['"Missing authentication. Go to api.rule34.xxx"', '[{"id": 1}]']
            self.assertIn('Missing', http.get(url + '&user_id=1&api_key=SECRET').text)
            first = http.get(url + '&user_id=1&api_key=SECRET')
            self.assertFalse(getattr(first, 'from_cache', False))
            again = http.get(url + '&user_id=1&api_key=SECRET')
            self.assertTrue(again.from_cache)

            # The stored request has no credentials.
            stored = ' '.join(str(r.request.url) + str(dict(r.request.headers))
                              for r in http.session.cache.responses.values())
            self.assertNotIn('SECRET', stored)

            # cache=False bypasses the cache (Gelbooru sort:random).
            with mock.patch.object(self.rb.requests, 'get', return_value='plain') as plain:
                self.assertEqual(http.get(url, cache=False), 'plain')
                plain.assert_called_once()
        finally:
            http.set_cache(False)
        self.assertIsNone(http.session)


if __name__ == '__main__':
    unittest.main()
