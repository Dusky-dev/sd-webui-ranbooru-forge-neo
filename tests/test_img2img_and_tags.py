"""Forge-independent tests of tag removal, the img2img second pass helpers and
Send to ControlNet (Forge Neo's built-in ControlNet).

    python -m unittest discover -s tests -v
"""
import ast
import copy
import dataclasses
import re
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py'
NAMES = {
    '_normalize_pose_tag', 'parse_tag_list', 'remove_listed_tags',
    'prompt_tag_key', 'prompt_tag_keys',
    'img2img_settings_from', 'source_infotext', 'send_image_to_controlnet',
}


def load(namespace):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in NAMES)
            or (isinstance(n, ast.Assign) and any(
                getattr(t, 'id', None) == 'IMG2IMG_COPIED_SETTINGS' for t in n.targets))]
    if len(body) != len(NAMES) + 1:
        raise AssertionError('Unable to isolate the helpers under test')
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    return namespace


class TagRemovalTests(unittest.TestCase):
    def setUp(self):
        self.ns = load({'re': re})

    def test_parse_trims_and_splits_lines(self):
        self.assertEqual(self.ns['parse_tag_list']('solo, 1girl ,\n smile\n\n,'),
                         ['solo', '1girl', 'smile'])
        self.assertEqual(self.ns['parse_tag_list'](''), [])
        self.assertEqual(self.ns['parse_tag_list'](None), [])

    def test_entries_after_a_space_are_removed(self):
        tags = ['1girl', ' solo', ' smile', ' long_hair']
        bad = self.ns['parse_tag_list']('solo, 1girl')
        self.assertEqual(self.ns['remove_listed_tags'](tags, bad), [' smile', ' long_hair'])

    def test_wildcard_and_spelling(self):
        tags = ['1girl', ' short_hair', ' long hair', ' hair_ornament', ' Blue_Eyes', ' smile']
        bad = self.ns['parse_tag_list']('*hair, blue eyes')
        self.assertEqual(self.ns['remove_listed_tags'](tags, bad), ['1girl', ' smile'])

    def test_lone_star_and_empty_entries_remove_nothing(self):
        tags = ['1girl', ' solo', '']
        self.assertEqual(self.ns['remove_listed_tags'](tags, ['*', '', ' ']), ['1girl', ' solo'])


class SkipPromptTagsTests(unittest.TestCase):
    def setUp(self):
        self.ns = load({'re': re})
        self.key = self.ns['prompt_tag_key']

    def test_attention_syntax_is_ignored(self):
        for typed in ('(blue hair:1.2)', 'blue hair', '((Blue_Hair))', '[blue hair]',
                      '(blue_hair: 0.8)'):
            self.assertEqual(self.key(typed), 'blue_hair', typed)

    def test_escaped_parentheses_are_part_of_the_name(self):
        self.assertEqual(self.key(r'hatsune miku \(cosplay\)'), 'hatsune_miku_(cosplay)')
        self.assertEqual(self.key(r'hatsune_miku_\(cosplay\)'), 'hatsune_miku_(cosplay)')

    def test_extra_networks_and_empty(self):
        for tag in ('<lora:style:0.7>', '<hypernet:x:1>', '', '  ', '()'):
            self.assertIsNone(self.key(tag), tag)

    def test_prompt_keys_split_groups(self):
        keys = self.ns['prompt_tag_keys']('masterpiece, (red eyes, smile:1.1), <lora:a:1>, solo')
        self.assertEqual(keys, {'masterpiece', 'red_eyes', 'smile', 'solo'})
        self.assertEqual(self.ns['prompt_tag_keys'](['a', 'b, c']), {'a', 'b', 'c'})


@dataclasses.dataclass
class FakeImg2Img:
    prompt: str = ''
    negative_prompt: str = ''
    styles: list = None
    seed: int = -1
    subseed: int = -1
    cfg_scale: float = 7.0
    override_settings: dict = None
    init_images: list = None
    denoising_strength: float = 0.75
    internal: int = dataclasses.field(default=0, init=False)


class Img2ImgPassTests(unittest.TestCase):
    def setUp(self):
        self.ns = load({'re': re, 'dataclasses': dataclasses})

    def test_copies_only_settings_the_class_accepts(self):
        txt2img = types.SimpleNamespace(
            prompt='a', negative_prompt='b', styles=['s'], seed=5, subseed=6,
            cfg_scale=4.0, override_settings={'CLIP_stop_at_last_layers': 2},
            distilled_cfg_scale=3.5, scripts='never copied', enable_hr=True)
        settings = self.ns['img2img_settings_from'](txt2img, FakeImg2Img)
        self.assertEqual(settings, {
            'prompt': 'a', 'negative_prompt': 'b', 'styles': ['s'], 'seed': 5,
            'subseed': 6, 'cfg_scale': 4.0,
            'override_settings': {'CLIP_stop_at_last_layers': 2}})
        FakeImg2Img(**settings)

    def test_source_infotext_follows_the_grid_offset(self):
        source = self.ns['source_infotext']
        self.assertEqual(source(['grid', 'a', 'b'], 1, 1), 'b')
        # No grid (single image, or grids off): the old num + 1 went past the end.
        self.assertEqual(source(['a'], 0, 0), 'a')
        self.assertEqual(source(['a', 'b'], 0, 5), 'b')
        self.assertEqual(source([], 0, 0), '')


@dataclasses.dataclass
class Unit:
    enabled: bool = False
    model: str = 'None'
    weight: float = 1.0
    image: object = None
    use_preview_as_input: bool = False


class Image:
    def __init__(self, pixels):
        self.pixels = pixels
        self.converted = None

    def convert(self, mode):
        self.converted = mode
        return self


class Script:
    def __init__(self, title, start, end):
        self._title, self.args_from, self.args_to = title, start, end

    def title(self):
        return self._title


class ControlNetTests(unittest.TestCase):
    def setUp(self):
        numpy = types.SimpleNamespace(asarray=lambda img: ('array', img.pixels))
        self.ns = load({'re': re, 'dataclasses': dataclasses, 'copy': copy, 'np': numpy})
        self.send = self.ns['send_image_to_controlnet']

    def processing(self, units, scripts=None):
        args = ('ranbooru', 'args') + tuple(units) + ('other',)
        scripts = scripts or [Script('Ranbooru', 0, 2),
                              Script('ControlNet', 2, 2 + len(units))]
        return types.SimpleNamespace(scripts=types.SimpleNamespace(alwayson_scripts=scripts),
                                     script_args=args)

    def test_first_unit_gets_the_image_and_weight(self):
        unit1, unit2 = Unit(model='canny'), Unit(model='depth')
        p = self.processing([unit1, unit2])
        self.send(p, Image('px'), 0.6)
        self.assertIsInstance(p.script_args, tuple)
        new = p.script_args[2]
        self.assertTrue(new.enabled)
        self.assertEqual(new.weight, 0.6)
        self.assertEqual(new.image, {'image': ('array', 'px'), 'mask': None})
        self.assertEqual(new.model, 'canny')
        self.assertIs(p.script_args[3], unit2)
        self.assertEqual(p.script_args[:2], ('ranbooru', 'args'))
        self.assertEqual(p.script_args[-1], 'other')
        self.assertFalse(unit1.enabled)  # the UI's unit object is not mutated

    def test_dict_units_from_the_api(self):
        p = self.processing([{'enabled': False, 'model': 'canny'}])
        self.send(p, Image('px'), 1.0)
        self.assertEqual(p.script_args[2]['image']['image'], ('array', 'px'))
        self.assertTrue(p.script_args[2]['enabled'])

    def test_missing_controlnet_is_a_readable_error(self):
        p = self.processing([], scripts=[Script('Ranbooru', 0, 2)])
        with self.assertRaisesRegex(RuntimeError, "built-in ControlNet"):
            self.send(p, Image('px'), 1.0)
        p = self.processing([], scripts=[Script('ControlNet', 2, 2)])
        with self.assertRaisesRegex(RuntimeError, 'no ControlNet unit'):
            self.send(p, Image('px'), 1.0)


def load_image_helpers(namespace):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {'orientation_size', 'image_versions', 'image_download_urls',
             'images_for_posts'}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(body) != len(names):
        raise AssertionError('Unable to isolate the image helpers')
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    return namespace


class ImageVersionTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_image_helpers({})
        self.urls = self.ns['image_download_urls']

    def test_sample_used_only_when_it_covers_the_output(self):
        post = {'file_url': 'https://x/orig.png', 'sample_url': 'https://x/s.jpg',
                'sample_width': 850, 'sample_height': 1043}
        # Portrait: output 512 x 768 without Crop center.
        self.assertEqual(self.urls(post), ['https://x/s.jpg', 'https://x/orig.png'])
        self.assertEqual(self.urls(post, (832, 1024)), ['https://x/s.jpg', 'https://x/orig.png'])
        # 1024 x 1024 with Crop center: an 850 px sample would be upscaled.
        self.assertEqual(self.urls(post, (1024, 1024)), ['https://x/orig.png'])

    def test_smallest_sufficient_first(self):
        post = {'file_url': 'https://k/orig.png',
                'sample_url': 'https://k/sample.jpg', 'sample_width': 1500, 'sample_height': 1125,
                'jpeg_url': 'https://k/jpeg.jpg', 'jpeg_width': 3500, 'jpeg_height': 2625}
        self.assertEqual(self.urls(post, (1024, 768)),
                         ['https://k/sample.jpg', 'https://k/jpeg.jpg', 'https://k/orig.png'])
        self.assertEqual(self.urls(post, (2048, 1536)),
                         ['https://k/jpeg.jpg', 'https://k/orig.png'])

    def test_api_shapes(self):
        versions = self.ns['image_versions']
        # Safebooru with no sample: width 0 and the original URL.
        self.assertEqual(versions({'file_url': 'https://s/a.jpg', 'sample': False,
                                   'sample_url': 'https://s/a.jpg', 'sample_width': 0,
                                   'sample_height': 0}), [])
        danbooru = {'file_url': 'https://d/o.png', 'media_asset': {'variants': [
            {'type': '720x720', 'url': 'https://d/720.webp', 'width': 720, 'height': 516},
            {'type': 'sample', 'url': 'https://d/s.jpg', 'width': 850, 'height': 609},
            {'type': 'original', 'url': 'https://d/o.png', 'width': 2529, 'height': 1812}]}}
        self.assertEqual(versions(danbooru), [('https://d/s.jpg', 850, 609)])
        e621 = {'file_url': 'https://e/o.png', 'files': {'sample': {
            'width': 1133, 'height': 850, 'jpg': 'https://e/s.jpg', 'webp': 'https://e/s.webp'}}}
        self.assertEqual(versions(e621), [('https://e/s.jpg', 1133, 850)])
        legacy = {'file_url': 'https://e/o.png',
                  'sample': {'url': 'https://e/s2.jpg', 'width': 850, 'height': 600}}
        self.assertEqual(versions(legacy), [('https://e/s2.jpg', 850, 600)])
        # Gelbooru: sample flag 0 and an empty sample URL.
        self.assertEqual(versions({'file_url': 'https://g/o.png', 'sample': 0,
                                   'sample_url': '', 'sample_width': 0}), [])
        self.assertEqual(self.urls({'file_url': None}), [])


class ImagesForPostsTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_image_helpers({'random': types.SimpleNamespace(choice=lambda xs: xs[0])})
        self.pick = self.ns['images_for_posts']

    def load(self, failing):
        def load(post):
            if post['id'] in failing:
                raise RuntimeError('404')
            return f"img{post['id']}"
        return load

    def run_quiet(self, *args, **kwargs):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return self.pick(*args, **kwargs)

    def test_failed_post_is_swapped_and_its_prompt_follows(self):
        pool = [{'id': n} for n in range(5)]
        posts = [pool[0], pool[1]]
        images = self.run_quiet(posts, [0, 1], pool, self.load({1, 2}))
        self.assertEqual([p['id'] for p in posts], [0, 3])
        self.assertEqual(images, {0: 'img0', 1: 'img3'})

    def test_same_prompt_post_is_swapped_everywhere(self):
        pool = [{'id': n} for n in range(3)]
        posts = [pool[0]] * 3
        images = self.run_quiet(posts, [0, 1, 2], pool, self.load({0}))
        self.assertEqual([p['id'] for p in posts], [1, 1, 1])
        self.assertEqual(set(images.values()), {'img1'})

    def test_no_spare_reuses_a_loaded_pair(self):
        pool = [{'id': 0}, {'id': 1}]
        posts = list(pool)
        images = self.run_quiet(posts, [0, 1], pool, self.load({1}))
        self.assertEqual([p['id'] for p in posts], [0, 0])
        self.assertEqual(images, {0: 'img0', 1: 'img0'})

    def test_only_needed_posts_are_loaded(self):
        pool = [{'id': n} for n in range(3)]
        posts = list(pool)
        images = self.run_quiet(posts, [2], pool, self.load(set()))
        self.assertEqual(images, {2: 'img2'})

    def test_errors(self):
        pool = [{'id': 0}, {'id': 1}]
        with self.assertRaisesRegex(RuntimeError, 'none of the 2 booru image'):
            self.run_quiet(list(pool), [0, 1], pool, self.load({0, 1}))
        with self.assertRaisesRegex(RuntimeError, 'image of post 7 could not be downloaded'):
            self.run_quiet([{'id': 7}], [0], [{'id': 7}], self.load({7}), fixed=True)
        # Swaps are bounded.
        pool = [{'id': n} for n in range(50)]
        calls = []

        def failing(post):
            calls.append(post)
            raise RuntimeError('404')
        with self.assertRaises(RuntimeError):
            self.run_quiet([pool[0]], [0], pool, failing, max_swaps=3,
                           max_failures_first=100)
        self.assertEqual(len(calls), 4)
        # Host down: stop after the first few failures, for all images.
        calls.clear()
        with self.assertRaisesRegex(RuntimeError, 'none of the 4'):
            self.run_quiet(pool[:4], [0, 1, 2, 3], pool, failing)
        self.assertEqual(len(calls), 3)
        # Once one image loaded, swaps continue as usual.
        pool = [{'id': n} for n in range(10)]
        posts = [pool[0], pool[1]]
        self.run_quiet(posts, [0, 1], pool, self.load({1, 2, 3, 4, 5}))
        self.assertEqual([p['id'] for p in posts], [0, 6])


class RemovedFeaturesTests(unittest.TestCase):
    """LoRAnado, the File panel, Deepbooru and Extra (except Use cache) are gone."""

    def test_no_leftovers(self):
        source = SOURCE.read_text(encoding='utf-8')
        for name in ('lora_enabled', 'loranado', 'use_search_txt', 'use_remove_txt',
                     'use_deepbooru', 'from modules import deepbooru', 'mix_prompt',
                     'chaos_mode', 'negative_mode', 'use_same_seed', 'model_hijack',
                     'sd-webui-controlnet', 'pic.re', 'Yandere', "'yande.re'"):
            self.assertNotIn(name, source)
        self.assertIn("'use_cache'", source)


if __name__ == '__main__':
    unittest.main()
