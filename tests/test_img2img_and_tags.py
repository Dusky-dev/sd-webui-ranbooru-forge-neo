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


class RemovedFeaturesTests(unittest.TestCase):
    """LoRAnado, the File panel, Deepbooru and Extra (except Use cache) are gone."""

    def test_no_leftovers(self):
        source = SOURCE.read_text(encoding='utf-8')
        for name in ('lora_enabled', 'loranado', 'use_search_txt', 'use_remove_txt',
                     'use_deepbooru', 'from modules import deepbooru', 'mix_prompt',
                     'chaos_mode', 'negative_mode', 'use_same_seed', 'model_hijack',
                     'sd-webui-controlnet'):
            self.assertNotIn(name, source)
        self.assertIn("'use_cache'", source)


if __name__ == '__main__':
    unittest.main()
