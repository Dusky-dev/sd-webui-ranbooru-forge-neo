"""Forge-independent tests of Replace stripped clothing."""
import ast
import json
import random
import re
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py'
GROUPS_PATH = MODULE_PATH.parents[1] / 'pose_tag_groups.json'
FUNCTIONS = {
    '_normalize_pose_tag', '_matches_structure', 'is_characteristic_tag',
    'load_pose_tag_groups', 'prompt_tag_key', 'prompt_tag_keys',
    'parse_tag_list', 'clothing_slot', 'swap_clothing_tags',
}


def load_helpers():
    parsed = ast.parse(MODULE_PATH.read_text(encoding='utf-8'))
    nodes = []
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith(('POSE_', 'CLOTHING_'))
            for t in node.targets
        ):
            nodes.append(node)
    found = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    if found != FUNCTIONS:
        raise AssertionError(f'Missing helpers: {FUNCTIONS - found}')
    namespace = {'json': json, 're': re, 'random': random}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(MODULE_PATH), 'exec'), namespace)
    namespace['pose_tag_groups'] = namespace['load_pose_tag_groups'](str(GROUPS_PATH))
    return namespace


class ReplaceClothingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = load_helpers()
        cls.ns = ns
        cls.slot = staticmethod(ns['clothing_slot'])
        cls.parse = staticmethod(ns['parse_tag_list'])
        is_char = ns['is_characteristic_tag']
        cls.is_clothing = staticmethod(
            lambda tag: is_char(tag, strip_traits=False, strip_clothing=True))

    def swap(self, tags, mine, seed=0, taken=()):
        return self.ns['swap_clothing_tags'](
            tags, self.parse(mine), self.is_clothing,
            rng=random.Random(seed), taken=taken)

    def test_slots(self):
        cases = {
            'white_shirt': 'top', 'hoodie': 'top', 'open_jacket': 'top',
            'red_topwear': 'top', 'pleated_skirt': 'bottom',
            'denim_shorts': 'bottom', 'boxers_(clothing)': 'bottom',
            'school_uniform': 'full', 'black_dress': 'full',
            'one-piece_swimsuit': 'full', 'white_thighhighs': 'legwear',
            'black_pantyhose': 'legwear', 'high_heels': 'footwear',
            'loafers': 'footwear', 'baseball_cap': 'headwear',
            'hair_ribbon': 'headwear', 'white_gloves': 'handwear',
            'earrings': 'accessory', 'choker': 'accessory',
            'Blue Hoodie': 'top', '(sneakers:1.2)': 'footwear',
        }
        for tag, slot in cases.items():
            key = self.ns['prompt_tag_key'](tag)
            self.assertEqual(self.slot(key), slot, tag)

    def test_same_kind_goes_where_the_outfit_was(self):
        tags = ['1girl', 'white_shirt', 'sitting', 'pleated_skirt', 'smile']
        out = self.swap(tags, 'hoodie, denim shorts')
        self.assertEqual(out, ['1girl', 'hoodie', 'denim shorts', 'sitting', 'smile'])
        tags = ['white_shirt', 'sitting', 'black_thighhighs', 'smile', 'loafers']
        out = self.swap(tags, 'hoodie, sneakers, kneehighs')
        self.assertEqual(out, ['hoodie', 'sitting', 'kneehighs', 'smile', 'sneakers'])

    def test_never_a_dress_and_a_top_together(self):
        for seed in range(20):
            out = self.swap(['school_uniform', 'white_shirt', 'pleated_skirt', 'sitting'],
                            'sundress, hoodie, jeans', seed=seed)
            self.assertEqual(out, ['sundress', 'sitting'])
            out = self.swap(['white_shirt', 'pleated_skirt', 'sitting'],
                            'hoodie, sundress', seed=seed)
            self.assertEqual(out, ['hoodie', 'sitting'])

    def test_apron_and_armor_are_not_outfits(self):
        self.assertEqual(self.slot('apron'), 'accessory')
        self.assertEqual(self.slot('shoulder_armor'), 'accessory')
        for tag in ('sweater_guard', 'shirt_stay', 'clitoral_hood_piercing',
                    'corset_piercing', 'glove_cutout'):
            self.assertEqual(self.slot(tag), 'accessory', tag)
        self.assertEqual(self.slot('hat_ornament'), 'headwear')
        self.assertEqual(self.slot('glove_cuffs'), 'handwear')

    def test_one_replacement_per_slot(self):
        tags = ['white_shirt', 'collared_shirt', 'long_sleeves', 'sitting']
        out = self.swap(tags, 'hoodie, tank_top')
        self.assertEqual(len([t for t in out if t in ('hoodie', 'tank_top')]), 1)
        self.assertNotIn('white_shirt', out)
        self.assertNotIn('collared_shirt', out)
        self.assertIn('sitting', out)

    def test_random_choice_among_same_slot(self):
        seen = {self.swap(['white_shirt'], 'hoodie, tank_top, sweater', seed=s)[0]
                for s in range(40)}
        self.assertEqual(seen, {'hoodie', 'tank_top', 'sweater'})

    def test_dress_stands_in_for_top_and_skirt_once(self):
        out = self.swap(['white_shirt', 'sitting', 'pleated_skirt'], 'sundress')
        self.assertEqual(out, ['sundress', 'sitting'])

    def test_top_and_bottom_stand_in_for_a_dress(self):
        out = self.swap(['black_dress', 'sitting'], 'hoodie, jeans')
        self.assertEqual(out, ['hoodie', 'jeans', 'sitting'])

    def test_random_fallback_for_body_garments(self):
        out = self.swap(['1girl', 'white_shirt', 'sitting'], 'sneakers')
        self.assertEqual(out, ['1girl', 'sneakers', 'sitting'])
        # The random fallback can be any of the user's tags.
        seen = {self.swap(['white_shirt'], 'sneakers, earrings', seed=s)[0]
                for s in range(40)}
        self.assertEqual(seen, {'sneakers', 'earrings'})

    def test_no_fallback_for_accessory_only_posts(self):
        out = self.swap(['nude', 'earrings', 'sitting'], 'hoodie')
        self.assertEqual(out, ['nude', 'sitting'])

    def test_unmatched_accessory_slot_is_just_dropped(self):
        out = self.swap(['white_shirt', 'earrings', 'sitting'], 'hoodie')
        self.assertEqual(out, ['hoodie', 'sitting'])

    def test_skips_tags_already_in_prompt_and_duplicates(self):
        out = self.swap(['white_shirt', 'sitting'], 'hoodie, Hoodie',
                        taken={'hoodie'})
        self.assertEqual(out, ['sitting'])
        out = self.swap(['white_shirt', 'pleated_skirt'], 'hoodie, hoodie')
        self.assertEqual(out, ['hoodie'])

    def test_empty_list_just_strips(self):
        self.assertEqual(self.swap(['white_shirt', 'sitting'], ' , '), ['sitting'])

    def test_replacement_text_is_kept_verbatim(self):
        out = self.swap(['white_shirt'], '(red hoodie:1.2)')
        self.assertEqual(out, ['(red hoodie:1.2)'])


class ReplaceClothingWiring(unittest.TestCase):
    def test_new_controls_follow_strip_clothing(self):
        source = MODULE_PATH.read_text(encoding='utf-8')
        parsed = ast.parse(source)
        script = next(n for n in parsed.body
                      if isinstance(n, ast.ClassDef) and n.name == 'Script')
        methods = {n.name: n for n in script.body if isinstance(n, ast.FunctionDef)}
        before = [a.arg for a in methods['before_process'].args.args][2:]
        i = before.index('strip_clothing')
        self.assertEqual(before[i:i + 3],
                         ['strip_clothing', 'replace_clothing', 'clothing_tags'])
        self.assertIn("'replace_clothing',", source)
        self.assertIn("'clothing_tags',", source)


if __name__ == '__main__':
    unittest.main()
