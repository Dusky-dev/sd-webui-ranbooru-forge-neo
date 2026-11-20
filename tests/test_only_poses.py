"""Forge-independent tests of the Only Poses / clothing tag filter."""
import ast
import json
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py'
FUNCTIONS = {
    '_normalize_pose_tag', '_matches_structure', 'is_characteristic_tag',
    'filter_pose_tags', 'post_category_tags', 'load_pose_tag_groups',
    'load_pose_name_tags', 'pose_names_for',
}
GROUPS_PATH = MODULE_PATH.parents[1] / 'pose_tag_groups.json'
NAMES_PATH = MODULE_PATH.parents[1] / 'pose_name_tags.json'


def load_pose_filter():
    parsed = ast.parse(MODULE_PATH.read_text(encoding='utf-8'))
    nodes = []
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith('POSE_') for t in node.targets
        ):
            nodes.append(node)
    found = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    if found != FUNCTIONS:
        raise AssertionError(f'Missing pose helpers: {FUNCTIONS - found}')
    namespace = {'json': json}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(MODULE_PATH), 'exec'), namespace)
    namespace['pose_tag_groups'] = namespace['load_pose_tag_groups'](str(GROUPS_PATH))
    namespace['pose_name_tags'] = namespace['load_pose_name_tags'](str(NAMES_PATH))
    return namespace


class TestOnlyPoses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = load_pose_filter()
        cls.is_char = staticmethod(ns['is_characteristic_tag'])
        cls.filter = staticmethod(ns['filter_pose_tags'])
        cls.categories = staticmethod(ns['post_category_tags'])
        cls.load_groups = staticmethod(ns['load_pose_tag_groups'])
        cls.groups = ns['pose_tag_groups']
        cls.names = ns['pose_name_tags']
        cls.load_names = staticmethod(ns['load_pose_name_tags'])
        cls.pose_names_for = staticmethod(ns['pose_names_for'])

    def test_strips_appearance_traits(self):
        for tag in (
            'long_hair', 'very_long_hair', 'blue_eyes', 'light_brown_hair',
            'hair_between_eyes', 'twintails', 'side_ponytail', 'ahoge',
            'blunt_bangs', 'multicolored_hair', 'large_breasts', 'cat_ears',
            'fox_tail', 'tail', 'dark_skin', 'dark-skinned_female',
            'heterochromia', 'demon_horns', 'slit_pupils', 'grey_eyes',
            'ponytail', 'braid', 'cat_tail', 'cat_girl', 'blue_halo', 'horns',
            'wings', 'fang', 'tsurime', 'mole_under_eye', 'thick_thighs',
        ):
            self.assertTrue(self.is_char(tag), tag)

    def test_keeps_pose_expression_and_composition(self):
        for tag in (
            'closed_eyes', 'half-closed_eyes', 'covering_eyes', 'one_eye_closed',
            'hand_in_own_hair', 'adjusting_hair', 'floating_hair', 'wet_hair',
            'arms_up', 'looking_at_viewer', '1girl', 'solo', 'multiple_girls',
            'sitting', 'smile', 'open_mouth', 'breasts', 'hair', 'eyes',
            'hand_on_hip', 'holding_own_tail', 'tail_wagging', 'from_side',
            'skirt_lift', 'hand_in_pocket', 'white_shirt', 'pleated_skirt',
            'parted_lips', 'v-shaped_eyebrows', 'looking_back', 'crossed_arms',
            'magical_girl', 'multiple_boys', 'hand_on_own_chest',
        ):
            self.assertFalse(self.is_char(tag), tag)

    def test_prompt_formatting_variants(self):
        # Prompt tags arrive with leading spaces, escaped parens or spaces.
        self.assertTrue(self.is_char(' long_hair'))
        self.assertTrue(self.is_char('blue eyes'))
        self.assertTrue(self.is_char(r'someone\_(artist\)'.replace(r'\_', '_')))
        self.assertTrue(self.is_char(r' hatsune_miku', excluded={'hatsune_miku'}))
        self.assertTrue(self.is_char(r'kafka_\(honkai:_star_rail\)',
                                     excluded={'kafka_(honkai:_star_rail)'}))
        self.assertFalse(self.is_char(r'bow_\(weapon\)', strip_clothing=True))
        self.assertFalse(self.is_char('', True, True))

    def test_clothing_is_a_separate_toggle(self):
        for tag in ('white_shirt', 'pleated_skirt', 'black_thighhighs',
                    'school_uniform', 'hair_ribbon', 'long_sleeves', 'glasses',
                    'red_bowtie', 'frilled_dress', 'black_thighhighs',
                    'blue_skirt', 'striped_thighhighs', 'jewelry', 'panties'):
            self.assertFalse(self.is_char(tag, strip_clothing=False), tag)
            self.assertTrue(self.is_char(tag, strip_traits=False, strip_clothing=True), tag)
        for tag in ('skirt_lift', 'hand_in_pocket', 'holding_hat',
                    'shirt_lift', 'no_shirt', 'boxing_ring', 'sitting',
                    'clothes_lift', 'hood_down', 'holding_umbrella', 'umbrella',
                    'adjusting_eyewear', 'fingernails'):
            self.assertFalse(self.is_char(tag, strip_traits=False, strip_clothing=True), tag)

    def test_keep_groups_are_never_stripped(self):
        self.assertGreater(len(self.groups['keep']), 500)
        for tag in self.groups['keep']:
            self.assertFalse(self.is_char(tag, True, True), tag)

    def test_generated_lists_loaded(self):
        self.assertGreater(len(self.groups['traits']), 300)
        self.assertGreater(len(self.groups['clothing']), 800)
        self.assertFalse(self.groups['traits'] & self.groups['keep'])

    def test_missing_lists_fall_back_to_rules(self):
        empty = self.load_groups(str(GROUPS_PATH) + '.missing')
        self.assertTrue(self.is_char('blue_eyes', groups=empty))
        self.assertTrue(self.is_char('cat_tail', groups=empty))
        self.assertFalse(self.is_char('closed_eyes', groups=empty))
        self.assertFalse(self.is_char('white_shirt', True, True, groups=empty))

    def test_name_list_strips_only_when_enabled(self):
        self.assertGreater(len(self.names), 20000)
        for tag in ('hatsune_miku', 'touhou', 'genshin_impact', 'hakurei_reimu'):
            self.assertTrue(self.is_char(tag, names=self.names), tag)
            self.assertFalse(self.is_char(tag), tag)
        self.assertTrue(self.is_char(r' kirisame_marisa', names=self.names))
        # Name tier needs Only poses itself.
        self.assertFalse(self.is_char('hatsune_miku', False, True, names=self.names))

    def test_name_list_never_hits_keep_or_common_tags(self):
        self.assertFalse(self.names & self.groups['keep'])
        for tag in self.groups['keep']:
            self.assertFalse(self.is_char(tag, True, True, names=self.names), tag)
        for tag in ('1girl', 'solo', 'sitting', 'smile', 'saturday_night_fever'):
            self.assertFalse(self.is_char(tag, names=self.names), tag)

    def test_name_list_switch(self):
        names_for = self.pose_names_for
        names = frozenset({'hatsune_miku'})
        self.assertEqual(names_for('gelbooru', True, False, names), names)
        self.assertEqual(names_for('safebooru', True, False, names), frozenset())
        self.assertEqual(names_for('safebooru', True, True, names), names)
        self.assertEqual(names_for('gelbooru', False, True, names), frozenset())

    def test_missing_name_list_is_empty(self):
        self.assertEqual(self.load_names(str(NAMES_PATH) + '.missing'), frozenset())

    def test_clothing_only_keeps_traits(self):
        self.assertFalse(self.is_char('long_hair', strip_traits=False, strip_clothing=True))

    def test_filter_preserves_order_and_disabled_is_noop(self):
        tags = ['1girl', ' long_hair', ' sitting', ' blue_eyes', ' white_shirt', ' smile']
        self.assertEqual(self.filter(tags), ['1girl', ' sitting', ' white_shirt', ' smile'])
        self.assertEqual(self.filter(tags, True, True), ['1girl', ' sitting', ' smile'])
        self.assertEqual(self.filter(tags, False, False), tags)

    def test_category_tags_from_danbooru_and_e621(self):
        danbooru = {
            'tag_string_artist': 'some_artist',
            'tag_string_character': 'hatsune_miku',
            'tag_string_copyright': 'vocaloid',
            'tag_string_general': '1girl sitting',
        }
        self.assertEqual(self.categories(danbooru),
                         {'some_artist', 'hatsune_miku', 'vocaloid'})
        e621 = {'ranbooru_tag_categories': {
            'artist': ['artist_a'], 'character': [], 'species': ['canine'],
        }}
        self.assertEqual(self.categories(e621), {'artist_a', 'canine'})
        self.assertEqual(self.categories({'tags': 'a b'}), set())


if __name__ == '__main__':
    unittest.main()


class TestOnlyPosesWiring(unittest.TestCase):
    """UI controls are passed positionally; a misplaced entry shifts every argument."""

    def test_controls_match_hook_signatures(self):
        parsed = ast.parse(MODULE_PATH.read_text(encoding='utf-8'))
        script = next(n for n in parsed.body
                      if isinstance(n, ast.ClassDef) and n.name == 'Script')
        methods = {n.name: n for n in script.body if isinstance(n, ast.FunctionDef)}
        controls = next(
            [elt.id for elt in node.value.elts]
            for node in ast.walk(methods['ui'])
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'controls' for t in node.targets)
        )
        self.assertIn('only_poses', controls)
        self.assertIn('strip_clothing', controls)
        before = [a.arg for a in methods['before_process'].args.args][2:]
        post = [a.arg for a in methods['postprocess'].args.args][3:]
        self.assertEqual(before, controls)
        self.assertEqual(post, controls)
