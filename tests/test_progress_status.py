"""Forge-independent tests of Ranbooru's temporary progress lifecycle."""
import ast
import inspect
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
import unittest


class TestProgressStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module_path = Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py'
        parsed = ast.parse(module_path.read_text(encoding='utf-8'))
        names = {
            'show_ranbooru_progress',
            'clear_ranbooru_progress',
            'clear_ranbooru_progress_on_error',
            'clear_ranbooru_progress_after_postprocess',
        }
        nodes = [
            node for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        if {node.name for node in nodes} != names:
            raise AssertionError('Missing progress lifecycle helpers')
        cls.namespace = {
            'shared': SimpleNamespace(state=SimpleNamespace(textinfo=None)),
            'wraps': wraps,
        }
        isolated = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(isolated, str(module_path), 'exec'), cls.namespace)
        cls.parsed = parsed

    def setUp(self):
        self.state = SimpleNamespace(textinfo=None)
        self.namespace['shared'].state = self.state
        self.show = self.namespace['show_ranbooru_progress']
        self.clear = self.namespace['clear_ranbooru_progress']
        self.preprocess_wrapper = self.namespace['clear_ranbooru_progress_on_error']
        self.postprocess_wrapper = self.namespace['clear_ranbooru_progress_after_postprocess']

    def test_visible_during_search(self):
        self.show('Processing 6003 results')
        self.assertEqual(self.state.textinfo, 'Ranbooru: Processing 6003 results')

    def test_retained_during_sampling(self):
        def before_process():
            self.show('Processing 6003 results')
        self.preprocess_wrapper(before_process)()
        self.assertEqual(self.state.textinfo, 'Ranbooru: Processing 6003 results')

    def test_cleared_on_preprocessing_failure(self):
        def before_process():
            self.show('Searching gelbooru...')
            raise RuntimeError('API unavailable')
        with self.assertRaisesRegex(RuntimeError, 'API unavailable'):
            self.preprocess_wrapper(before_process)()
        self.assertIsNone(self.state.textinfo)

    def test_cleared_after_completed_generation(self):
        self.show('Processing 6003 results')
        self.assertEqual(self.postprocess_wrapper(lambda: 21)(), 21)
        self.assertIsNone(self.state.textinfo)

    def test_cleared_after_postprocess_failure(self):
        self.show('Processing 6003 results')
        def broken():
            raise RuntimeError('Processing failed')
        with self.assertRaisesRegex(RuntimeError, 'Processing failed'):
            self.postprocess_wrapper(broken)()
        self.assertIsNone(self.state.textinfo)

    def test_does_not_erase_another_extension(self):
        self.show('Processing 6003 results')
        self.state.textinfo = 'ControlNet: loading models'
        self.postprocess_wrapper(lambda: None)()
        self.assertEqual(self.state.textinfo, 'ControlNet: loading models')

    def test_idempotent(self):
        self.clear()
        self.clear()
        self.assertIsNone(self.state.textinfo)

    def test_decorators_preserve_signature(self):
        def before_process(foo, bar=4):
            return foo + bar
        self.assertEqual(inspect.signature(self.preprocess_wrapper(before_process)), inspect.signature(before_process))
        self.assertEqual(inspect.signature(self.postprocess_wrapper(before_process)), inspect.signature(before_process))

    def test_no_shared_state_is_safe(self):
        self.namespace['shared'].state = None
        self.show('Searching danbooru...')
        self.clear()

    def test_both_hooks_have_correct_decorators(self):
        methods = {
            node.name: node for cls in self.parsed.body if isinstance(cls, ast.ClassDef)
            for node in cls.body if isinstance(node, ast.FunctionDef)
            and node.name in ('before_process', 'postprocess')
        }
        self.assertEqual(set(methods), {'before_process', 'postprocess'})
        for name, decorator in (
            ('before_process', 'clear_ranbooru_progress_on_error'),
            ('postprocess', 'clear_ranbooru_progress_after_postprocess'),
        ):
            self.assertTrue(any(
                isinstance(item, ast.Name) and item.id == decorator
                for item in methods[name].decorator_list
            ))

if __name__ == '__main__':
    unittest.main()
