"""Test temporary progress cleanup without importing Forge Neo dependencies.

Run with: python -m unittest discover -s tests
"""
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
            'clear_ranbooru_progress_after_preprocessing',
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

    def setUp(self):
        self.state = SimpleNamespace(textinfo=None)
        self.namespace['shared'].state = self.state
        self.show = self.namespace['show_ranbooru_progress']
        self.clear = self.namespace['clear_ranbooru_progress']
        self.decorate = self.namespace['clear_ranbooru_progress_after_preprocessing']

    def test_visible_during_search(self):
        self.show('Processing 6003 results')
        self.assertEqual(self.state.textinfo, 'Ranbooru: Processing 6003 results')

    def test_cleared_before_sampling(self):
        self.show('Processing 6003 results')
        self.clear()
        self.assertIsNone(self.state.textinfo)

    def test_does_not_erase_another_extension(self):
        self.show('Searching danbooru...')
        self.state.textinfo = 'ControlNet: loading models'
        self.clear()
        self.assertEqual(self.state.textinfo, 'ControlNet: loading models')

    def test_idempotent(self):
        self.clear()
        self.clear()
        self.assertIsNone(self.state.textinfo)

    def test_cleared_after_success_and_preserves_signature(self):
        def before_process(number):
            self.show('Processing 6003 results')
            self.assertIsNotNone(self.state.textinfo)
            return number * 2

        wrapped = self.decorate(before_process)
        self.assertEqual(wrapped(21), 42)
        self.assertIsNone(self.state.textinfo)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(before_process))

    def test_cleared_after_failure(self):
        def before_process():
            self.show('Searching gelbooru...')
            raise RuntimeError('API unavailable')

        with self.assertRaisesRegex(RuntimeError, 'API unavailable'):
            self.decorate(before_process)()
        self.assertIsNone(self.state.textinfo)

    def test_no_shared_state_is_safe(self):
        self.namespace['shared'].state = None
        self.show('Searching danbooru...')
        self.clear()

    def test_decorator_applied_to_real_hook(self):
        module_path = Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py'
        parsed = ast.parse(module_path.read_text(encoding='utf-8'))
        methods = [
            node for cls in parsed.body if isinstance(cls, ast.ClassDef)
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == 'before_process'
        ]
        self.assertEqual(len(methods), 1)
        self.assertTrue(any(
            isinstance(decorator, ast.Name)
            and decorator.id == 'clear_ranbooru_progress_after_preprocessing'
            for decorator in methods[0].decorator_list
        ))


if __name__ == '__main__':
    unittest.main()
