"""Forge-independent tests of the e621 adapter against current and legacy API shapes."""
import ast
import types
import unittest
from pathlib import Path


class Response:
    def __init__(self, data=None, status=200, invalid_json=False):
        self.data = data
        self.status_code = status
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError('Invalid JSON')
        return self.data


class RequestsStub:
    RequestException = ConnectionError

    def __init__(self):
        self.responses = []
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def current_post(post_id=1, tags=('1girl', 'solo', 'sitting'), url='https://static1.e621.net/a.jpg',
                 score=5, deleted=False):
    return {
        'id': post_id,
        'tags': list(tags),
        'rating': 's',
        'stats': {'score': {'up': score, 'down': 0, 'total': score}},
        'files': {'original': {'url': url, 'width': 1, 'height': 1}},
        'flags': {'deleted': deleted},
    }


def legacy_post(post_id=2):
    return {
        'id': post_id,
        'tags': {'general': ['solo', 'sitting'], 'artist': ['some_artist'],
                 'copyright': ['some_series'], 'character': ['some_char'],
                 'species': ['wolf'], 'meta': ['hi_res']},
        'score': {'up': 3, 'down': -1, 'total': 2},
        'file': {'url': 'https://static1.e621.net/b.jpg'},
    }


class E621AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        classes = [n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name in ('Booru', 'e621')]
        if len(classes) != 2:
            raise AssertionError('Unable to isolate Booru and e621 classes')
        cls.module = compile(
            ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])),
            str(source), 'exec')

    def setUp(self):
        self.requests = RequestsStub()
        self.pages = []

        def randint(low, high):
            self.pages.append((low, high))
            return high

        self.ns = {'requests': self.requests, 'POST_AMOUNT': 100, 'COUNT': 0,
                   'random': types.SimpleNamespace(randint=randint)}
        exec(self.module, self.ns)
        self.adapter = self.ns['e621']()

    def test_current_bare_list(self):
        self.requests.responses = [Response([current_post()])]
        posts = self.adapter.get_data('&tags=-animated', 3)['post']
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]['tags'], '1girl solo sitting')
        self.assertEqual(posts[0]['score'], 5)
        self.assertEqual(posts[0]['file_url'], 'https://static1.e621.net/a.jpg')
        self.assertEqual(self.ns['COUNT'], 1)
        self.assertEqual(len(self.requests.calls), 1)
        url, kwargs = self.requests.calls[0]
        self.assertIn('&page=3&tags=-animated', url)
        self.assertEqual(kwargs['timeout'], 20)

    def test_legacy_wrapped_categorised_shape(self):
        self.requests.responses = [Response({'posts': [legacy_post()]})]
        post = self.adapter.get_data('', 1)['post'][0]
        self.assertEqual(post['tags'], 'solo sitting some_artist some_series some_char wolf')
        self.assertEqual(post['score'], 2)
        self.assertEqual(post['ranbooru_tag_categories']['species'], ['wolf'])

    def test_skips_deleted_missing_url_and_animated(self):
        self.requests.responses = [Response([
            current_post(1, url=None),
            current_post(2, deleted=True),
            current_post(3, tags=('animated', 'solo')),
            'not a post',
            current_post(4),
        ])]
        posts = self.adapter.get_data('', 1)['post']
        self.assertEqual([p['id'] for p in posts], [4])

    def test_pages_are_one_based_and_capped(self):
        self.requests.responses = [Response([])] * 3
        self.adapter.get_data('', 0)
        self.adapter.get_data('', 10)
        self.adapter.get_data('', 5000)
        self.assertEqual(self.pages, [(1, 1), (1, 10), (1, 750)])

    def test_error_object_raises_readable_error(self):
        self.requests.responses = [Response(
            {'success': False, 'message': 'You cannot go beyond page 750.'}, status=410)]
        with self.assertRaisesRegex(RuntimeError, 'HTTP 410.*page 750'):
            self.adapter.get_data('', 1)

    def test_rate_limit_and_non_json(self):
        self.requests.responses = [Response({'success': False, 'message': 'slow down'}, status=429)]
        with self.assertRaisesRegex(RuntimeError, 'Rate limit'):
            self.adapter.get_data('', 1)
        self.requests.responses = [Response(invalid_json=True)]
        with self.assertRaisesRegex(RuntimeError, 'non-JSON'):
            self.adapter.get_data('', 1)

    def test_network_error(self):
        self.requests.responses = [ConnectionError('down')]
        with self.assertRaisesRegex(RuntimeError, 'Unable to contact e621'):
            self.adapter.get_data('', 1)

    def test_name_tag_lookup_is_chunked_and_cached(self):
        self.ns['e621']._category_cache.clear()
        self.adapter.LOOKUP_CHUNK = 2
        self.requests.responses = [
            Response([{'name': 'some_artist', 'category': 1},
                      {'name': 'solo', 'category': 0}]),
            Response([{'name': 'wolf', 'category': 5},
                      {'name': "zeta_(o'neil)", 'category': 4}]),
        ]
        found = self.adapter.name_tags({'solo', 'some_artist', 'wolf', "zeta_(o'neil)"})
        self.assertEqual(found, {'some_artist', 'wolf', "zeta_(o'neil)"})
        self.assertEqual(len(self.requests.calls), 2)
        self.assertIn('search[name]=solo,some_artist', self.requests.calls[0][0])
        self.assertIn('zeta_%28o%27neil%29', self.requests.calls[1][0])
        # Cached: no further requests.
        self.assertEqual(self.adapter.name_tags({'wolf', 'solo'}), {'wolf'})
        self.assertEqual(len(self.requests.calls), 2)

    def test_tag_categories_mark_meta_lore_invalid(self):
        self.ns['e621']._category_cache.clear()
        self.requests.responses = [Response([
            {'name': 'hi_res', 'category': 7}, {'name': 'sibling_(lore)', 'category': 8},
            {'name': 'tough_guy', 'category': 6}, {'name': 'solo', 'category': 0},
        ])]
        cats = self.adapter.tag_categories({'hi_res', 'sibling_(lore)', 'tough_guy', 'solo', 'gone'})
        drop = {t for t, c in cats.items() if c in self.adapter.DROP_CATEGORIES}
        self.assertEqual(drop, {'hi_res', 'sibling_(lore)', 'tough_guy'})
        self.assertIsNone(cats['gone'])

    def test_name_tag_lookup_fails_soft(self):
        self.ns['e621']._category_cache.clear()
        self.requests.responses = [ConnectionError('down')]
        self.assertEqual(self.adapter.name_tags({'some_artist'}), set())
        self.requests.responses = [Response({'success': False}, status=429)]
        self.assertEqual(self.adapter.name_tags({'other'}), set())


if __name__ == '__main__':
    unittest.main()
