"""Forge-independent tests of the XBooru, Safebooru, Rule34, Konachan,
Yande.re and AIBooru adapters, plus per-booru credential handling.

    python -m unittest discover -s tests -v
"""
import ast
import io
import re
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import quote_plus


class Response:
    def __init__(self, data=None, status=200, text=None, invalid_json=False):
        self.data = data
        self.status_code = status
        self.invalid_json = invalid_json
        if text is None:
            text = '' if data is None else repr(data)
        self.text = text

    def json(self):
        if self.invalid_json:
            raise ValueError('Invalid JSON')
        return self.data


class RequestsStub:
    exceptions = types.SimpleNamespace(RequestException=ConnectionError)
    RequestException = ConnectionError

    def __init__(self):
        self.responses = []
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError(f'Unexpected request: {url}')
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def count_xml(count):
    return Response(text=f'<?xml version="1.0"?><posts count="{count}" offset="0"></posts>')


def dapi_post(post_id=1, image='a.jpg', directory='12', tags='1girl solo', file_url=None):
    post = {'id': post_id, 'image': image, 'directory': directory,
            'tags': tags, 'score': None}
    if file_url is not None:
        post['file_url'] = file_url
    return post


def moe_post(post_id=1, file_url='https://files.example/a.jpg'):
    return {'id': post_id, 'tags': 'thighhighs solo', 'score': 3, 'file_url': file_url}


NAMES = {
    'Booru', 'GelbooruBeta', 'XBooru', 'Safebooru', 'Rule34', 'Moebooru',
    'Konachan', 'Yandere', 'AIBooru', 'redact_secrets', 'fetch_booru_json',
    'fetch_xml_count', 'pages_for_count', 'parse_post_id',
    'resolve_booru_credentials', 'check_exception',
}
CONSTANTS = {'POSTS_COUNT_PATTERN', 'XML_REASON_PATTERN', 'CREDENTIAL_BOORUS'}


class AdapterTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        body = []
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in NAMES:
                body.append(node)
            elif isinstance(node, ast.Assign) and any(
                    getattr(t, 'id', None) in CONSTANTS for t in node.targets):
                body.append(node)
        found = {getattr(n, 'name', None) for n in body} | {
            t.id for n in body if isinstance(n, ast.Assign) for t in n.targets}
        missing = (NAMES | CONSTANTS) - found
        if missing:
            raise AssertionError(f'Unable to isolate {sorted(missing)}')
        cls.module = compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(source), 'exec')

    def setUp(self):
        self.requests = RequestsStub()
        self.ranges = []

        def randint(low, high):
            self.ranges.append((low, high))
            return high

        self.ns = {'requests': self.requests, 'booru_http': self.requests, 're': re, 'quote_plus': quote_plus,
                   'POST_AMOUNT': 100, 'COUNT': 0,
                   'random': types.SimpleNamespace(randint=randint)}
        exec(self.module, self.ns)

    def quiet(self, fn, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            result = fn(*args)
        return result, out.getvalue()


class FetchHelperTests(AdapterTestBase):
    def fetch(self, response, secrets=()):
        self.requests.responses.append(response)
        return self.ns['fetch_booru_json']('https://x.test/api', 'xbooru', {}, secrets=secrets)

    def test_empty_body_is_empty_result(self):
        self.assertEqual(self.fetch(Response(text='')), [])

    def test_every_request_has_a_timeout(self):
        self.fetch(Response(data=[], text='[]'))
        self.assertEqual(self.requests.calls[0][1]['timeout'], 20)

    def test_xml_error_reason_is_reported(self):
        body = '<?xml version="1.0"?><response success="false" reason="Search error: API limited due to abuse."/>'
        with self.assertRaisesRegex(RuntimeError, 'xbooru API error: Search error: API limited due to abuse'):
            self.fetch(Response(text=body))

    def test_missing_authentication_names_the_fix(self):
        with self.assertRaisesRegex(RuntimeError, 'requires an API key and user ID'):
            self.fetch(Response(data='Missing authentication. Go to api.rule34.xxx',
                                text='"Missing authentication. Go to api.rule34.xxx"'))

    def test_rejected_credentials(self):
        with self.assertRaisesRegex(RuntimeError, 'rejected the request \\(HTTP 401\\)'):
            self.fetch(Response(status=401, text='Unauthorized'))

    def test_rate_limit(self):
        with self.assertRaisesRegex(RuntimeError, 'HTTP 429.*Rate limit'):
            self.fetch(Response(status=429, text=''))

    def test_html_page_is_an_error_not_a_crash(self):
        with self.assertRaisesRegex(RuntimeError, 'xbooru API error'):
            self.fetch(Response(text='<html><body>Cloudflare</body></html>'))

    def test_invalid_json(self):
        with self.assertRaisesRegex(RuntimeError, 'non-JSON response'):
            self.fetch(Response(text='{broken', invalid_json=True))

    def test_connection_error_is_redacted(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.fetch(ConnectionError('failed for https://x.test/?api_key=SECRET'), secrets=('SECRET',))
        self.assertIn('Unable to contact xbooru', str(ctx.exception))
        self.assertNotIn('SECRET', str(ctx.exception))

    def test_xml_count_parses_and_fails_soft(self):
        fetch_count = self.ns['fetch_xml_count']
        self.requests.responses = [count_xml(321), Response(text='nope'), ConnectionError('down')]
        self.assertEqual(fetch_count('u'), 321)
        self.assertIsNone(fetch_count('u'))
        self.assertIsNone(fetch_count('u'))

    def test_pages_for_count(self):
        pages = self.ns['pages_for_count']
        self.assertEqual(pages(None, 10), 10)
        self.assertEqual(pages(250, 10), 3)
        self.assertEqual(pages(5, 10), 1)
        self.assertEqual(pages(10 ** 9, 5000, 2001), 2001)
        self.assertEqual(pages(0, 10), 1)


class GelbooruBetaTests(AdapterTestBase):
    def test_max_pages_is_honoured_within_result_count(self):
        # The old loop shrank Max Pages to 1-2; now it is only capped by results.
        self.requests.responses = [count_xml(380_000), Response(data=[dapi_post()], text='[..]')]
        data, _ = self.quiet(self.ns['XBooru']().get_data, '&tags=1girl', 50)
        self.assertEqual(self.ranges, [(0, 49)])
        self.assertEqual(len(self.requests.calls), 2)
        self.assertIn('limit=0', self.requests.calls[0][0])
        self.assertIn('&pid=49&tags=1girl', self.requests.calls[1][0])
        self.assertEqual(self.ns['COUNT'], 380_000)
        self.assertEqual(len(data['post']), 1)

    def test_pages_capped_by_result_count_and_dapi_limit(self):
        self.requests.responses = [count_xml(250), Response(data=[dapi_post()], text='[..]')]
        self.quiet(self.ns['Safebooru']().get_data, '', 50)
        self.requests.responses = [Response(text=''), Response(data=[dapi_post()], text='[..]')]
        self.quiet(self.ns['Safebooru']().get_data, '', 5000)
        self.assertEqual(self.ranges, [(0, 2), (0, 2000)])

    def test_zero_count_skips_the_post_request(self):
        self.requests.responses = [count_xml(0)]
        data, _ = self.quiet(self.ns['XBooru']().get_data, '&tags=nothing', 10)
        self.assertEqual(data, {'post': []})
        self.assertEqual(len(self.requests.calls), 1)

    def test_empty_body_page_is_empty_list(self):
        self.requests.responses = [Response(text='nope'), Response(text='')]
        data, _ = self.quiet(self.ns['Safebooru']().get_data, '', 3)
        self.assertEqual(data, {'post': []})

    def test_posts_are_normalised(self):
        posts = [
            dapi_post(1),
            dapi_post(2, file_url='https://img.xbooru.com/images/12/b.jpg'),
            dapi_post(3, image=None, directory=None, tags=''),  # missing-ID phantom
            'junk',
        ]
        self.requests.responses = [count_xml(4), Response(data=posts, text='[..]')]
        data, _ = self.quiet(self.ns['XBooru']().get_data, '', 1)
        self.assertEqual([p['id'] for p in data['post']], [1, 2])
        self.assertEqual(data['post'][0]['file_url'], 'https://xbooru.com/images/12/a.jpg')
        self.assertEqual(data['post'][1]['file_url'], 'https://img.xbooru.com/images/12/b.jpg')
        self.assertEqual(data['post'][0]['score'], 0)

    def test_rule34_shapes_file_url_only_and_null_body(self):
        rows = [{'id': 5, 'file_url': 'https://api-cdn.rule34.xxx/images/1/x.jpg', 'tags': 'solo'},
                dapi_post(6, image=None, directory=None, tags='',
                          file_url='https://safebooru.org/images//')]
        self.requests.responses = [count_xml(2), Response(data=rows, text='[..]'),
                                   count_xml(2), Response(data=None, text='null')]
        data, _ = self.quiet(self.ns['Rule34'](username='1', api_key='k').get_data, '', 1)
        self.assertEqual([p['id'] for p in data['post']], [5])
        data, _ = self.quiet(self.ns['Rule34'](username='1', api_key='k').get_data, '', 1)
        self.assertEqual(data, {'post': []})

    def test_post_id(self):
        self.requests.responses = [Response(data=[dapi_post(1000)], text='[..]')]
        data, _ = self.quiet(self.ns['Safebooru']().get_post, '&tags=-animated', 10, ' 1000 ')
        self.assertEqual(data['post'][0]['id'], 1000)
        url = self.requests.calls[0][0]
        self.assertIn('&id=1000', url)
        self.assertNotIn('tags=', url)
        self.assertNotIn('pid=', url)

    def test_post_id_not_found_and_non_numeric(self):
        phantom = dapi_post(999999999, image=None, directory=None, tags='')
        self.requests.responses = [Response(data=[phantom], text='[..]')]
        with self.assertRaisesRegex(ValueError, 'post 999999999 was not found'):
            self.quiet(self.ns['Safebooru']().get_post, '', 10, '999999999')
        with self.assertRaisesRegex(ValueError, 'post IDs are numbers'):
            self.ns['XBooru']().get_post('', 10, 'abc')
        self.assertEqual(len(self.requests.calls), 1)


class Rule34Tests(AdapterTestBase):
    def test_credentials_sent_but_never_logged(self):
        self.requests.responses = [count_xml(500), Response(data=[dapi_post()], text='[..]')]
        booru = self.ns['Rule34'](username='12345', api_key='S3CRET+KEY')
        _, output = self.quiet(booru.get_data, '&tags=solo', 2)
        for url, _ in self.requests.calls:
            self.assertIn('api_key=S3CRET%2BKEY', url)
            self.assertIn('user_id=12345', url)
        self.assertNotIn('S3CRET', output)
        self.assertNotIn('S3CRET', booru.last_url)

    def test_no_credentials_raises_readable_error(self):
        missing = Response(data='Missing authentication. Go to api.rule34.xxx for more information',
                           text='"Missing authentication. Go to api.rule34.xxx for more information"')
        self.requests.responses = [Response(text='<error>Missing authentication</error>'), missing]
        with self.assertRaisesRegex(RuntimeError, 'rule34 requires an API key and user ID'):
            self.quiet(self.ns['Rule34']().get_data, '', 2)
        self.assertNotIn('api_key', self.requests.calls[1][0])


class MoebooruTests(AdapterTestBase):
    def test_pages_are_one_based_and_capped_by_count(self):
        for cls in ('Konachan', 'Yandere'):
            self.requests.responses = [count_xml(250), Response(data=[moe_post()], text='[..]')]
            data, _ = self.quiet(self.ns[cls]().get_data, '&tags=thighhighs', 50)
            self.assertEqual(len(data['post']), 1)
        self.assertEqual(self.ranges, [(1, 3), (1, 3)])
        self.assertIn('post.xml?limit=1&tags=thighhighs', self.requests.calls[0][0])
        self.assertIn('&page=3&tags=thighhighs', self.requests.calls[1][0])

    def test_skips_posts_without_image(self):
        self.requests.responses = [count_xml(3), Response(data=[moe_post(1), moe_post(2, None)], text='[..]')]
        data, _ = self.quiet(self.ns['Yandere']().get_data, '', 1)
        self.assertEqual([p['id'] for p in data['post']], [1])

    def test_post_id(self):
        self.requests.responses = [Response(data=[moe_post(1000)], text='[..]')]
        data, _ = self.quiet(self.ns['Konachan']().get_post, '&tags=-animated', 10, '1000')
        self.assertEqual(data['post'][0]['id'], 1000)
        self.assertTrue(self.requests.calls[0][0].endswith('&tags=id:1000'))

    def test_post_id_not_found(self):
        self.requests.responses = [Response(data=[], text='[]')]
        with self.assertRaisesRegex(ValueError, 'yande.re post 999 was not found'):
            self.quiet(self.ns['Yandere']().get_post, '', 10, '999')

    def test_check_exception_allows_moebooru_post_ids(self):
        for booru in ('konachan', 'yande.re'):
            self.ns['check_exception'](booru, {'post_id': '1000', 'tags': ''})


class AIBooruTests(AdapterTestBase):
    def test_count_caps_pages_and_posts_are_normalised(self):
        posts = [{'id': 1, 'tag_string': '1girl solo', 'score': None, 'file_url': 'https://cdn/a.png'},
                 {'id': 2, 'tag_string': 'hidden'}]
        self.requests.responses = [Response(data={'counts': {'posts': 10 ** 7}}, text='{..}'),
                                   Response(data=posts, text='[..]')]
        data, _ = self.quiet(self.ns['AIBooru']().get_data, '&tags=1girl', 5000)
        self.assertEqual(self.ranges, [(1, 1000)])
        self.assertEqual(data['post'], [{'id': 1, 'tag_string': '1girl solo', 'score': 0,
                                         'file_url': 'https://cdn/a.png', 'tags': '1girl solo'}])
        self.assertIn('counts/posts.json?tags=1girl', self.requests.calls[0][0])

    def test_error_page_is_readable(self):
        self.requests.responses = [ConnectionError('down'), Response(status=500, text='<html>oops</html>')]
        with self.assertRaisesRegex(RuntimeError, 'AIBooru API error \\(HTTP 500\\)'):
            self.quiet(self.ns['AIBooru']().get_data, '', 3)


class CredentialManagerStub:
    def __init__(self, saved=None):
        self.saved = saved or {}
        self.writes = []

    def get_booru_credentials(self, booru):
        return dict(self.saved.get(booru, {}))

    def save_booru_credentials(self, booru, api_key, user_id=None):
        self.writes.append((booru, api_key, user_id))


class CredentialResolutionTests(AdapterTestBase):
    def resolve(self, *args):
        return self.ns['resolve_booru_credentials'](*args)

    def test_saved_credentials_of_selected_booru_win(self):
        manager = CredentialManagerStub({'rule34': {'api_key': 'r34key', 'user_id': '7'}})
        self.assertEqual(self.resolve('rule34', 'gelkey', 'gelid', True, manager), ('r34key', '7'))
        self.assertEqual(manager.writes, [])

    def test_saved_key_never_crosses_to_another_booru(self):
        manager = CredentialManagerStub({'gelbooru': {'api_key': 'gelkey', 'user_id': '1'}})
        for booru in ('rule34', 'e621'):
            self.assertEqual(self.resolve(booru, '', '', False, manager), (None, None))
        # Even if the page still held Gelbooru's key, Gelbooru uses its saved one.
        self.assertEqual(self.resolve('gelbooru', 'other', '2', True, manager), ('gelkey', '1'))
        self.assertEqual(manager.writes, [])

    def test_typed_credentials_used_and_saved_on_request(self):
        manager = CredentialManagerStub()
        self.assertEqual(self.resolve('e621', ' key ', ' fox ', False, manager), ('key', 'fox'))
        self.assertEqual(manager.writes, [])
        self.assertEqual(self.resolve('rule34', 'key', '7', True, manager), ('key', '7'))
        self.assertEqual(manager.writes, [('rule34', 'key', '7')])

    def test_incomplete_or_unsupported(self):
        manager = CredentialManagerStub()
        self.assertEqual(self.resolve('gelbooru', 'key', '', True, manager), (None, None))
        self.assertEqual(self.resolve('safebooru', 'key', '1', True, manager), (None, None))
        self.assertEqual(manager.writes, [])

    def test_credential_boorus(self):
        self.assertEqual(set(self.ns['CREDENTIAL_BOORUS']), {'gelbooru', 'rule34', 'e621'})


class CredentialPanelTests(unittest.TestCase):
    """booru_credential_updates, run with stub Gradio and credentials."""

    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        script = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Script')
        methods = [n for n in script.body if isinstance(n, ast.FunctionDef) and n.name in (
            'booru_credential_updates', 'load_booru_credentials', 'clear_booru_credentials')]
        spec = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], 'id', None) == 'CREDENTIAL_BOORUS')
        panel = ast.ClassDef(name='Panel', bases=[], keywords=[], body=methods, decorator_list=[])
        cls.module = compile(ast.fix_missing_locations(
            ast.Module(body=[spec, panel], type_ignores=[])), str(source), 'exec')

    def setUp(self):
        manager = CredentialManagerStub({'gelbooru': {'api_key': 'GELKEY', 'user_id': '42'}})
        manager.credentials_file = '/ext/user/credentials/credentials.json'
        manager.cleared = []
        manager.clear_booru_credentials = manager.cleared.append
        self.manager = manager
        self.ns = {'gr': types.SimpleNamespace(update=lambda **kw: kw),
                   'credentials_manager': manager}
        exec(self.module, self.ns)
        self.panel = self.ns['Panel']()

    def test_saved_key_is_never_put_in_the_page(self):
        group, heading, key, user, status, clear = self.panel.load_booru_credentials('gelbooru')
        self.assertEqual((key['value'], user['value']), ('', ''))
        self.assertFalse(key['visible'])
        self.assertTrue(status['visible'] and clear['visible'])
        self.assertIn('credentials.json', status['value'])

    def test_fields_follow_the_selected_booru(self):
        group, heading, key, user, status, clear = self.panel.load_booru_credentials('e621')
        self.assertTrue(group['visible'] and key['visible'])
        self.assertEqual(user['label'], 'Username')
        self.assertIn('e621', heading['value'])
        self.assertFalse(status['visible'])
        group, *_ = self.panel.load_booru_credentials('safebooru')
        self.assertFalse(group['visible'])

    def test_clear_only_clears_selected_booru(self):
        _, _, key, _, status, _ = self.panel.clear_booru_credentials('rule34')
        self.assertEqual(self.manager.cleared, ['rule34'])
        self.assertTrue(key['visible'])
        self.panel.clear_booru_credentials('safebooru')
        self.assertEqual(self.manager.cleared, ['rule34'])


class CredentialWiringTests(unittest.TestCase):
    """The credentials panel must follow the booru however it changes."""

    def test_browser_restore_updates_the_credentials_panel(self):
        source = (Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py').read_text(encoding='utf-8')
        self.assertNotIn('gelbooru_credentials_group', source)
        self.assertIn('load_outputs.extend(credential_outputs)', source)
        self.assertIn('results.extend(self.booru_credential_updates(current_booru))', source)

    def test_loader_uses_change_and_selected_booru(self):
        source = (Path(__file__).resolve().parents[1] / 'scripts' / 'ranbooru.py').read_text(encoding='utf-8')
        self.assertIn('booru.change(\n                fn=self.load_booru_credentials', source)
        self.assertIn('fn=self.clear_booru_credentials,\n                inputs=[booru]', source)
        self.assertNotIn('load_gelbooru_credentials', source)
        self.assertIn('resolve_booru_credentials(\n                booru, api_key, user_id', source)


if __name__ == '__main__':
    unittest.main()
