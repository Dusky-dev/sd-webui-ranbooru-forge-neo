"""Regression tests for Gelbooru HTTP/JSON responses without Forge dependencies.

    python -m unittest discover -s tests -v
"""
import ast
import random
import types
import unittest
from pathlib import Path


class Response:
    def __init__(self, data=None, status=200, text="", content_type="application/json", invalid_json=False):
        self.data = data
        self.status_code = status
        self.text = text
        self.headers = {"Content-Type": content_type}
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError("Invalid JSON")
        return self.data


class RequestsStub:
    exceptions = types.SimpleNamespace(RequestException=ConnectionError)

    def __init__(self):
        self.responses = []
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected request; retry count exceeded")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def posts_result(count=10, posts=None):
    return Response(data={
        "@attributes": {"count": count},
        "post": [{"id": 1, "tags": "example", "file_url": "https://example.com/a.jpg"}]
        if posts is None else posts,
    })


class GelbooruResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_file = Path(__file__).resolve().parents[1] / "scripts" / "ranbooru.py"
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in ("Booru", "Gelbooru")
        ]
        if len(classes) != 2:
            raise AssertionError("Unable to isolate both Booru and Gelbooru classes")
        cls.isolated_module = compile(
            ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])),
            str(source_file), "exec",
        )

    def setUp(self):
        self.requests = RequestsStub()
        self.sleeps = []
        self.namespace = {
            "requests": self.requests,
            "POST_AMOUNT": 100,
            "COUNT": 100,
            "random": types.SimpleNamespace(randrange=lambda stop: 0),
            "time": types.SimpleNamespace(sleep=self.sleeps.append),
        }
        exec(self.isolated_module, self.namespace)
        self.adapter = self.namespace["Gelbooru"](False)

    def test_valid_response(self):
        self.requests.responses = [posts_result(6003)]
        result = self.adapter.get_data("&tags=example", 3)
        self.assertEqual(result["post"][0]["id"], 1)
        self.assertEqual(self.namespace["COUNT"], 6003)
        self.assertEqual(len(self.requests.calls), 1)

    def test_string_count_and_missing_post_list(self):
        self.requests.responses = [Response(data={"@attributes": {"count": "0"}})]
        result = self.adapter.get_data("&tags=none", 1)
        self.assertEqual(result["post"], [])
        self.assertEqual(self.namespace["COUNT"], 0)

    def test_empty_list_response(self):
        self.requests.responses = [Response(data=[])]
        self.assertEqual(self.adapter.get_data("", 1)["post"], [])

    def test_html_error_page_is_not_retried(self):
        self.requests.responses = [
            Response(status=200, text="<html>Login required</html>", content_type="text/html", invalid_json=True)
        ]
        with self.assertRaisesRegex(RuntimeError, "HTML page instead of JSON"):
            self.adapter.get_data("", 1)
        self.assertEqual(len(self.requests.calls), 1)

    def test_401_auth_error_and_no_credentials_in_exception(self):
        self.adapter = self.namespace["Gelbooru"](False, "super-secret-key", "12345")
        self.requests.responses = [Response(status=401, text="API key super-secret-key")]
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            self.adapter.get_data("", 1)
        self.assertNotIn("super-secret-key", str(caught.exception))
        self.assertNotIn("super-secret-key", self.adapter.last_url)
        url, kwargs = self.requests.calls[0]
        self.assertNotIn("super-secret-key", url)
        self.assertEqual(kwargs["params"], {"api_key": "super-secret-key", "user_id": "12345"})
        self.assertEqual(len(self.requests.calls), 1)

    def test_403_denied_without_retry(self):
        self.requests.responses = [Response(status=403)]
        with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            self.adapter.get_data("", 1)
        self.assertEqual(len(self.requests.calls), 1)

    def test_429_retries_then_recovers(self):
        self.requests.responses = [Response(status=429), posts_result()]
        self.assertEqual(len(self.adapter.get_data("", 1)["post"]), 1)
        self.assertEqual(len(self.requests.calls), 2)
        self.assertEqual(self.sleeps, [1])

    def test_repeated_rate_limits_eventually_fail(self):
        self.requests.responses = [Response(status=429) for _ in range(3)]
        with self.assertRaisesRegex(RuntimeError, "HTTP 429.*3 attempts"):
            self.adapter.get_data("", 1)
        self.assertEqual(len(self.requests.calls), 3)
        self.assertEqual(self.sleeps, [1, 2])

    def test_503_retries(self):
        self.requests.responses = [Response(status=503), posts_result()]
        self.adapter.get_data("", 1)
        self.assertEqual(len(self.requests.calls), 2)

    def test_non_json_empty_response_retries(self):
        self.requests.responses = [Response(invalid_json=True), posts_result()]
        self.adapter.get_data("", 1)
        self.assertEqual(len(self.requests.calls), 2)

    def test_failed_json_after_three_attempts(self):
        self.requests.responses = [Response(invalid_json=True) for _ in range(3)]
        with self.assertRaisesRegex(RuntimeError, "empty or invalid JSON after 3 attempts"):
            self.adapter.get_data("", 1)
        self.assertEqual(len(self.requests.calls), 3)

    def test_network_failure_retries(self):
        self.requests.responses = [ConnectionError("secret query params"), posts_result()]
        self.adapter.get_data("", 1)
        self.assertEqual(len(self.requests.calls), 2)

    def test_network_failure_has_no_secret_in_error(self):
        self.requests.responses = [ConnectionError("api_key=SECRET") for _ in range(3)]
        with self.assertRaises(RuntimeError) as caught:
            self.adapter.get_data("", 1)
        self.assertNotIn("SECRET", str(caught.exception))

    def test_malformed_json_shape_produces_clear_error(self):
        self.requests.responses = [Response(data={"post": []})]
        with self.assertRaisesRegex(RuntimeError, "post count"):
            self.adapter.get_data("", 1)

    def test_failed_search_object_produces_clear_error(self):
        self.requests.responses = [Response(data={"success": False, "message": "search down"})]
        with self.assertRaisesRegex(RuntimeError, "failed API search"):
            self.adapter.get_data("", 1)

    def test_malformed_post_entries_are_skipped(self):
        self.requests.responses = [
            posts_result(posts=[{"id": 1}, "not a post", None])
        ]
        result = self.adapter.get_data("", 1)
        self.assertEqual(result["post"], [{"id": 1}])

    def test_invalid_page_falls_back_to_valid_page(self):
        pages = iter([9, 1])
        self.namespace["random"].randrange = lambda count: next(pages)
        self.requests.responses = [posts_result(count=201, posts=[]), posts_result(count=201)]
        result = self.adapter.get_data("", 10)
        self.assertEqual(len(result["post"]), 1)
        self.assertIn("&pid=9", self.requests.calls[0][0])
        self.assertIn("&pid=1", self.requests.calls[1][0])

    def test_post_id_search_is_single_request(self):
        self.requests.responses = [posts_result(count=1)]
        self.adapter.get_post("", 50, "12345")
        self.assertEqual(len(self.requests.calls), 1)
        self.assertIn("&id=12345", self.requests.calls[0][0])

    def test_fringe_benefits_cookie(self):
        self.adapter = self.namespace["Gelbooru"](True)
        self.requests.responses = [posts_result()]
        self.adapter.get_data("", 1)
        self.assertEqual(self.requests.calls[0][1]["cookies"], {"fringeBenefits": "yup"})

    def test_non_retryable_http_400(self):
        self.requests.responses = [Response(status=400)]
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            self.adapter.get_data("", 1)
        self.assertEqual(len(self.requests.calls), 1)


if __name__ == "__main__":
    unittest.main()
