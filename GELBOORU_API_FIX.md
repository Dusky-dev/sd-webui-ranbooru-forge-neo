# Gelbooru API response handling

Ranbooru's Gelbooru adapter now checks the HTTP status and validates the JSON response **before** reading `@attributes.count` or the `post` list.

## Behavior

- **Successful JSON:** accepts the normal `@attributes.count` and `post` fields, including an empty `post` list. Accepts count as an integer or numeric string.
- **Empty/malformed JSON:** retries at most two times (three requests total), then reports a descriptive error.
- **HTTP 429 or common 5xx errors:** retries with short 1- and 2-second backoffs; persistent failures are reported with their HTTP status.
- **Connection failures:** two retries before a readable error, without printing a raw request exception.
- **HTML login/challenge pages:** reports that the API returned HTML instead of JSON, without blindly retrying.
- **HTTP 401/403:** immediately explains how to check the Gelbooru API key and user ID.
- **Explicit JSON failures** (e.g., `success: false`): show a meaningful error instead of being interpreted as an empty list of posts.
- **Page selection:** if the randomly selected page exceeds the real number of pages or comes back empty, retry a valid page.

The adapter deliberately does **not** print API responses or authenticated URLs, because the API key may be in a request query string. `last_url` is now a credential-free URL. Credentials are still sent to Gelbooru through request parameters when configured.

The response is only parsed when the server supplies JSON; no website scraping or HTML fallback is attempted. The normal result-selection, prompt processing, and local state-persistence flows remain unchanged.

## Testing

Run:

```sh
python -m unittest tests.test_gelbooru_api -v
```

The tests use mocked HTTP responses and do not make live calls to Gelbooru. Live API behavior still depends on site access, throttling and whether account credentials are valid.

See [USER_DATA.md](USER_DATA.md) for how to keep your credentials local and safe from `git pull`.
