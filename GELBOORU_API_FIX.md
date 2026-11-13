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
- **Page selection:** query page 0 first for a reliable total count, then choose a random page only within both the user's **Max pages** setting and the actual available pages. An empty deeper page falls back to the already validated first page.
- **HTML response from a deeper page:** use the successfully fetched page 0 result rather than failing generation. No redundant third request is needed; this is gentler on Gelbooru's per-account request limit.
- **Safe diagnostics:** log the numeric page index, HTTP status, response type (JSON/HTML/other), and attempt number. No request URL, HTML body, user ID, or API key is logged. First-page HTML still raises an explicit API error because a valid result count cannot be obtained.

The adapter deliberately does **not** print API responses or authenticated URLs, because the API key may be in a request query string. `last_url` is now a credential-free URL. Credentials are still sent to Gelbooru through request parameters when configured.

The response is only parsed when the server supplies JSON; no website scraping or HTML fallback is attempted. The normal result-selection, prompt processing, and local state-persistence flows remain unchanged.

## Testing

Run:

```sh
python -m unittest tests.test_gelbooru_api -v
```

The tests use mocked HTTP responses and do not make live calls to Gelbooru. Live API behavior still depends on site access, throttling and whether account credentials are valid.

See [USER_DATA.md](USER_DATA.md) for how to keep your credentials local and safe from `git pull`.

## Why tag search may fail while post ID works

Gelbooru started requiring `api_key` and `user_id` for many API requests in June 2025. The website's own API documentation states that API authentication may be required. A post-ID lookup working **does not prove** that tag-search access is permitted.

If you previously updated from a Git version that tracked `user/credentials/credentials.json`, your saved keys may have been replaced. The repository has since stopped tracking `user/`, but that does not restore lost credentials.

Ranbooru now logs one of two **non-secret** statuses when Gelbooru is selected:

- `Gelbooru API credentials: MISSING/INCOMPLETE (search is anonymous)` — re-enter both your API key and numeric user ID in the Gelbooru settings, and enable **Save credentials** before generating.
- `Gelbooru API credentials: key and user ID supplied` — the credentials were sent. If the tag search still returns HTML, verify the pair, try **Max pages = 1** with a simple tag, and check for temporary Gelbooru restrictions.

In both cases, the error message distinguishes anonymous vs authenticated failures, and neither the log nor the exception reveals the actual secret.


## Max pages and account request quotas

The Gelbooru account limit of **10 requests per 1 second** is a rate cap, **not** a 10-page search limit. With this patch, a normal tag search uses one request to page 0 and at most one additional request for a random deeper page (excluding transient-error retries). A direct post-ID lookup still uses one request.

The extension does not add an artificial rate limiter without evidence of exceeding the quota. Existing rate-limit handling continues to retry HTTP 429 with bounded delays. If Gelbooru serves an HTML page on page 0, inspect the status/type diagnostic line and check account access; that cannot be resolved by selecting a different page.
