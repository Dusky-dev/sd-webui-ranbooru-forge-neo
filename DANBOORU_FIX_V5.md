# Danbooru adapter fixes

- Detects Danbooru API error objects instead of treating dictionary keys as post records.
- Reports HTTP status and API messages for rejected searches, permission errors, and rate limits.
- Validates post objects and JSON responses.
- Uses one-based Danbooru pages and retries page one when a randomly selected page has no usable posts.
- Filters animated posts locally instead of spending a Danbooru search-tag slot on `-animated`.
- Supports two user-supplied comma-separated tags with ordinary anonymous/basic account limits.

Danbooru may count rating metatags toward the search limit. Set rating to **All** or remove tags if the API rejects the request. This does not add API-key authentication.

These changes were tested with isolated mocked responses, not a live authenticated Danbooru session.
