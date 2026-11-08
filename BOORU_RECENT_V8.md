# Ranbooru v8 — recently used boorus and Forge Neo progress

- Uses the **standard booru dropdown** without drag handles or manual sort buttons.
- Pressing Generate with Ranbooru enabled moves the selected booru to the top. Other previously used sites follow in recency order; never-used sites retain the original order.
- txt2img and img2img share a recency list at `user/booru_recent.json`.
- The old `user/booru_order.json` is ignored.
- Browser refresh/new tab reloads the order from disk. Already-open pages do not reorder until reloaded.
- Search and result-count status is reported through Forge Neo's `shared.state.textinfo`, which appears inside its native `.progressDiv` during generation. The display depends on Forge Neo's progress-bar setting.
- These messages are temporary: Ranbooru clears them when its `before_process` hook exits, before normal sampling progress/ETA starts. Error paths also clear the message. Other extensions' newer status text is preserved.

## Upgrading from v7

Install the updated extension code while keeping `user/`, then restart the WebUI process once. The no-op `javascript/booru_sort.js` and `style.css` replace obsolete v7 drag-sort assets so those UI controls no longer appear. Generating with a booru, then refreshing the page, should put that booru first.
