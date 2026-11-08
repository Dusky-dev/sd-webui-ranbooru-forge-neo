# Ranbooru v8 — recently used boorus and Forge Neo progress

- Uses the **standard booru dropdown** without drag handles or manual sort buttons.
- Pressing Generate with Ranbooru enabled moves the selected booru to the top. Other previously used sites follow in recency order; never-used sites retain the original order.
- txt2img and img2img share a recency list at `user/booru_recent.json`.
- The old `user/booru_order.json` is ignored.
- Browser refresh/new tab reloads the order from disk. Already-open pages do not reorder until reloaded.
- Search and result-count status is reported through Forge Neo's `shared.state.textinfo`, which appears inside its native `.progressDiv` during generation. The display depends on Forge Neo's progress-bar setting.
- The `javascript/ranbooru_progress.js` overlay displays Ranbooru's status on the **right** side of `.progressDiv`, and Forge Neo's **percentage/ETA on the left**. The progress fill width and animation remain unchanged. On narrow bars, labels truncate instead of overlapping.
- The status stays available during sampling and clears when `postprocess` finishes, or immediately on preprocessing failure. The progress bar/overlays themselves disappear when Forge Neo ends the task. Another extension's newer status text is not overwritten.

## Upgrading from v7

Install the updated extension code while keeping `user/`, then restart the WebUI process once. The no-op `javascript/booru_sort.js` and updated `style.css` replace obsolete v7 drag-sort assets so those UI controls no longer appear. Generating with a booru, then refreshing the page, should put that booru first.
