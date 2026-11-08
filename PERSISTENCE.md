# Ranbooru state persistence (Forge Neo)

Ranbooru and LoRAnado remember their UI settings from the last generation performed with at least one of them enabled.

- Saved settings file: `user/last_settings.json`
- Separate values are stored for `txt2img` and `img2img`.
- Generating with both Ranbooru and LoRAnado disabled does not replace the last enabled state.
- API key and user ID are not recorded in `last_settings.json`; the existing "Save credentials" opt-in controls credential persistence.
- Ranbooru controls opt out of Forge Neo's `ui-config.json` restoration because it otherwise overwrites the last-generated values on startup.
- **Every browser page load** (reload, opening a new tab, closing and reopening the browser) fetches the current saved values from disk, even while the WebUI server stays running.
- Restoring a booru also restores rating choices and Gelbooru-specific control visibility.

## Check persistence

1. Enable Ranbooru (or LoRAnado), edit a tag/text field and slider, and press **Generate**.
2. Look for `[Ranbooru] Saved 43 settings for txt2img` (or `img2img`) in the terminal.
3. Verify `user/last_settings.json` exists.
4. Without stopping WebUI, refresh the page or open a new tab. Saved values should reappear.
5. A full WebUI restart should also restore them.

When installing an update, preserve your existing `user/` directory.

## Returned-post selection fix

The API may report 201 total matches while the selected result page contains one post. Ranbooru samples indices only from posts actually returned by that page; for an empty page it tries page one before reporting a clear error. This avoids repeated Generate/Interrupt cycles from out-of-range selection.
