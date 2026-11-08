# Local Ranbooru data and safe Git updates

All files under `user/` are **local-only** and ignored by Git, including:

- `user/credentials/credentials.json` (Gelbooru and Rule34 credentials)
- `user/last_settings.json` (last used UI controls)
- `user/booru_recent.json` (recently used boorus)
- `user/search/*.txt` and `user/remove/*.txt` (custom tag lists)

These files are created by Ranbooru when needed. Do **not** commit them or publish credential values.

## One-time migration from older versions

Older versions accidentally committed `user/credentials/credentials.json`,
`user/search/tags_search.txt`, and `user/remove/tags_remove.txt`.
Git ignores *new* files matching `.gitignore`, but still updates files
that are already tracked. Removing them from the repository protects future
updates, **but the first pull with the removal can delete local copies**.

Before your next `git pull`, back up the entire `user/` directory outside
the repository:

```bash
cd ~/AI/sd-webui-forge-neo/extensions/sd-webui-ranbooru-forge-neo
backup="$HOME/ranbooru-user-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup"
cp -a user/. "$backup"/
git pull --ff-only origin main
mkdir -p user
cp -a "$backup"/. user/
```

The backup is deliberately **outside** the Git checkout. Keep it until you
have verified the restored settings and credentials.

If Git refuses the pull due to local modifications in tracked user files,
the backup must be confirmed first. In that case, you can restore only
the three formerly tracked files to their Git version, perform the pull,
then restore the backup. **Do not discard local changes without a backup.**

After updating, confirm `git ls-files user/` prints nothing. Everything
in `user/` is then ignored; normal `git pull` will not replace it.

## If your credentials have already been replaced

A Git pull **cannot recover local secret values** that were never committed
or otherwise backed up. Look for a previous installation folder, personal
backup, or Git stash/commit containing your version. If unavailable,
re-enter your API credentials in Ranbooru and save them again.

If API keys were ever committed to a public Git repository, rotate/revoke
them with their providers. Removing a file from Git tracking does not remove
older Git-history snapshots.
