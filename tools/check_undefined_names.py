"""Fail when pyflakes reports an undefined name in the given files.

Other pyflakes warnings (unused imports and the like) are printed but do not
fail. An undefined name is the kind of mistake that only shows up as a crash
when Forge Neo builds the panel or runs a generation.

    python tools/check_undefined_names.py scripts/ranbooru.py tests
"""
import subprocess
import sys


def main(paths):
    result = subprocess.run(
        [sys.executable, '-m', 'pyflakes', *paths],
        capture_output=True, text=True,
    )
    if result.returncode not in (0, 1) or 'No module named pyflakes' in result.stderr:
        sys.stderr.write(result.stderr)
        return 2
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    undefined = [line for line in lines if 'undefined name' in line]
    for line in lines:
        print(line)
    if undefined:
        print(f'\n{len(undefined)} undefined name(s) found.', file=sys.stderr)
        return 1
    print('No undefined names.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or ['scripts/ranbooru.py']))
