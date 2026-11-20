import launch

# Use cache needs requests-cache 1.0+ (CachedSession with ignored_parameters,
# match_headers and filter_fn). is_installed() alone accepts older releases.
try:
    from importlib.metadata import version
    installed = version("requests-cache")
    major = int(installed.split(".")[0])
except Exception:
    major = None

if major is None or major < 1:
    launch.run_pip('install "requests-cache>=1.0"', "requests-cache>=1.0")
