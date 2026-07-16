"""Patched vLLM server launcher that fixes the prometheus _IncludedRouter bug.

The prometheus-fastapi-instrumentator crashes when iterating routes that include
_IncludedRouter objects (from the RLHF dev router). This patches the route
name extraction to handle them gracefully.
"""
import sys

# Monkey-patch before any vLLM imports
try:
    import prometheus_fastapi_instrumentator.routing as _pfir
    _original = _pfir._get_route_name

    def _patched(scope, routes):
        try:
            return _original(scope, routes)
        except AttributeError:
            return scope.get("path", "unknown")

    _pfir._get_route_name = _patched
except ImportError:
    pass

import runpy
if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
