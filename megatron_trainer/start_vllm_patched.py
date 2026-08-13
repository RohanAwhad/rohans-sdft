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

# Monkey-patch layerwise reload to a no-op: gpt-oss's fused params (w13_weight)
# break the placeholder re-registration ("attribute already exists"). The NCCL
# receive path loads into existing params via model.load_weights, so the
# placeholder pass (a memory optimization) is optional. Patch both the
# layerwise submodule and the reload package re-export (gpu_worker imports
# from the package).
try:
    import vllm.model_executor.model_loader.reload as _reload
    import vllm.model_executor.model_loader.reload.layerwise as _lw
    _noop = lambda model: None
    _lw.initialize_layerwise_reload = _noop
    _reload.initialize_layerwise_reload = _noop
except ImportError:
    pass

import sys
import runpy
if __name__ == "__main__":
    # Logprobs must be post-temperature/post-top-p ("processed") for importance
    # sampling: the IS ratio needs the logp of the actual sampling distribution,
    # which raw mode only equals when temperature=1.0.
    if "--logprobs-mode" not in sys.argv:
        sys.argv += ["--logprobs-mode", "processed_logprobs"]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
