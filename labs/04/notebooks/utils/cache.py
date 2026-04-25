from functools import lru_cache

from joblib import Memory


_memory = Memory(".joblib-cache", verbose=0)


def _cache_stable(func=None, *, module: str | None = None):
    """Cache a function in the shared `.joblib-cache` store."""

    def deco(inner_func):
        if module is not None:
            inner_func.__module__ = module
        return _memory.cache(inner_func)

    if func is None:
        return deco
    return deco(func)


# ---------------------------------------------------------------------------
# LRU cache registry with config-aware invalidation
#
# All lru_cached functions in the pipeline register themselves here so that
# clear_all_lru_caches() can invalidate them when config/cosmology changes.
# cosmology.init() calls clear_all_lru_caches() on re-initialization.
# ---------------------------------------------------------------------------

_lru_registry: list = []


def _register_lru(cached_func):
    """Register an @lru_cache-decorated function for bulk invalidation.

    Use as a decorator *outside* @lru_cache::

        @_register_lru
        @lru_cache(maxsize=128)
        def _my_cached_func(...):
            ...

    The outer decorator receives the lru_cache wrapper (which has
    .cache_clear()) and stores it in the registry.
    """
    _lru_registry.append(cached_func)
    return cached_func


def clear_all_lru_caches():
    """Clear every registered lru_cache.

    Called by cosmology.init() when the cosmological parameters change,
    ensuring that no in-memory cache returns stale results.
    """
    for func in _lru_registry:
        func.cache_clear()
