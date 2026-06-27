from __future__ import annotations

from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

try:
    from .cache import DEFAULT_IMAGE_CAPTION_CACHE_TTL, ImageCaptionCache
    from .cache import resolve_image_caption_cache_ttl as normalize_ttl
    from .patcher import ImageCaptionCachePatcher
except ImportError:
    from cache import DEFAULT_IMAGE_CAPTION_CACHE_TTL, ImageCaptionCache
    from cache import resolve_image_caption_cache_ttl as normalize_ttl
    from patcher import ImageCaptionCachePatcher


class ImageCaptionCachePlugin(Star):
    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        self.config = config
        self.cache = ImageCaptionCache()
        self._patcher = ImageCaptionCachePatcher(
            cache=self.cache,
            ttl_resolver=self._cache_ttl,
            logger=logger,
        )
        self._patched_targets: list[str] = []

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """Apply image caption cache patches after AstrBot is ready."""
        if not self._config_bool("enabled", True):
            logger.info("image_caption_cache plugin is disabled.")
            return

        self._patched_targets = self._patcher.apply(
            patch_main_agent=self._config_bool("patch_main_agent", True),
            patch_group_chat_context=self._config_bool("patch_group_chat_context", True),
            patch_quoted_message=self._config_bool("patch_quoted_message", True),
        )
        logger.info(
            "image_caption_cache plugin loaded. "
            f"ttl={self._cache_ttl(None)}, "
            f"patched={','.join(self._patched_targets) or 'none'}"
        )

    @filter.command("image_caption_cache_clear")
    async def clear_cache(self, event: AstrMessageEvent):
        """清空图片转述缓存。"""
        removed = self.cache.clear()
        yield event.plain_result(f"已清空图片转述缓存（{removed} 条）。")

    @filter.command("image_caption_cache_stats")
    async def cache_stats(self, event: AstrMessageEvent):
        """查看图片转述缓存状态。"""
        stats = self.cache.stats()
        ttl = self._cache_ttl(None)
        yield event.plain_result(
            f"图片转述缓存：{stats.entries} 条，锁 {stats.locks} 个，TTL {ttl} 秒。"
        )

    async def terminate(self):
        """Restore patched AstrBot functions when the plugin is unloaded."""
        self._patcher.restore()
        self.cache.clear()
        logger.info("image_caption_cache plugin unloaded.")

    def _cache_ttl(self, runtime_config: object | None) -> int:
        plugin_value = self._config_value(
            "image_caption_cache_ttl",
            DEFAULT_IMAGE_CAPTION_CACHE_TTL,
        )
        if plugin_value is not None:
            return normalize_ttl(plugin_value)
        if isinstance(runtime_config, dict):
            return normalize_ttl(runtime_config.get("image_caption_cache_ttl"))
        return DEFAULT_IMAGE_CAPTION_CACHE_TTL

    def _config_bool(self, key: str, default: bool) -> bool:
        value = self._config_value(key, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on", "enable"}

    def _config_value(self, key: str, default: object | None = None) -> object | None:
        if self.config is None:
            return default
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        getter = getattr(self.config, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                value = getter(key)
                return default if value is None else value
        return getattr(self.config, key, default)
