from __future__ import annotations

import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

try:
    from .cache import DEFAULT_IMAGE_CAPTION_CACHE_MAX_IMAGES
    from .cache import DEFAULT_IMAGE_CAPTION_CACHE_TTL, ImageCaptionCache
    from .cache import resolve_image_caption_cache_ttl as normalize_ttl
    from .patcher import ImageCaptionCachePatcher
except ImportError:
    from cache import DEFAULT_IMAGE_CAPTION_CACHE_MAX_IMAGES
    from cache import DEFAULT_IMAGE_CAPTION_CACHE_TTL, ImageCaptionCache
    from cache import resolve_image_caption_cache_ttl as normalize_ttl
    from patcher import ImageCaptionCachePatcher


_CACHE_HIT_LOG_DEDUP_SECONDS = 1.0


class ImageCaptionCachePlugin(Star):
    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        self.config = config
        self.cache = ImageCaptionCache(
            on_cache_hit=self._log_cache_hit,
            ttl_enabled=self._ttl_strategy_enabled(),
            image_count_enabled=self._image_count_strategy_enabled(),
            max_cached_images=self._config_int(
                "max_cached_images",
                DEFAULT_IMAGE_CAPTION_CACHE_MAX_IMAGES,
            ),
            fingerprint_remote_images=self._config_bool(
                "fingerprint_remote_images",
                True,
            ),
            remote_fingerprint_timeout=self._config_float(
                "remote_fingerprint_timeout",
                8.0,
            ),
            remote_fingerprint_max_bytes=self._config_int(
                "remote_fingerprint_max_bytes",
                20 * 1024 * 1024,
            ),
        )
        self._patcher = ImageCaptionCachePatcher(
            cache=self.cache,
            ttl_resolver=self._cache_ttl,
            logger=logger,
        )
        self._patched_targets: list[str] = []
        self._recent_cache_hit_logs: dict[str, float] = {}
        self._apply_patches("init")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """Apply image caption cache patches after AstrBot is ready."""
        self._apply_patches("astrbot_loaded")

    def _apply_patches(self, reason: str) -> None:
        if not self._config_bool("enabled", True):
            logger.info("image_caption_cache plugin is disabled.")
            return
        if self._patched_targets:
            logger.debug(
                "image_caption_cache plugin already patched. "
                f"reason={reason}, patched={','.join(self._patched_targets)}"
            )
            return

        self._patched_targets = self._patcher.apply(
            patch_main_agent=self._config_bool("patch_main_agent", True),
            patch_quoted_message=self._config_bool("patch_quoted_message", True),
        )
        logger.info(
            "image_caption_cache plugin loaded. "
            f"reason={reason}, "
            f"ttl_enabled={self._ttl_strategy_enabled()}, "
            f"ttl={self._cache_ttl(None)}, "
            f"image_count_enabled={self._image_count_strategy_enabled()}, "
            f"max_cached_images={self._max_cached_images()}, "
            f"patched={','.join(self._patched_targets) or 'none'}"
        )
        if not self._patched_targets:
            logger.warning(
                "image_caption_cache plugin did not patch any target. "
                "Check AstrBot version and core function signatures."
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
            f"图片转述缓存：{stats.entries} 条，{stats.images} 张图，"
            f"锁 {stats.locks} 个；"
            f"TTL 策略：{self._enabled_text(self._ttl_strategy_enabled())}"
            f"（{ttl} 秒）；"
            "图片数量策略："
            f"{self._enabled_text(self._image_count_strategy_enabled())}"
            f"（上限 {self._max_cached_images()} 张）；"
            f"补丁：{','.join(self._patched_targets) or 'none'}。"
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

    def _log_cache_hit(self, provider_id: str, image_count: int, cache_key: str) -> None:
        now = time.monotonic()
        last_logged_at = self._recent_cache_hit_logs.get(cache_key)
        if (
            last_logged_at is not None
            and now - last_logged_at < _CACHE_HIT_LOG_DEDUP_SECONDS
        ):
            return
        self._recent_cache_hit_logs[cache_key] = now
        self._cleanup_recent_cache_hit_logs(now)
        logger.info(
            "图片转述缓存命中。"
            f"provider={provider_id or '<default>'}, images={image_count}"
        )

    def _cleanup_recent_cache_hit_logs(self, now: float) -> None:
        expired_keys = [
            key
            for key, logged_at in self._recent_cache_hit_logs.items()
            if now - logged_at >= _CACHE_HIT_LOG_DEDUP_SECONDS
        ]
        for key in expired_keys:
            self._recent_cache_hit_logs.pop(key, None)

    def _ttl_strategy_enabled(self) -> bool:
        return self._config_bool("enable_ttl_cache", True) and self._cache_ttl(None) > 0

    def _image_count_strategy_enabled(self) -> bool:
        return (
            self._config_bool("enable_image_count_cache", True)
            and self._max_cached_images() > 0
        )

    def _max_cached_images(self) -> int:
        return self._config_int(
            "max_cached_images",
            DEFAULT_IMAGE_CAPTION_CACHE_MAX_IMAGES,
        )

    def _enabled_text(self, enabled: bool) -> str:
        return "开启" if enabled else "关闭"

    def _config_int(self, key: str, default: int) -> int:
        value = self._config_value(key, default)
        if isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _config_float(self, key: str, default: float) -> float:
        value = self._config_value(key, default)
        if isinstance(value, bool):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

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
