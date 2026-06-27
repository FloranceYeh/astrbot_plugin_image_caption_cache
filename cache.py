from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

DEFAULT_IMAGE_CAPTION_CACHE_TTL = 600
DEFAULT_IMAGE_CAPTION_CACHE_MAX_IMAGES = 200
CacheHitCallback = Callable[[str, int], None]


def resolve_image_caption_cache_ttl(raw: object) -> int:
    if isinstance(raw, bool) or raw is None:
        return DEFAULT_IMAGE_CAPTION_CACHE_TTL
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return DEFAULT_IMAGE_CAPTION_CACHE_TTL


@dataclass(slots=True)
class CacheStats:
    entries: int
    images: int
    locks: int


@dataclass(slots=True)
class _ImageCaptionCacheEntry:
    caption: str
    image_count: int
    expires_at: float | None
    last_accessed_at: float


class ImageCaptionCache:
    def __init__(
        self,
        on_cache_hit: CacheHitCallback | None = None,
        *,
        ttl_enabled: bool = True,
        image_count_enabled: bool = True,
        max_cached_images: int = DEFAULT_IMAGE_CAPTION_CACHE_MAX_IMAGES,
        fingerprint_remote_images: bool = True,
        remote_fingerprint_timeout: float = 8.0,
        remote_fingerprint_max_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self._entries: dict[str, _ImageCaptionCacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._on_cache_hit = on_cache_hit
        self._ttl_enabled = ttl_enabled
        self._image_count_enabled = image_count_enabled
        self._max_cached_images = max(int(max_cached_images), 0)
        self._fingerprint_remote_images = fingerprint_remote_images
        self._remote_fingerprint_timeout = max(float(remote_fingerprint_timeout), 1.0)
        self._remote_fingerprint_max_bytes = max(int(remote_fingerprint_max_bytes), 1)

    def clear(self) -> int:
        removed = len(self._entries)
        self._entries.clear()
        self._locks.clear()
        return removed

    def stats(self) -> CacheStats:
        self._cleanup_expired_entries()
        return CacheStats(
            entries=len(self._entries),
            images=self._cached_image_count(),
            locks=len(self._locks),
        )

    async def get_or_create(
        self,
        *,
        provider_id: str,
        prompt: str,
        image_urls: list[str],
        ttl_seconds: int,
        caption_factory: Callable[[], Awaitable[str]],
        ttl_enabled: bool | None = None,
        image_count_enabled: bool | None = None,
        max_cached_images: int | None = None,
    ) -> str:
        ttl_seconds = max(int(ttl_seconds), 0)
        ttl_active = (self._ttl_enabled if ttl_enabled is None else ttl_enabled) and (
            ttl_seconds > 0
        )
        count_limit = (
            self._max_cached_images
            if max_cached_images is None
            else max(int(max_cached_images), 0)
        )
        image_count_active = (
            self._image_count_enabled
            if image_count_enabled is None
            else image_count_enabled
        ) and count_limit > 0
        if not ttl_active and not image_count_active:
            return await caption_factory()

        cache_key = await self._build_cache_key(
            provider_id=provider_id,
            prompt=prompt,
            image_urls=image_urls,
        )
        cached_caption = self._get(cache_key)
        if cached_caption is not None:
            self._notify_cache_hit(provider_id, len(image_urls))
            return cached_caption

        lock = self._get_lock(cache_key)
        async with lock:
            cached_caption = self._get(cache_key)
            if cached_caption is not None:
                self._notify_cache_hit(provider_id, len(image_urls))
                return cached_caption

            caption = await caption_factory()
            self._entries[cache_key] = _ImageCaptionCacheEntry(
                caption=caption,
                image_count=max(len(image_urls), 1),
                expires_at=time.monotonic() + ttl_seconds if ttl_active else None,
                last_accessed_at=time.monotonic(),
            )
            self._cleanup_expired_entries()
            if image_count_active:
                self._evict_by_image_count_limit(count_limit)
            return caption

    def _get(self, cache_key: str) -> str | None:
        entry = self._entries.get(cache_key)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at <= time.monotonic():
            self._entries.pop(cache_key, None)
            self._locks.pop(cache_key, None)
            return None
        entry.last_accessed_at = time.monotonic()
        return entry.caption

    def _notify_cache_hit(self, provider_id: str, image_count: int) -> None:
        if self._on_cache_hit is None:
            return
        try:
            self._on_cache_hit(provider_id, image_count)
        except Exception:
            pass

    def _cleanup_expired_entries(self) -> None:
        now = time.monotonic()
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at is not None and entry.expires_at <= now
        ]
        for key in expired_keys:
            self._entries.pop(key, None)
            self._locks.pop(key, None)

    def _evict_by_image_count_limit(self, max_cached_images: int) -> None:
        while self._cached_image_count() > max_cached_images and self._entries:
            oldest_key = min(
                self._entries,
                key=lambda key: self._entries[key].last_accessed_at,
            )
            self._entries.pop(oldest_key, None)
            self._locks.pop(oldest_key, None)

    def _cached_image_count(self) -> int:
        return sum(entry.image_count for entry in self._entries.values())

    def _get_lock(self, cache_key: str) -> asyncio.Lock:
        lock = self._locks.get(cache_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[cache_key] = lock
        return lock

    async def _build_cache_key(
        self,
        *,
        provider_id: str,
        prompt: str,
        image_urls: list[str],
    ) -> str:
        image_fingerprints = []
        for image_url in image_urls:
            image_fingerprints.append(await self._fingerprint_image(image_url))

        joined = "\n".join([provider_id, prompt, *image_fingerprints])
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    async def _fingerprint_image(self, image_url: str) -> str:
        if image_url.startswith("base64://"):
            return self._fingerprint_base64_image(image_url)

        if image_url.startswith("data:image"):
            return self._fingerprint_data_uri_image(image_url)

        if image_url.startswith(("http://", "https://")):
            return await self._fingerprint_remote_image(image_url)

        return await self._fingerprint_local_image(image_url)

    def _fingerprint_base64_image(self, image_url: str) -> str:
        raw_base64 = image_url.removeprefix("base64://")
        try:
            image_bytes = base64.b64decode(raw_base64)
        except Exception:
            return self._reference_fingerprint(image_url)
        return self._hash_bytes(image_bytes)

    def _fingerprint_data_uri_image(self, image_url: str) -> str:
        try:
            _, encoded = image_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        except Exception:
            return self._reference_fingerprint(image_url)
        return self._hash_bytes(image_bytes)

    async def _fingerprint_remote_image(self, image_url: str) -> str:
        if not self._fingerprint_remote_images:
            return f"url:{image_url}"

        try:
            import aiohttp
        except ImportError:
            return f"url:{image_url}"

        try:
            timeout = aiohttp.ClientTimeout(total=self._remote_fingerprint_timeout)
            async with aiohttp.ClientSession(
                trust_env=True,
                timeout=timeout,
            ) as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        return f"url:{image_url}"

                    content_length = response.headers.get("content-length")
                    if (
                        content_length
                        and int(content_length) > self._remote_fingerprint_max_bytes
                    ):
                        return f"url:{image_url}"

                    digest = hashlib.sha256()
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(8192):
                        downloaded += len(chunk)
                        if downloaded > self._remote_fingerprint_max_bytes:
                            return f"url:{image_url}"
                        digest.update(chunk)
                    return f"remote:{digest.hexdigest()}"
        except Exception:
            return f"url:{image_url}"

    async def _fingerprint_local_image(self, image_url: str) -> str:
        local_path = self._to_local_path(image_url)
        if local_path and local_path.is_file():
            image_bytes = await asyncio.to_thread(local_path.read_bytes)
            return self._hash_bytes(image_bytes)

        return self._reference_fingerprint(image_url)

    def _to_local_path(self, image_url: str) -> Path | None:
        if image_url.startswith("file://"):
            parsed = urlparse(image_url)
            parsed_path = unquote(parsed.path)
            if (
                parsed_path.startswith("/")
                and len(parsed_path) >= 3
                and parsed_path[2] == ":"
            ):
                parsed_path = parsed_path[1:]
            return Path(parsed_path)

        if image_url.startswith(("http://", "https://", "base64://", "data:image")):
            return None

        return Path(image_url)

    def _hash_bytes(self, payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _reference_fingerprint(self, image_url: str) -> str:
        return f"ref:{image_url}"
