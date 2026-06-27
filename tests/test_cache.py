import asyncio
import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

from cache import ImageCaptionCache


class ImageCaptionCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_cached_caption_for_same_local_file(self):
        cache_hits = []
        cache = ImageCaptionCache(
            on_cache_hit=lambda provider_id, image_count: cache_hits.append(
                (provider_id, image_count)
            )
        )
        image_path = self._temp_path("same-image.png")
        image_path.write_bytes(b"same-image-bytes")
        calls = 0

        async def caption_factory():
            nonlocal calls
            calls += 1
            return "cached caption"

        caption1 = await cache.get_or_create(
            provider_id="caption-provider",
            prompt="describe",
            image_urls=[str(image_path)],
            ttl_seconds=600,
            caption_factory=caption_factory,
        )
        caption2 = await cache.get_or_create(
            provider_id="caption-provider",
            prompt="describe",
            image_urls=[str(image_path)],
            ttl_seconds=600,
            caption_factory=caption_factory,
        )

        self.assertEqual(caption1, "cached caption")
        self.assertEqual(caption2, "cached caption")
        self.assertEqual(calls, 1)
        self.assertEqual(cache_hits, [("caption-provider", 1)])

    async def test_concurrent_waiters_share_caption_factory(self):
        cache = ImageCaptionCache()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def caption_factory():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return "cached caption"

        task1 = asyncio.create_task(
            cache.get_or_create(
                provider_id="caption-provider",
                prompt="describe",
                image_urls=["same-image.png"],
                ttl_seconds=600,
                caption_factory=caption_factory,
            )
        )
        await started.wait()
        task2 = asyncio.create_task(
            cache.get_or_create(
                provider_id="caption-provider",
                prompt="describe",
                image_urls=["same-image.png"],
                ttl_seconds=600,
                caption_factory=caption_factory,
            )
        )

        await asyncio.sleep(0)
        release.set()

        self.assertEqual(await task1, "cached caption")
        self.assertEqual(await task2, "cached caption")
        self.assertEqual(calls, 1)

    async def test_ttl_zero_disables_cache(self):
        cache = ImageCaptionCache()
        calls = 0

        async def caption_factory():
            nonlocal calls
            calls += 1
            return f"caption {calls}"

        caption1 = await cache.get_or_create(
            provider_id="caption-provider",
            prompt="describe",
            image_urls=["same-image.png"],
            ttl_seconds=0,
            caption_factory=caption_factory,
        )
        caption2 = await cache.get_or_create(
            provider_id="caption-provider",
            prompt="describe",
            image_urls=["same-image.png"],
            ttl_seconds=0,
            caption_factory=caption_factory,
        )

        self.assertEqual(caption1, "caption 1")
        self.assertEqual(caption2, "caption 2")
        self.assertEqual(calls, 2)

    async def test_fingerprints_supported_image_reference_types(self):
        cache = ImageCaptionCache(fingerprint_remote_images=False)
        image_bytes = b"same-image-bytes"
        expected_hash = hashlib.sha256(image_bytes).hexdigest()
        image_path = self._temp_path("fingerprint-image.png")
        image_path.write_bytes(image_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")

        self.assertEqual(
            await cache._fingerprint_image(f"base64://{encoded}"),
            expected_hash,
        )
        self.assertEqual(
            await cache._fingerprint_image(f"data:image/png;base64,{encoded}"),
            expected_hash,
        )
        self.assertEqual(await cache._fingerprint_image(str(image_path)), expected_hash)
        self.assertEqual(
            await cache._fingerprint_image("https://example.com/image.png"),
            "url:https://example.com/image.png",
        )
        self.assertEqual(
            await cache._fingerprint_image("missing-image.png"),
            "ref:missing-image.png",
        )

    def _temp_path(self, name):
        return self.tmp_dir / name

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    async def asyncTearDown(self):
        self._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
