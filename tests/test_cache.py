import asyncio
import base64
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

from cache import ImageCaptionCache
from patcher import ImageCaptionCachePatcher


class _FakeProvider:
    def __init__(self, provider_id, model, provider_type="openai_chat_completion"):
        self.provider_config = {"id": provider_id, "type": provider_type}
        self.model = model
        self.calls = []

    def get_model(self):
        return self.model

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(completion_text="generated caption")


class _FakeLogger:
    def __init__(self):
        self.debugs = []
        self.infos = []
        self.warnings = []

    def debug(self, message):
        self.debugs.append(message)

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


class ImageCaptionCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_cached_caption_for_same_local_file(self):
        cache_hits = []
        cache = ImageCaptionCache(
            on_cache_hit=lambda provider_id, image_count, cache_key: cache_hits.append(
                (provider_id, image_count, cache_key)
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
        self.assertEqual(len(cache_hits), 1)
        self.assertEqual(cache_hits[0][0], "caption-provider")
        self.assertEqual(cache_hits[0][1], 1)
        self.assertIsInstance(cache_hits[0][2], str)

    async def test_reports_every_cache_hit_to_callback(self):
        cache_hits = []
        cache = ImageCaptionCache(
            on_cache_hit=lambda provider_id, image_count, cache_key: cache_hits.append(
                (provider_id, image_count, cache_key)
            )
        )
        calls = 0

        async def caption_factory():
            nonlocal calls
            calls += 1
            return "cached caption"

        for _ in range(3):
            await cache.get_or_create(
                provider_id="caption-provider",
                prompt="describe",
                image_urls=["same-image.png"],
                ttl_seconds=600,
                caption_factory=caption_factory,
            )

        self.assertEqual(calls, 1)
        self.assertEqual(len(cache_hits), 2)
        self.assertEqual(cache_hits[0][2], cache_hits[1][2])

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

    async def test_disables_cache_when_all_strategies_are_disabled(self):
        cache = ImageCaptionCache(image_count_enabled=False)
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

    async def test_image_count_cache_reuses_when_ttl_disabled(self):
        cache = ImageCaptionCache(
            ttl_enabled=False,
            image_count_enabled=True,
            max_cached_images=10,
        )
        calls = 0

        async def caption_factory():
            nonlocal calls
            calls += 1
            return "cached by image count"

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

        self.assertEqual(caption1, "cached by image count")
        self.assertEqual(caption2, "cached by image count")
        self.assertEqual(calls, 1)
        self.assertEqual(cache.stats().images, 1)

    async def test_image_count_cache_evicts_least_recently_used_entries(self):
        cache = ImageCaptionCache(
            ttl_enabled=False,
            image_count_enabled=True,
            max_cached_images=1,
        )
        calls = 0

        async def caption_factory():
            nonlocal calls
            calls += 1
            return f"caption {calls}"

        first_caption = await cache.get_or_create(
            provider_id="caption-provider",
            prompt="describe",
            image_urls=["image-one.png"],
            ttl_seconds=0,
            caption_factory=caption_factory,
        )
        second_caption = await cache.get_or_create(
            provider_id="caption-provider",
            prompt="describe",
            image_urls=["image-two.png"],
            ttl_seconds=0,
            caption_factory=caption_factory,
        )
        first_caption_after_eviction = await cache.get_or_create(
            provider_id="caption-provider",
            prompt="describe",
            image_urls=["image-one.png"],
            ttl_seconds=0,
            caption_factory=caption_factory,
        )

        self.assertEqual(first_caption, "caption 1")
        self.assertEqual(second_caption, "caption 2")
        self.assertEqual(first_caption_after_eviction, "caption 3")
        self.assertEqual(calls, 3)
        self.assertEqual(cache.stats().entries, 1)
        self.assertEqual(cache.stats().images, 1)

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


class ProviderCacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.logger = _FakeLogger()
        self.patcher = ImageCaptionCachePatcher(
            cache=ImageCaptionCache(),
            ttl_resolver=lambda _: 600,
            logger=self.logger,
        )

    def test_identity_uses_resolved_provider_instead_of_configured_provider(self):
        provider = _FakeProvider("fallback-provider", "vision-model")

        identity = self.patcher._resolve_provider_cache_identity(
            provider,
            configured_provider_id="configured-provider",
        )

        self.assertEqual(identity, "fallback-provider")

    def test_identity_does_not_duplicate_the_model_name(self):
        provider = _FakeProvider("caption-provider", "vision-model-a")
        identity_a = self.patcher._resolve_provider_cache_identity(
            provider,
            configured_provider_id="caption-provider",
        )

        provider.model = "vision-model-b"
        identity_b = self.patcher._resolve_provider_cache_identity(
            provider,
            configured_provider_id="caption-provider",
        )

        self.assertEqual(identity_a, "caption-provider")
        self.assertEqual(identity_b, "caption-provider")

    def test_identity_falls_back_to_configured_id_when_provider_has_no_id(self):
        provider = _FakeProvider("", "vision-model")

        identity = self.patcher._resolve_provider_cache_identity(
            provider,
            configured_provider_id="configured-provider",
        )

        self.assertEqual(identity, "configured-provider")


class VisualModelLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_every_visual_model_call_with_actual_identity(self):
        logger = _FakeLogger()
        patcher = ImageCaptionCachePatcher(
            cache=ImageCaptionCache(),
            ttl_resolver=lambda _: 600,
            logger=logger,
        )
        provider = _FakeProvider("fallback-provider", "vision-model")

        caption = await patcher._call_visual_model(
            provider,
            provider_identity="fallback-provider",
            prompt="describe",
            image_urls=["image-one.png", "image-two.png"],
        )

        self.assertEqual(caption, "generated caption")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            logger.infos,
            [
                "Image caption visual model call. "
                "provider=fallback-provider, images=2"
            ],
        )


class NativeVisionCaptionCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_vision_requests_are_captioned_and_cached(self):
        cache_hits = []
        cache = ImageCaptionCache(
            on_cache_hit=lambda provider_id, image_count, cache_key: cache_hits.append(
                (provider_id, image_count, cache_key)
            )
        )
        logger = _FakeLogger()
        patcher = ImageCaptionCachePatcher(
            cache=cache,
            ttl_resolver=lambda _: 600,
            logger=logger,
        )
        provider = _FakeProvider("caption-provider", "vision-model")
        context = SimpleNamespace(
            get_provider_by_id=lambda provider_id: (
                provider if provider_id == "caption-provider" else None
            ),
            get_config=lambda **_: {"provider_settings": {}},
        )
        event = SimpleNamespace(unified_msg_origin="webchat:friend:test")
        provider_settings = {
            "default_image_caption_provider_id": "caption-provider",
            "image_caption_prompt": "describe",
        }
        config = SimpleNamespace(provider_settings=provider_settings)
        original_requests = []
        ama = ModuleType("astrbot.core.astr_main_agent")
        ama.Provider = _FakeProvider
        ama._provider_supports_modality = lambda _provider, modality: (
            modality == "image"
        )

        async def original_request_img_caption(
            provider_id,
            cfg,
            image_urls,
            plugin_context,
        ):
            raise AssertionError("the original image caption request must be patched")

        async def ensure_img_caption(
            _event,
            req,
            cfg,
            plugin_context,
            image_caption_provider,
        ):
            caption = await ama._request_img_caption(
                image_caption_provider,
                cfg,
                req.image_urls,
                plugin_context,
            )
            req.extra_user_content_parts.append(caption)
            req.image_urls = []

        async def original_decorate_llm_request(
            event,
            req,
            plugin_context,
            config,
            provider=None,
        ):
            del event, plugin_context, config
            original_requests.append(
                (list(req.image_urls), list(req.extra_user_content_parts), provider)
            )

        ama._request_img_caption = original_request_img_caption
        ama._ensure_img_caption = ensure_img_caption
        ama._decorate_llm_request = original_decorate_llm_request
        patcher._import_module = lambda _module_name: ama

        applied = patcher.apply(patch_quoted_message=False)
        self.assertEqual(applied, ["main_agent", "native_vision"])

        for _ in range(2):
            req = SimpleNamespace(
                conversation=object(),
                image_urls=["same-image.png"],
                extra_user_content_parts=[],
            )
            await ama._decorate_llm_request(event, req, context, config, provider)

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(cache_hits), 1)
        self.assertEqual(cache.stats().entries, 1)
        self.assertEqual(
            [(images, captions) for images, captions, _ in original_requests],
            [([], ["generated caption"]), ([], ["generated caption"])],
        )


if __name__ == "__main__":
    unittest.main()
