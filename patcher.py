from __future__ import annotations

import importlib
import inspect
import os
import uuid
from collections.abc import Callable
from types import ModuleType
from typing import Any

try:
    from .cache import ImageCaptionCache
except ImportError:
    from cache import ImageCaptionCache


PatchRecord = tuple[ModuleType | type, str, str, Any]


class ImageCaptionCachePatcher:
    def __init__(
        self,
        *,
        cache: ImageCaptionCache,
        ttl_resolver: Callable[[object | None], int],
        logger: Any,
    ) -> None:
        self._cache = cache
        self._ttl_resolver = ttl_resolver
        self._logger = logger
        self._patches: list[PatchRecord] = []

    def apply(
        self,
        *,
        patch_main_agent: bool = True,
        patch_group_chat_context: bool = True,
        patch_quoted_message: bool = True,
    ) -> list[str]:
        applied = []
        if patch_main_agent and self._patch_main_agent_request():
            applied.append("main_agent")
        if patch_quoted_message and self._patch_quoted_message():
            applied.append("quoted_message")
        if patch_group_chat_context and self._patch_group_chat_context():
            applied.append("group_chat_context")
        return applied

    def restore(self) -> None:
        for target, name, original_attr, replacement in reversed(self._patches):
            if getattr(target, name, None) is replacement and hasattr(target, original_attr):
                setattr(target, name, getattr(target, original_attr))
                delattr(target, original_attr)
        self._patches.clear()

    def _patch_main_agent_request(self) -> bool:
        ama = self._import_module("astrbot.core.astr_main_agent")
        if ama is None or not hasattr(ama, "_request_img_caption"):
            return False

        original = getattr(ama, "_request_img_caption")
        if not self._signature_has_prefix(
            original,
            ["provider_id", "cfg", "image_urls", "plugin_context"],
        ):
            self._logger.warning(
                "Skip image caption cache patch: unsupported _request_img_caption signature."
            )
            return False

        async def cached_request_img_caption(
            provider_id: str,
            cfg: dict,
            image_urls: list[str],
            plugin_context: Any,
            prompt: str | None = None,
        ) -> str:
            provider = plugin_context.get_provider_by_id(provider_id)
            provider_cls = getattr(ama, "Provider", None)
            if provider is None:
                raise ValueError(
                    f"Cannot get image caption because provider `{provider_id}` does not exist."
                )
            if provider_cls is not None and not isinstance(provider, provider_cls):
                raise ValueError(
                    "Cannot get image caption because provider "
                    f"`{provider_id}` is not a valid Provider, it is {type(provider)}."
                )

            caption_prompt = prompt or (cfg or {}).get(
                "image_caption_prompt",
                "Please describe the image.",
            )
            cache_provider_id = self._resolve_provider_cache_identity(
                provider,
                configured_provider_id=provider_id,
            )
            ttl = self._ttl_resolver(cfg)

            async def caption_factory() -> str:
                response = await provider.text_chat(
                    prompt=caption_prompt,
                    image_urls=image_urls,
                )
                return getattr(response, "completion_text", "") if response else ""

            return await self._cache.get_or_create(
                provider_id=cache_provider_id,
                prompt=caption_prompt,
                image_urls=list(image_urls),
                ttl_seconds=ttl,
                caption_factory=caption_factory,
            )

        self._replace(
            ama,
            "_request_img_caption",
            cached_request_img_caption,
            "__image_caption_cache_original_request_img_caption",
        )
        return True

    def _patch_quoted_message(self) -> bool:
        ama = self._import_module("astrbot.core.astr_main_agent")
        if ama is None or not hasattr(ama, "_process_quote_message"):
            return False

        original = getattr(ama, "_process_quote_message")
        if not self._signature_has_prefix(
            original,
            ["event", "req", "img_cap_prov_id", "plugin_context"],
        ):
            self._logger.warning(
                "Skip quoted image caption cache patch: unsupported _process_quote_message signature."
            )
            return False

        async def cached_process_quote_message(
            event: Any,
            req: Any,
            img_cap_prov_id: str,
            plugin_context: Any,
            quoted_message_settings: Any = None,
            config: Any = None,
            main_provider_supports_image: bool = False,
            skip_quote_image_caption: bool = False,
        ) -> None:
            if quoted_message_settings is None:
                quoted_message_settings = getattr(
                    ama,
                    "DEFAULT_QUOTED_MESSAGE_SETTINGS",
                    None,
                )

            quote = None
            for comp in event.message_obj.message:
                if isinstance(comp, ama.Reply):
                    quote = comp
                    break
            if not quote:
                return

            content_parts = []
            sender_info = f"({quote.sender_nickname}): " if quote.sender_nickname else ""
            message_str = (
                await ama.extract_quoted_message_text(
                    event,
                    quote,
                    settings=quoted_message_settings,
                )
                or quote.message_str
                or "[Empty Text]"
            )
            content_parts.append(f"{sender_info}{message_str}")

            image_seg = None
            if quote.chain:
                for comp in quote.chain:
                    if isinstance(comp, ama.Image):
                        image_seg = comp
                        break

            if image_seg:
                if skip_quote_image_caption:
                    self._logger.debug(
                        "Skipping quote image captioning because image captioning already handled this request."
                    )
                elif main_provider_supports_image:
                    self._logger.debug(
                        "Skipping quote image captioning because the main provider supports image input."
                    )
                elif not img_cap_prov_id:
                    self._logger.debug(
                        "No dedicated image caption provider configured. "
                        "Skipping quote image captioning."
                    )
                else:
                    await self._append_cached_quoted_image_caption(
                        ama,
                        event,
                        content_parts,
                        image_seg,
                        img_cap_prov_id,
                        plugin_context,
                        config,
                    )

            quoted_content = "\n".join(content_parts)
            quoted_text = f"<Quoted Message>\n{quoted_content}\n</Quoted Message>"
            req.extra_user_content_parts.append(ama.TextPart(text=quoted_text))

        self._replace(
            ama,
            "_process_quote_message",
            cached_process_quote_message,
            "__image_caption_cache_original_process_quote_message",
        )
        return True

    async def _append_cached_quoted_image_caption(
        self,
        ama: ModuleType,
        event: Any,
        content_parts: list[str],
        image_seg: Any,
        img_cap_prov_id: str,
        plugin_context: Any,
        config: Any,
    ) -> None:
        provider = None
        path = None
        compress_path = None
        try:
            provider = plugin_context.get_provider_by_id(img_cap_prov_id)
            if provider is None:
                provider = plugin_context.get_using_provider(event.unified_msg_origin)

            provider_cls = getattr(ama, "Provider", None)
            if not provider or (
                provider_cls is not None and not isinstance(provider, provider_cls)
            ):
                self._logger.warning("No provider found for image captioning in quote.")
                return

            path = await image_seg.convert_to_file_path()
            provider_settings = getattr(config, "provider_settings", None)
            compress_path = await ama._compress_image_for_provider(
                path,
                provider_settings,
            )
            if path and ama._is_generated_compressed_image_path(path, compress_path):
                event.track_temporary_local_file(compress_path)

            caption_prompt = "Please describe the image content."
            cache_provider_id = self._resolve_provider_cache_identity(
                provider,
                configured_provider_id=img_cap_prov_id,
            )
            ttl = self._ttl_resolver(provider_settings)

            async def caption_factory() -> str:
                response = await provider.text_chat(
                    prompt=caption_prompt,
                    image_urls=[compress_path],
                )
                return getattr(response, "completion_text", "") if response else ""

            caption = await self._cache.get_or_create(
                provider_id=cache_provider_id,
                prompt=caption_prompt,
                image_urls=[compress_path],
                ttl_seconds=ttl,
                caption_factory=caption_factory,
            )
            if caption:
                content_parts.append(f"[Image Caption in quoted message]: {caption}")
        except Exception as exc:
            self._logger.error(f"处理引用图片失败: {exc}")
        finally:
            if compress_path and compress_path != path and os.path.exists(compress_path):
                try:
                    os.remove(compress_path)
                except Exception as exc:
                    self._logger.warning(
                        f"Fail to remove temporary compressed image: {exc}"
                    )

    def _patch_group_chat_context(self) -> bool:
        group_context_module = self._import_module(
            "astrbot.builtin_stars.astrbot.group_chat_context"
        )
        if group_context_module is None:
            return False
        group_context_cls = getattr(group_context_module, "GroupChatContext", None)
        if group_context_cls is None or not hasattr(group_context_cls, "get_image_caption"):
            return False

        original = getattr(group_context_cls, "get_image_caption")
        if not self._signature_has_prefix(
            original,
            ["self", "image_url", "image_caption_provider_id", "image_caption_prompt"],
        ):
            self._logger.warning(
                "Skip group chat image caption cache patch: unsupported get_image_caption signature."
            )
            return False

        async def cached_get_image_caption(
            instance: Any,
            image_url: str,
            image_caption_provider_id: str,
            image_caption_prompt: str,
            cache_ttl: int | None = None,
        ) -> str:
            if not image_caption_provider_id:
                provider = instance.context.get_using_provider()
            else:
                provider = instance.context.get_provider_by_id(image_caption_provider_id)
                if not provider:
                    raise ValueError(
                        f"Provider `{image_caption_provider_id}` was not found."
                    )

            provider_cls = getattr(group_context_module, "Provider", None)
            if provider_cls is not None and not isinstance(provider, provider_cls):
                raise ValueError(
                    f"Provider type is invalid for image captioning: {type(provider)}."
                )

            cache_provider_id = self._resolve_provider_cache_identity(
                provider,
                configured_provider_id=image_caption_provider_id,
            )
            ttl = cache_ttl if cache_ttl is not None else self._ttl_resolver(None)

            async def caption_factory() -> str:
                response = await provider.text_chat(
                    prompt=image_caption_prompt,
                    session_id=uuid.uuid4().hex,
                    image_urls=[image_url],
                    persist=False,
                )
                return getattr(response, "completion_text", "") if response else ""

            return await self._cache.get_or_create(
                provider_id=cache_provider_id,
                prompt=image_caption_prompt,
                image_urls=[image_url],
                ttl_seconds=ttl,
                caption_factory=caption_factory,
            )

        self._replace(
            group_context_cls,
            "get_image_caption",
            cached_get_image_caption,
            "__image_caption_cache_original_get_image_caption",
        )
        return True

    def _replace(
        self,
        target: ModuleType | type,
        name: str,
        replacement: Any,
        original_attr: str,
    ) -> None:
        if not hasattr(target, original_attr):
            setattr(target, original_attr, getattr(target, name))
        setattr(target, name, replacement)
        self._patches.append((target, name, original_attr, replacement))

    def _import_module(self, module_name: str) -> ModuleType | None:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            self._logger.warning(
                f"Skip image caption cache patch for {module_name}: {exc}"
            )
            return None

    def _signature_has_prefix(self, func: Any, names: list[str]) -> bool:
        try:
            params = list(inspect.signature(func).parameters)
        except (TypeError, ValueError):
            return False
        return params[: len(names)] == names

    def _resolve_provider_cache_identity(
        self,
        provider: Any,
        *,
        configured_provider_id: str,
    ) -> str:
        if configured_provider_id:
            return configured_provider_id

        provider_config = (
            provider.provider_config if isinstance(provider.provider_config, dict) else {}
        )
        provider_id = provider_config.get("id", "")
        if isinstance(provider_id, str) and provider_id:
            return provider_id

        provider_type = provider_config.get("type", "")
        get_model = getattr(provider, "get_model", None)
        model = get_model() if callable(get_model) else ""
        return ":".join(
            [
                provider.__class__.__module__,
                provider.__class__.__qualname__,
                "" if provider_type is None else str(provider_type),
                "" if model is None else str(model),
            ]
        )
