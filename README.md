# astrbot_plugin_image_caption_cache

为 AstrBot 图片转述结果增加 TTL 内存缓存，参考 AstrBotDevs/AstrBot#8966 的实现思路做成外置插件。

## 功能

- 主对话图片转述缓存：同一 provider、同一提示词、同一图片在 TTL 内复用结果。
- 引用消息图片转述缓存。
- 群聊上下文图片转述缓存。
- 支持 `base64://`、`data:image...`、本地文件、`file://` 和远程 URL 的图片指纹。
- 插件卸载时自动恢复被补丁覆盖的 AstrBot 核心函数。

远程图片不会被下载求哈希，缓存键使用 URL 本身；本地图片和 base64 图片使用内容哈希。

## 配置

在 AstrBot WebUI 的插件配置中设置：

- `enabled`：是否启用补丁，默认开启。
- `image_caption_cache_ttl`：缓存有效期，单位秒，默认 `600`；设为 `0` 表示禁用缓存。
- `patch_main_agent`：缓存主对话图片转述，默认开启。
- `patch_quoted_message`：缓存引用消息图片转述，默认开启。
- `patch_group_chat_context`：缓存群聊上下文图片转述，默认开启。
- `fingerprint_remote_images`：远程图片使用内容指纹，默认开启；用于处理平台每次生成不同临时 URL 的情况。
- `remote_fingerprint_timeout`：远程图片内容指纹下载超时，默认 `8` 秒。
- `remote_fingerprint_max_bytes`：远程图片内容指纹最大下载字节数，默认 `20971520`。

修改配置后请重载插件。

## 命令

- `/image_caption_cache_stats`：查看当前缓存条目数、锁数量和 TTL。
- `/image_caption_cache_clear`：清空缓存。

## 说明

AstrBot 目前没有为外部插件暴露“图片转述前”的稳定 hook，因此本插件在构造时和 AstrBot 加载完成时都会尝试对核心函数做运行时补丁。补丁会检查目标函数签名；如果当前 AstrBot 版本接口不兼容，会跳过对应接入点并写入日志。可通过 `/image_caption_cache_stats` 查看当前补丁目标。
