# 完成总结

## 最终范围

- 本集合成新增可选的「合成后配乐」步骤：拼接成功后把成片交给 Sonilo 按画面生成配乐（时长与成片自动对齐，无提示词也可用），可选同时生成音效音轨，混音后作为最终成片持久化。
- 每条音轨的 license_id 与持久化 URL 随分镜集落库，作为商用留档凭证；配乐音轨同时单独存档。
- 默认关闭（`video.compose.bgm.enabled=false`），未开启时行为与改动前完全一致。

## 关键实现

- `SoniloAudioClient`（OkHttp）：`POST /v1/video-to-music`（mode=async）/ `POST /v1/video-to-sfx` 提交任务，`GET /v1/tasks/{task_id}` 轮询到终态；结果音频是预签名地址，下载时不携带 API Key；401/402/413/422/429 映射为可读中文提示。
- `EpisodeBgmService`：本地 ffprobe 时长预检（配乐 6 分钟、音效 3 分钟上限，探测不到时交后端强校验）；混音统一 `-c:v copy` 零转码，成片有原声时配乐先按 `music-volume` 降音量再 `amix`。
- `VideoComposeService`：配乐步骤在拼接成功后、持久化前执行；配乐失败或超上限跳过均保留无配乐成片并完成合成，原因写入 `bgm_error_msg` 与任务流消息。
- Flyway 迁移 `V1.1.1.2.0__add_episode_bgm_fields.sql` 新增 5 个字段；重新合成时统一重置。

## 验证结果

- `SoniloAudioClientTests`：7 个用例，JDK 内置 HttpServer 模拟接口（不新增测试依赖），覆盖鉴权头、multipart 字段、轮询终态、失败与退款提示、余额不足映射、预签名下载不带鉴权头。
- `EpisodeBgmServiceTests`：6 个用例，覆盖默认关闭、时长上限、三种混音命令形态。
- `VideoComposeServiceTests`、`FlywayMigrationNamingTests` 回归通过。

## 遗留事项

见 `technical-debt.md`：配乐/音效失败的降级取舍、ffmpeg 进程辅助方法的少量重复。
