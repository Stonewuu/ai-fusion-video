package com.stonewu.fusion.service.ai.sonilo;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.SoniloBgmProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

/**
 * Sonilo 音频生成 REST 客户端。
 * <p>
 * 两个接口都是异步任务模型：上传视频提交任务拿 task_id，
 * 轮询 {@code GET /v1/tasks/{task_id}} 到终态（succeeded / failed），
 * 结果里的音轨带预签名下载地址和 license_id（商用留档凭证）。
 * <ul>
 * <li>配乐：{@code POST /v1/video-to-music}（mode=async），视频上限 6 分钟</li>
 * <li>音效：{@code POST /v1/video-to-sfx}（本身即异步任务），视频上限 3 分钟</li>
 * </ul>
 * 任务受理即计费，生成失败自动退款，因此提交不做自动重试；
 * 轮询是免费幂等查询，网络抖动 / 5xx 在时限内继续重试。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SoniloAudioClient {

    private static final String PATH_VIDEO_TO_MUSIC = "/v1/video-to-music";
    private static final String PATH_VIDEO_TO_SFX = "/v1/video-to-sfx";
    private static final String PATH_TASKS = "/v1/tasks/";
    private static final String USER_AGENT = "ai-fusion-video";
    private static final MediaType VIDEO_MP4 = MediaType.get("video/mp4");
    private static final int MAX_ERROR_BODY_CHARS = 300;

    private final SoniloBgmProperties properties;

    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.MINUTES)
            .writeTimeout(15, TimeUnit.MINUTES)
            .build();

    /**
     * 提交配乐生成任务。
     *
     * @param video  成片文件
     * @param prompt 风格提示，可为空（留空按画面内容自动作曲）
     * @return task_id
     */
    public String submitMusicTask(Path video, String prompt) {
        MultipartBody.Builder builder = new MultipartBody.Builder().setType(MultipartBody.FORM)
                // 配乐接口默认流式返回，异步模式才有 task_id 可轮询
                .addFormDataPart("mode", "async");
        return submitTask(PATH_VIDEO_TO_MUSIC, builder, video, prompt);
    }

    /**
     * 提交音效生成任务。
     *
     * @param video  成片文件
     * @param prompt 提示词，可为空
     * @return task_id
     */
    public String submitSfxTask(Path video, String prompt) {
        MultipartBody.Builder builder = new MultipartBody.Builder().setType(MultipartBody.FORM);
        return submitTask(PATH_VIDEO_TO_SFX, builder, video, prompt);
    }

    /**
     * 轮询任务直到终态。succeeded 返回完整任务 JSON；failed 抛业务异常。
     */
    public JSONObject awaitTask(String taskId) {
        long deadline = System.nanoTime() + Duration.ofSeconds(properties.getTimeoutSeconds()).toNanos();
        long pollIntervalMillis = Math.max(0, properties.getPollIntervalSeconds()) * 1000L;
        while (true) {
            JSONObject task = tryQueryTask(taskId);
            if (task != null) {
                String status = task.getStr("status");
                if ("succeeded".equals(status)) {
                    return task;
                }
                if ("failed".equals(status)) {
                    throw new BusinessException(buildTaskFailureMessage(task, taskId));
                }
                // pending / processing 等非终态继续等
            }
            if (System.nanoTime() >= deadline) {
                throw new BusinessException("等待 Sonilo 生成超时（" + properties.getTimeoutSeconds()
                        + "s），任务仍在后台执行，task_id: " + taskId);
            }
            sleep(pollIntervalMillis);
        }
    }

    /**
     * 取任务结果中的主音轨（audio 字段）。
     */
    public SoniloAudioTrack extractPrimaryTrack(JSONObject task, String taskId) {
        JSONObject audio = task.getJSONObject("audio");
        String url = audio != null ? audio.getStr("url") : null;
        if (StrUtil.isBlank(url)) {
            throw new BusinessException("Sonilo 任务成功但未返回音轨结果，task_id: " + taskId);
        }
        return new SoniloAudioTrack(url, audio.getStr("license_id"));
    }

    /**
     * 下载结果音频。结果地址是预签名 URL、自带鉴权，
     * 不能把 API Key 发给存储域名，这里不带任何鉴权头。
     */
    public void downloadAudio(String url, Path dest) {
        Request request = new Request.Builder()
                .url(url)
                .header("User-Agent", USER_AGENT)
                .get()
                .build();
        try (Response response = httpClient.newCall(request).execute()) {
            ResponseBody body = response.body();
            if (!response.isSuccessful() || body == null) {
                throw new BusinessException("配乐结果下载失败 (HTTP " + response.code() + ")");
            }
            try (InputStream in = body.byteStream()) {
                Files.copy(in, dest, StandardCopyOption.REPLACE_EXISTING);
            }
            if (!Files.exists(dest) || Files.size(dest) == 0) {
                throw new BusinessException("配乐结果音频为空");
            }
        } catch (IOException e) {
            throw new BusinessException("配乐结果下载失败: " + e.getMessage());
        }
    }

    private String submitTask(String path, MultipartBody.Builder builder, Path video, String prompt) {
        if (StrUtil.isNotBlank(prompt)) {
            builder.addFormDataPart("prompt", prompt.trim());
        }
        try {
            builder.addFormDataPart("video", video.getFileName().toString(),
                    RequestBody.create(Files.readAllBytes(video), VIDEO_MP4));
        } catch (IOException e) {
            throw new BusinessException("读取成片文件失败: " + e.getMessage());
        }

        Request request = new Request.Builder()
                .url(properties.getBaseUrl() + path)
                .header("Authorization", "Bearer " + properties.getApiKey())
                .header("User-Agent", USER_AGENT)
                .post(builder.build())
                .build();
        try (Response response = httpClient.newCall(request).execute()) {
            String body = readBody(response);
            if (!response.isSuccessful()) {
                throw new BusinessException(buildHttpErrorMessage(response.code(), body));
            }
            String taskId = JSONUtil.isTypeJSONObject(body)
                    ? JSONUtil.parseObj(body).getStr("task_id")
                    : null;
            if (StrUtil.isBlank(taskId)) {
                throw new BusinessException("Sonilo 任务已受理但未返回 task_id，请联系 sonilo.com 支持并附上时间点");
            }
            return taskId;
        } catch (IOException e) {
            throw new BusinessException("Sonilo 请求失败: " + e.getMessage());
        }
    }

    /**
     * 单次查询任务。5xx / 网络异常返回 null 交给上层按时限重试；4xx 直接抛出。
     */
    private JSONObject tryQueryTask(String taskId) {
        Request request = new Request.Builder()
                .url(properties.getBaseUrl() + PATH_TASKS + taskId)
                .header("Authorization", "Bearer " + properties.getApiKey())
                .header("User-Agent", USER_AGENT)
                .get()
                .build();
        try (Response response = httpClient.newCall(request).execute()) {
            String body = readBody(response);
            if (response.code() >= 500) {
                log.warn("[SoniloAudio] 查询任务返回 {}，将重试: taskId={}", response.code(), taskId);
                return null;
            }
            if (!response.isSuccessful()) {
                throw new BusinessException(buildHttpErrorMessage(response.code(), body)
                        + "（task_id: " + taskId + "）");
            }
            return JSONUtil.isTypeJSONObject(body) ? JSONUtil.parseObj(body) : null;
        } catch (IOException e) {
            log.warn("[SoniloAudio] 查询任务网络异常，将重试: taskId={}, {}", taskId, e.getMessage());
            return null;
        }
    }

    private String readBody(Response response) throws IOException {
        ResponseBody body = response.body();
        return body == null ? "" : body.string();
    }

    private String buildHttpErrorMessage(int code, String body) {
        String detail = extractErrorDetail(body);
        return switch (code) {
            case 401 -> "Sonilo API Key 无效，请检查 SONILO_API_KEY 配置";
            case 402 -> "Sonilo 账户余额不足（sonilo.com 可查用量）: " + detail;
            case 413 -> "视频文件过大，超出 Sonilo 上传体积上限: " + detail;
            case 422 -> "Sonilo 参数不合法（常见：视频超出时长上限）: " + detail;
            case 429 -> "触发 Sonilo 频率限制，请稍后重试: " + detail;
            default -> "Sonilo 接口错误 (" + code + "): " + detail;
        };
    }

    private String extractErrorDetail(String body) {
        if (StrUtil.isBlank(body)) {
            return "";
        }
        if (JSONUtil.isTypeJSONObject(body)) {
            JSONObject parsed = JSONUtil.parseObj(body);
            String detail = StrUtil.firstNonBlank(
                    parsed.getStr("detail"), parsed.getStr("message"), parsed.getStr("error"));
            if (StrUtil.isNotBlank(detail)) {
                return detail;
            }
        }
        return StrUtil.maxLength(body, MAX_ERROR_BODY_CHARS);
    }

    private String buildTaskFailureMessage(JSONObject task, String taskId) {
        Object error = task.get("error");
        String message = "生成失败";
        if (error instanceof JSONObject errorObj) {
            message = StrUtil.firstNonBlank(errorObj.getStr("message"), errorObj.getStr("code"), message);
        } else if (error instanceof String errorStr && StrUtil.isNotBlank(errorStr)) {
            message = errorStr;
        }
        String refundNote = Boolean.TRUE.equals(task.getBool("refunded")) ? "，费用已自动退还" : "";
        return "Sonilo 生成失败: " + message + "（task_id: " + taskId + refundNote + "）";
    }

    private void sleep(long millis) {
        if (millis <= 0) {
            return;
        }
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException("等待 Sonilo 生成被中断");
        }
    }
}
