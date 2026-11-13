package com.stonewu.fusion.service.ai.youkou;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;

/** 创建请求只发送一次；任务及素材地址均由已配置的 API 地址构造。 */
public final class YoukouHttpClient {
    private static final OkHttpClient BASE = GenerationHttpClients.providerRequestBuilder()
            .connectTimeout(Duration.ofSeconds(30)).readTimeout(Duration.ofSeconds(120))
            .writeTimeout(Duration.ofSeconds(120)).build();
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private final ApiConfig config;
    private final String base;
    private final OkHttpClient client;

    public YoukouHttpClient(ApiConfig config) {
        if (config.getApiKey() == null || config.getApiKey().isBlank()) {
            throw new BusinessException("请填写贰春平台的 API Key");
        }
        this.config = config;
        base = OfficialVideoHttpClient.baseUrl(config, "/v1");
        client = AiProxySupport.okHttpClient(BASE, config);
    }

    public ObjectNode get(String path) {
        Reply reply = read(path);
        if (reply.status() != 200) {
            throw new BusinessException("贰春请求过于频繁，请稍后重试");
        }
        return reply.body();
    }

    public Reply read(String path) {
        return exchange(request(path).get().build(), 200);
    }

    public ObjectNode create(String path, ObjectNode body, String idempotencyKey) {
        return exchange(request(path).header("Idempotency-Key", idempotencyKey)
                .post(RequestBody.create(body.toString(), JSON)).build(), 202).body();
    }

    public String upload(Path file, String mimeType) {
        var body = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("file", "reference", RequestBody.create(file.toFile(), MediaType.get(mimeType)))
                .build();
        ObjectNode response = exchange(request("/media/uploads").post(body).build(), 201).body();
        String id = response.path("media_id").asText();
        if (!id.matches("med_[a-f0-9]{32}")) {
            throw new BusinessException("贰春未返回有效的素材编号，请重新添加素材");
        }
        return id;
    }

    public String taskContent(String id) {
        return base + "/tasks/" + id + "/content";
    }

    private Request.Builder request(String path) {
        return new Request.Builder().url(base + path).header("Authorization", "Bearer " + config.getApiKey())
                .header("Accept", "application/json");
    }

    private Reply exchange(Request request, int expected) {
        GenerationHttpClients.requireNoTransaction();
        try (var response = client.newCall(request).execute()) {
            int retry = retrySeconds(response.header("Retry-After"));
            if (response.code() == 429 && request.method().equals("GET")) {
                return new Reply(429, null, retry);
            }
            if (response.body() == null) {
                throw new BusinessException("贰春响应为空，请查看任务状态后再继续");
            }
            byte[] bytes = response.body().byteStream().readNBytes(64 * 1024 * 1024 + 1);
            if (bytes.length > 64 * 1024 * 1024) {
                throw new BusinessException("贰春响应超过接收限制，请查看任务详情");
            }
            String text = new String(bytes, StandardCharsets.UTF_8);
            if (!response.isSuccessful()) {
                throw GenerationProviderFailure.http(response.code(), text, response.header("Retry-After"));
            }
            if (response.code() != expected) {
                throw new BusinessException("贰春返回了非预期的任务回执，请核对原任务");
            }
            return new Reply(response.code(), YoukouContract.object(text, "贰春响应格式无效，请核对原任务"), retry);
        } catch (IOException failure) {
            throw new UncheckedIOException("贰春连接中断，请核对原任务状态", failure);
        }
    }

    private int retrySeconds(String value) {
        if (value == null) {
            return 5;
        }
        try {
            long seconds = value.matches("[0-9]+") ? Long.parseLong(value)
                    : Duration.between(ZonedDateTime.now(), ZonedDateTime.parse(value,
                    DateTimeFormatter.RFC_1123_DATE_TIME)).toSeconds();
            return (int) Math.max(1, Math.min(Integer.MAX_VALUE, seconds));
        } catch (NumberFormatException | DateTimeParseException invalid) {
            throw new BusinessException("贰春返回的查询间隔无效，请稍后查看任务");
        }
    }

    public record Reply(int status, ObjectNode body, int retrySeconds) {
    }
}
