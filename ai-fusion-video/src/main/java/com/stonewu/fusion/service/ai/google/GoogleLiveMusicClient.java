package com.stonewu.fusion.service.ai.google;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.media.MediaIoBoundary;
import lombok.RequiredArgsConstructor;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;
import org.springframework.stereotype.Component;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** 将 Lyria 的音乐流收集为指定时长的音频文件。 */
@Component
@RequiredArgsConstructor
public class GoogleLiveMusicClient {
    private static final OkHttpClient CLIENT = new OkHttpClient.Builder().connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.SECONDS).retryOnConnectionFailure(false).followRedirects(false).followSslRedirects(false).build();
    private final GoogleNativeApi google;

    public GoogleAudioProtocol.Output generate(AiModel model, ApiConfig api, String prompt, JSONObject config, Runnable active) {
        MediaIoBoundary.requireOutsideTransaction();
        if (!google.developerApi(api)) {
            throw new BusinessException("此音乐模型需要 Gemini Developer API 渠道");
        }
        int seconds = config.getInt("durationSeconds", 10);
        if (seconds < 5 || seconds > 60) {
            throw new BusinessException("音乐时长应为 5–60 秒");
        }
        int length = seconds * 44100 * 4;
        var audio = new ByteArrayOutputStream();
        var done = new CompletableFuture<Void>();
        var url = HttpUrl.get(google.developerBase(api)).newBuilder()
                .encodedPath("/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateMusic")
                .query(null).addQueryParameter("key", google.headers(api).get("x-goog-api-key")).build();
        var socket = AiProxySupport.okHttpClient(CLIENT, api).newWebSocket(new Request.Builder().url(url).build(), new WebSocketListener() {
            @Override
            public void onOpen(WebSocket ws, Response response) {
                ws.send(new JSONObject().set("setup", new JSONObject().set("model", "models/" + model.getCode())).toString());
            }

            @Override
            public void onMessage(WebSocket ws, ByteString message) {
                onMessage(ws, message.utf8());
            }

            @Override
            public void onMessage(WebSocket ws, String message) {
                try {
                    active.run();
                    var result = JSONUtil.parseObj(message);
                    if (result.containsKey("error")) {
                        throw new BusinessException("Google 未接受音乐生成请求，请检查模型配置");
                    }
                    if (result.containsKey("setupComplete")) {
                        ws.send(new JSONObject().set("clientContent", new JSONObject().set("weightedPrompts",
                                new JSONArray().put(new JSONObject().set("text", prompt).set("weight", 1.0)))).toString());
                        ws.send(new JSONObject().set("playbackControl", "PLAY").toString());
                    }
                    var chunks = result.getByPath("serverContent.audioChunks", JSONArray.class);
                    if (chunks != null) {
                        for (JSONObject chunk : chunks.toList(JSONObject.class)) {
                            byte[] bytes = Base64.getDecoder().decode(chunk.getStr("data"));
                            audio.write(bytes, 0, Math.min(bytes.length, Math.max(0, length - audio.size())));
                        }
                        if (audio.size() >= length) {
                            done.complete(null);
                        }
                    }
                } catch (RuntimeException failure) {
                    done.completeExceptionally(failure);
                }
            }

            @Override
            public void onFailure(WebSocket ws, Throwable failure, Response response) {
                done.completeExceptionally(new BusinessException("音乐生成连接中断，请重试"));
            }

            @Override
            public void onClosing(WebSocket ws, int code, String reason) {
                done.completeExceptionally(new BusinessException("音乐生成提前结束，请重试"));
                ws.close(code, "acknowledged");
            }
        });
        try {
            done.get(120, TimeUnit.SECONDS);
            active.run();
            byte[] wave = GoogleAudio.wave(audio.toByteArray(), 44100, 2);
            return new GoogleAudioProtocol.Output("", List.of(new GoogleAudioProtocol.Audio(Base64.getEncoder().encodeToString(wave), "audio/wav")));
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new BusinessException("音乐生成已停止");
        } catch (ExecutionException failure) {
            if (failure.getCause() instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new BusinessException("音乐生成失败，请重试");
        } catch (TimeoutException timeout) {
            throw new BusinessException("音乐生成等待超时，请稍后重试");
        } finally {
            socket.close(1000, "completed");
            socket.cancel();
        }
    }
}
