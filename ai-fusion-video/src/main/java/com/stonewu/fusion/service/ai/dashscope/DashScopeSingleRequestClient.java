package com.stonewu.fusion.service.ai.dashscope;

import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationParam;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationResult;
import com.alibaba.dashscope.aigc.videosynthesis.VideoSynthesisParam;
import com.alibaba.dashscope.aigc.videosynthesis.VideoSynthesisResult;
import com.alibaba.dashscope.common.DashScopeResult;
import com.alibaba.dashscope.protocol.ApiServiceOption;
import com.alibaba.dashscope.protocol.AsyncTaskOption;
import com.alibaba.dashscope.protocol.HalfDuplexRequest;
import com.alibaba.dashscope.protocol.HttpMethod;
import com.alibaba.dashscope.protocol.okhttp.OkHttpHttpClient;
import com.alibaba.dashscope.task.AsyncTaskParam;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import okhttp3.OkHttpClient;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * 使用 SDK 的参数、请求和结果协议，显式注入不重发的 HTTP 传输。输入素材已由生成准备阶段解析。
 */
public final class DashScopeSingleRequestClient {
    private static final OkHttpClient BASE = GenerationHttpClients.providerRequestBuilder()
            .connectTimeout(1, TimeUnit.MINUTES)
            .readTimeout(25, TimeUnit.MINUTES)
            .writeTimeout(1, TimeUnit.MINUTES)
            .build();
    private final String baseUrl;
    private final OkHttpHttpClient client;

    public DashScopeSingleRequestClient(ApiConfig config) {
        baseUrl = DashScopeGenerationSupport.resolveGenerationBaseUrl(config);
        client = new OkHttpHttpClient(AiProxySupport.okHttpClient(BASE, config));
    }

    public Supplier<ImageGenerationResult> prepareImage(ImageGenerationParam param, boolean asynchronous) {
        try {
            param.validate();
        } catch (Exception failure) {
            throw new IllegalArgumentException("DashScope 图片参数无效", failure);
        }
        var option = ApiServiceOption.builder().baseHttpUrl(baseUrl)
                .taskGroup("aigc").task(asynchronous ? "image-generation" : "multimodal-generation")
                .function("generation").isAsyncTask(asynchronous).build();
        var request = new HalfDuplexRequest(param, option);
        return () -> {
            try {
                return ImageGenerationResult.fromDashScopeResult(client.send(request));
            } catch (Exception failure) {
                throw new IllegalStateException("DashScope 图片调用异常", failure);
            }
        };
    }

    public Supplier<VideoSynthesisResult> prepareVideo(VideoSynthesisParam param) {
        try {
            param.validate();
        } catch (Exception failure) {
            throw new IllegalArgumentException("DashScope 视频参数无效", failure);
        }
        var option = ApiServiceOption.builder().baseHttpUrl(baseUrl).taskGroup("aigc")
                .task(param.getModel().contains("kf2v") ? "image2video" : "video-generation")
                .function("video-synthesis").isAsyncTask(true).build();
        var request = new HalfDuplexRequest(param, option);
        return () -> {
            try {
                return VideoSynthesisResult.fromDashScopeResult(client.send(request));
            } catch (Exception failure) {
                throw new IllegalStateException("DashScope 视频调用异常", failure);
            }
        };
    }

    public DashScopeResult fetch(String taskId, String apiKey) throws Exception {
        var option = AsyncTaskOption.builder().baseHttpUrl(baseUrl).httpMethod(HttpMethod.GET)
                .url("/tasks/" + URLEncoder.encode(taskId, StandardCharsets.UTF_8).replace("+", "%20")).build();
        var param = AsyncTaskParam.builder().taskId(taskId).apiKey(apiKey).build();
        return client.send(new HalfDuplexRequest(param, option));
    }
}
