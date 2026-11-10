package com.stonewu.fusion.service.generation.image.strategy.gemini;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.provider.VertexAiAuthentication;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import lombok.RequiredArgsConstructor;
import okhttp3.MediaType;
import okhttp3.Request;
import okhttp3.RequestBody;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/** Gemini 生图使用 generateContent，Imagen 仍使用 predict；两者不能互换。 */
@Component
@RequiredArgsConstructor
public class VertexGeminiImageProtocol {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final ReferenceImageTransportService references;
    private final GenerationModelCapabilityService capabilities;

    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig api, int[] size) {
        var sources = references.parseJsonInputs(task.getRefImageUrls(), "参考图");
        var images = sources.stream().map(source -> references.toDataUri(source, api)).toList();
        var config = capabilities.getMergedModelConfig(model);
        var output = outputOptions(task, size, config);
        String body = requestBody(task.getPrompt(), output[0], output[1], images);
        boolean developerApi = "gemini".equals(api.getPlatform());
        String url = developerApi
                ? StrUtil.blankToDefault(api.getApiUrl(), "https://generativelanguage.googleapis.com").replaceAll("/+$", "")
                + "/v1beta/models/" + model.getCode() + ":generateContent"
                : VertexAiAuthentication.modelUrl(api, model.getCode(), "generateContent");
        var request = new Request.Builder().url(url)
                .post(RequestBody.create(body, MediaType.get("application/json")));
        if (developerApi) {
            if (StrUtil.isBlank(api.getApiKey())) {
                throw new BusinessException("请填写 Gemini API Key");
            }
            request.header("x-goog-api-key", api.getApiKey());
        } else {
            VertexAiAuthentication.headers(api).forEach(request::header);
        }
        var prepared = request.build();
        var client = AiProxySupport.okHttpClient(GenerationHttpClients.providerRequestBuilder()
                .connectTimeout(1, TimeUnit.MINUTES).readTimeout(25, TimeUnit.MINUTES).build(), api);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            try (var response = client.newCall(prepared).execute()) {
                String content = response.body() == null ? "" : response.body().string();
                if (!response.isSuccessful()) {
                    throw new BusinessException("图片生成失败，请检查模型配置后重试");
                }
                return GenerationProviderReceipt.completed(RemoteGenerationResult.success(
                        imageOutputs(content).stream().map(RemoteGenerationResult.Output::image).toList()));
            } catch (IOException failure) {
                throw new BusinessException("图片服务连接中断，请重试");
            }
        });
    }

    static String requestBody(String prompt, int width, int height, List<String> images) {
        return requestBody(prompt, aspectRatio(width, height), sizeTier(width, height), images);
    }

    static String[] outputOptions(ImageTask task, int[] size, JSONObject config) {
        String ratio = StrUtil.blankToDefault(task.getAspectRatio(), task.getRatio());
        String resolution = task.getResolution();
        // 历史任务及工具调用允许只传宽高；显式比例与档位始终原样传递。
        if (StrUtil.isBlank(ratio)) {
            ratio = aspectRatio(size[0], size[1]);
        }
        if (StrUtil.isBlank(resolution)) {
            resolution = sizeTier(size[0], size[1]);
        }
        return new String[]{ratio, Boolean.FALSE.equals(config.getBool("nativeImageSize")) ? null : resolution};
    }

    private static String sizeTier(int width, int height) {
        long pixels = (long) width * height;
        return pixels > 6_000_000 ? "4K" : pixels > 2_000_000 ? "2K" : "1K";
    }

    static String requestBody(String prompt, String ratio, String resolution, List<String> images) {
        var root = JSON.createObjectNode();
        var content = root.putArray("contents").addObject();
        content.put("role", "user");
        var parts = content.putArray("parts");
        for (String image : images) {
            int separator = image.indexOf(";base64,");
            if (!image.startsWith("data:image/") || separator < 0 || separator + 8 == image.length()) {
                throw new BusinessException("Gemini 参考图必须包含有效的图片 Data URI");
            }
            var inline = parts.addObject().putObject("inlineData");
            inline.put("mimeType", image.substring(5, separator));
            inline.put("data", image.substring(separator + 8));
        }
        parts.addObject().put("text", prompt);
        var config = root.putObject("generationConfig");
        config.putArray("responseModalities").add("TEXT").add("IMAGE");
        var imageConfig = config.putObject("imageConfig");
        imageConfig.put("aspectRatio", ratio);
        if (resolution != null) {
            imageConfig.put("imageSize", resolution);
        }
        return root.toString();
    }

    private static String aspectRatio(int width, int height) {
        for (String ratio : List.of("1:1", "1:4", "4:1", "1:8", "8:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")) {
            String[] pair = ratio.split(":");
            double target = Double.parseDouble(pair[0]) / Double.parseDouble(pair[1]);
            if (Math.abs((double) width / height - target) / target < 0.05) {
                return ratio;
            }
        }
        throw new BusinessException("Gemini 不支持当前图片比例，请选择模型支持的比例");
    }

    static List<String> imageOutputs(String body) {
        try {
            JsonNode root = JSON.readTree(body);
            var images = new ArrayList<String>();
            for (JsonNode candidate : root.path("candidates")) {
                for (JsonNode part : candidate.path("content").path("parts")) {
                    var inline = part.path("inlineData");
                    if (!part.path("thought").asBoolean() && inline.path("mimeType").asText().startsWith("image/")
                            && !inline.path("data").asText().isBlank()) {
                        images.add("data:" + inline.path("mimeType").asText() + ";base64," + inline.path("data").asText());
                    }
                }
            }
            if (images.isEmpty()) {
                throw new BusinessException("未返回图片，请调整提示词或更换生图模型");
            }
            return List.copyOf(images);
        } catch (IOException failure) {
            throw new BusinessException("图片结果读取失败，请重试");
        }
    }
}
