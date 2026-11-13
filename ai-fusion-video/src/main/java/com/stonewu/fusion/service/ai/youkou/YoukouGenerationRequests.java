package com.stonewu.fusion.service.ai.youkou;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class YoukouGenerationRequests {
    private final YoukouReferenceUploads uploads;

    public PreparedGenerationCall image(ImageTask task, AiModel model, ApiConfig api) {
        GenerationHttpClients.requireNoTransaction();
        var contract = YoukouContract.from(model);
        var references = sources(task.getRefImageUrls());
        String endpoint = references.isEmpty() ? "/v1/images/generations" : "/v1/images/edits";
        JsonNode fields = contract.fields(endpoint);
        var body = base(model, task.getPrompt(), fields, contract);
        if (task.getWidth() != null || task.getHeight() != null) {
            if (task.getWidth() == null || task.getHeight() == null) {
                throw new BusinessException("请同时填写图片宽度和高度");
            }
            if (notBlank(task.getResolution()) || notBlank(task.getRatio())) {
                throw new BusinessException("请在图片宽高与分辨率档位中选择一种尺寸方式");
            }
            body.put("size", task.getWidth() + "x" + task.getHeight());
        } else if (YoukouContract.object(model.getConfig(), "模型配置无法读取").path("youkouSizeOnly").asBoolean()) {
            contract.put(body, fields, "size", task.getResolution());
        } else {
            contract.put(body, fields, "resolution", task.getResolution());
            contract.put(body, fields, "aspect_ratio", task.getRatio());
        }
        body.put("n", 1);
        if (contract.supports(fields, "response_format")) {
            var formats = YoukouContract.values(fields.path("response_format").path("values"));
            body.put("response_format", formats.contains("url") ? "url" : "b64_json");
        }
        return finish(model, api, contract, endpoint, fields, body,
                Map.of("input_images", references, "input_videos", List.of(), "input_audios", List.of()));
    }

    public PreparedGenerationCall video(VideoTask task, AiModel model, ApiConfig api) {
        GenerationHttpClients.requireNoTransaction();
        var contract = YoukouContract.from(model);
        String endpoint = "/v1/videos";
        JsonNode fields = contract.fields(endpoint);
        var body = base(model, task.getPrompt(), fields, contract);
        contract.put(body, fields, "resolution", task.getResolution());
        contract.put(body, fields, "aspect_ratio", task.getRatio());
        contract.put(body, fields, "seconds", task.getDuration());
        contract.put(body, fields, "generate_audio", task.getGenerateAudio());
        contract.put(body, fields, "watermark", task.getWatermark());
        contract.put(body, fields, "seed", task.getSeed());
        contract.put(body, fields, "camera_fixed", task.getCameraFixed());
        var images = new ArrayList<String>();
        var modes = YoukouContract.values(contract.capabilities().path("image").path("modes"));
        if (notBlank(task.getFirstFrameImageUrl())) {
            if (!modes.contains("first") && !modes.contains("first_last")) {
                throw new BusinessException("当前模型不支持首帧输入，请改用参考图片");
            }
            images.add(task.getFirstFrameImageUrl());
        }
        if (notBlank(task.getLastFrameImageUrl())) {
            if (!modes.contains("first_last") || images.isEmpty()) {
                throw new BusinessException("当前模型的尾帧需要同时提供首帧");
            }
            images.add(task.getLastFrameImageUrl());
        }
        var references = sources(task.getReferenceImageUrls());
        if (!references.isEmpty() && !modes.contains("reference")) {
            throw new BusinessException("当前模型不支持参考图片，请使用首帧或尾帧输入");
        }
        images.addAll(references);
        return finish(model, api, contract, endpoint, fields, body, Map.of("input_images", images,
                "input_videos", sources(task.getReferenceVideoUrls()), "input_audios", sources(task.getReferenceAudioUrls())));
    }

    private ObjectNode base(AiModel model, String prompt, JsonNode fields, YoukouContract contract) {
        var body = YoukouContract.JSON.createObjectNode().put("model", model.getCode());
        contract.put(body, fields, "prompt", prompt);
        return body;
    }

    private PreparedGenerationCall finish(AiModel model, ApiConfig api, YoukouContract contract, String endpoint,
                                           JsonNode fields, ObjectNode body, Map<String, List<String>> sources) {
        JsonNode options = YoukouContract.object(model.getConfig(), "模型配置无法读取").path("youkouOptions");
        if (!options.isMissingNode()) {
            if (!options.isObject()) {
                throw new BusinessException("贰春模型选项格式无效，请检查模型配置");
            }
            Set<String> reserved = Set.of("model", "prompt", "n", "input_images", "input_videos", "input_audios",
                    "input_reference", "input_image_files", "image");
            options.properties().forEach(entry -> {
                if (reserved.contains(entry.getKey()) || body.has(entry.getKey())
                        && !body.get(entry.getKey()).equals(entry.getValue())) {
                    throw new BusinessException("模型选项与本次生成参数冲突，请检查模型配置");
                }
                body.set(entry.getKey(), entry.getValue().deepCopy());
            });
        }
        if (body.has("size") && (body.has("resolution") || body.has("aspect_ratio"))) {
            throw new BusinessException("图片尺寸与分辨率或比例冲突，请只保留一种尺寸方式");
        }
        contract.defaults(body, fields);
        if (body.has("size")) {
            // size 已包含分辨率和比例时，不能再发送目录为这两个别名填入的默认值。
            for (String resolved : YoukouContract.values(fields.path("size").path("resolves"))) {
                if (options.has(resolved)) {
                    throw new BusinessException("图片尺寸与模型预设参数冲突，请只保留一种尺寸方式");
                }
                body.remove(resolved);
            }
        }
        contract.validateValues(body, fields);
        var client = new YoukouHttpClient(api);
        uploads.attach(client, api, contract, fields, body, sources);
        contract.validate(body, fields);
        return YoukouTasks.prepare(client, contract, endpoint, body);
    }

    private List<String> sources(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        try {
            JsonNode array = YoukouContract.JSON.readTree(text);
            if (!array.isArray() || array.valueStream().anyMatch(item -> !item.isTextual() || item.asText().isBlank())) {
                throw new BusinessException("参考素材列表无效，请重新添加素材");
            }
            return YoukouContract.values(array);
        } catch (IOException invalid) {
            throw new BusinessException("参考素材列表无效，请重新添加素材");
        }
    }

    private boolean notBlank(String text) {
        return text != null && !text.isBlank();
    }
}
