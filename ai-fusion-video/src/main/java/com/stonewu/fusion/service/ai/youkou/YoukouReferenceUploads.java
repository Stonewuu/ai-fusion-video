package com.stonewu.fusion.service.ai.youkou;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import com.stonewu.fusion.service.media.MediaProbe;
import com.stonewu.fusion.service.media.ProviderMediaFetcher;
import com.stonewu.fusion.service.media.model.MediaFacts;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;

/** 使用平台公布的单文件上传接口，输入文件不携带供应商密钥下载。 */
@Component
@RequiredArgsConstructor
public class YoukouReferenceUploads {
    private final ProviderMediaFetcher fetcher;
    private final MediaProbe probe;

    public void attach(YoukouHttpClient client, ApiConfig api, YoukouContract contract, JsonNode fields,
                       ObjectNode body, Map<String, List<String>> sources) {
        GenerationHttpClients.requireNoTransaction();
        int total = sources.values().stream().mapToInt(List::size).sum();
        long kinds = sources.values().stream().filter(values -> !values.isEmpty()).count();
        JsonNode mix = contract.capabilities().path("mix");
        if (kinds > 1 && (!"supported".equals(mix.path("status").asText())
                || !mix.path("allow_mixed").asBoolean())) {
            throw new BusinessException("当前模型不支持混合使用不同类型的参考素材");
        }
        if (mix.path("max_total").asInt() > 0 && total > mix.get("max_total").asInt()) {
            throw new BusinessException("参考素材总数超过模型限制");
        }
        if (contract.capabilities().path("constraints").path("audio_requires_image_or_video").asBoolean()
                && !sources.get("input_audios").isEmpty() && sources.get("input_images").isEmpty()
                && sources.get("input_videos").isEmpty()) {
            throw new BusinessException("使用参考音频时，请同时添加参考图片或视频");
        }
        for (var entry : sources.entrySet()) {
            if (entry.getValue().isEmpty()) {
                continue;
            }
            JsonNode field = fields.path(entry.getKey());
            if (!contract.supports(fields, entry.getKey()) || !"media_id".equals(field.path("format").asText())
                    || !field.path("max_count").isInt() || entry.getValue().size() > field.get("max_count").asInt()
                    || entry.getValue().size() < field.path("min_count").asInt()) {
                throw YoukouContract.invalid(entry.getKey(), "数量或输入方式不符合模型要求");
            }
        }
        for (var entry : sources.entrySet()) {
            if (entry.getValue().isEmpty()) {
                continue;
            }
            JsonNode field = fields.path(entry.getKey());
            var references = body.putArray(entry.getKey());
            double totalSeconds = 0;
            for (String source : entry.getValue()) {
                try {
                    var path = Files.createTempFile("afv-youkou-input-", ".media");
                    try {
                        fetcher.copyPublic(source, api, path);
                        MediaFacts facts = probe.inspect(path, "application/octet-stream");
                        validate(facts, field, entry.getKey());
                        totalSeconds += facts.durationMs() == null ? 0 : facts.durationMs() / 1000.0;
                        double maxTotal = field.path("max_total_duration_seconds").asDouble();
                        if (maxTotal > 0 && totalSeconds > maxTotal) {
                            throw YoukouContract.invalid(entry.getKey(), "总时长超过模型限制");
                        }
                        if (entry.getKey().equals("input_videos")) {
                            double combined = contract.capabilities().path("constraints")
                                    .path("max_video_input_output_duration_seconds").asDouble();
                            if (combined > 0 && totalSeconds + body.path("seconds").asDouble() > combined) {
                                throw new BusinessException("参考视频与生成视频的总时长超过模型限制");
                            }
                        }
                        references.addObject().put("media_id", client.upload(path, facts.mimeType()));
                    } finally {
                        Files.deleteIfExists(path);
                    }
                } catch (IOException failure) {
                    throw new UncheckedIOException("参考素材读取失败，请重新添加素材", failure);
                }
            }
        }
    }

    private void validate(MediaFacts facts, JsonNode field, String name) {
        if (!YoukouContract.values(field.path("allowed_mime_types")).contains(facts.mimeType())) {
            throw YoukouContract.invalid(name, "文件格式不受支持");
        }
        if (field.path("max_bytes").asLong() > 0 && facts.byteSize() > field.get("max_bytes").asLong()) {
            throw YoukouContract.invalid(name, "文件大小超过模型限制");
        }
        if (facts.width() != null && facts.height() != null) {
            double ratio = (double) facts.width() / facts.height();
            if (field.path("min_dimension").asInt() > Math.min(facts.width(), facts.height())
                    || field.path("max_dimension").asInt() > 0
                    && Math.max(facts.width(), facts.height()) > field.get("max_dimension").asInt()
                    || field.path("min_aspect_ratio").asDouble() > ratio
                    || field.path("max_aspect_ratio").asDouble() > 0
                    && ratio > field.get("max_aspect_ratio").asDouble()) {
                throw YoukouContract.invalid(name, "画面尺寸或比例不符合模型要求");
            }
        }
        double seconds = facts.durationMs() == null ? 0 : facts.durationMs() / 1000.0;
        if (seconds < field.path("min_duration_seconds").asDouble()
                || field.path("max_duration_seconds").asDouble() > 0
                && seconds > field.get("max_duration_seconds").asDouble()) {
            throw YoukouContract.invalid(name, "时长不符合模型要求");
        }
    }
}
