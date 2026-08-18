package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Resolves only explicitly bound output nodes from a completed ComfyUI job. */
@Slf4j
@Component
public class ComfyUiOutputResolver {

    private static final int MAX_OUTPUT_FILES = 64;

    public List<ComfyUiRemoteOutput> resolve(ComfyUiJobResult job,
                                             List<ComfyUiOutputBinding> bindings) {
        List<ComfyUiRemoteOutput> outputs = new ArrayList<>();
        Set<String> uniqueFiles = new HashSet<>();
        for (ComfyUiOutputBinding binding : bindings) {
            JsonNode nodeOutput = job.outputs().get(binding.nodeId());
            if (nodeOutput == null || !nodeOutput.isObject()) {
                continue;
            }
            Iterator<Map.Entry<String, JsonNode>> fields = nodeOutput.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                if (!field.getValue().isArray()) continue;
                for (JsonNode item : field.getValue()) {
                    if (!item.isObject() || !item.path("filename").isTextual()) continue;
                    String actualMediaType = classify(field.getKey(), item);
                    if (!matches(binding.mediaType(), actualMediaType)) {
                        log.debug("ComfyUI output resolve skipped: nodeId={}, outputKey={}, filename={}, expectedMediaType={}, actualMediaType={}",
                                binding.nodeId(), field.getKey(), item.path("filename").asText(),
                                binding.mediaType(), actualMediaType);
                        continue;
                    }
                    String filename = item.path("filename").asText();
                    JsonNode subfolderNode = item.get("subfolder");
                    JsonNode typeNode = item.get("type");
                    if (filename.isBlank() || subfolderNode == null || !subfolderNode.isTextual()
                            || typeNode == null || !typeNode.isTextual()
                            || typeNode.asText().isBlank()) {
                        throw new BusinessException(502,
                                "ComfyUI 输出文件描述必须包含 filename、subfolder、type");
                    }
                    String subfolder = subfolderNode.asText();
                    String type = typeNode.asText();
                    String uniqueKey = filename + "\u0000" + subfolder + "\u0000" + type;
                    if (!uniqueFiles.add(uniqueKey)) continue;
                    outputs.add(new ComfyUiRemoteOutput(
                            binding.nodeId(), binding.mediaType(), binding.role(),
                            filename, subfolder, type));
                    if (outputs.size() > MAX_OUTPUT_FILES) {
                        throw new BusinessException(502, "ComfyUI 输出文件数量超过 64");
                    }
                }
            }
        }
        if (outputs.isEmpty()) {
            throw new BusinessException(502, "ComfyUI 任务已完成，但绑定的输出节点没有返回文件");
        }
        return List.copyOf(outputs);
    }

    private String classify(String outputKey, JsonNode item) {
        String key = outputKey.toLowerCase(Locale.ROOT);
        String format = item.path("format").asText("").toLowerCase(Locale.ROOT);
        String filename = item.path("filename").asText("").toLowerCase(Locale.ROOT);

        // ComfyUI 的输出字段名不可靠：视频常被放在 images 字段下（animated 输出）。
        // 因此文件扩展名优先级最高，其次 MIME format，最后才是输出字段名。
        if (hasExtension(filename, "mp4", "webm", "mov", "mkv", "avi", "m4v")) return "video";
        if (hasExtension(filename, "mp3", "wav", "flac", "m4a", "ogg", "aac")) return "audio";
        if (hasExtension(filename, "png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff")) return "image";

        if (format.startsWith("video/")) return "video";
        if (format.startsWith("audio/")) return "audio";
        if (format.startsWith("image/")) return "image";

        if (key.contains("video") || key.equals("gifs")) return "video";
        if (key.contains("audio")) return "audio";
        if (key.contains("image")) return "image";

        return "file";
    }

    private boolean matches(String expected, String actual) {
        return "file".equals(expected) || expected.equals(actual);
    }

    private boolean hasExtension(String filename, String... extensions) {
        for (String extension : extensions) {
            if (filename.endsWith("." + extension)) return true;
        }
        return false;
    }
}
