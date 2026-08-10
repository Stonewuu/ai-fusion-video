package com.stonewu.fusion.service.ai;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.AiMultimodalInputVO;
import com.stonewu.fusion.entity.ai.AiModel;

import java.net.URI;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Canonical validation for model multimodal input capabilities and message inputs. */
public final class AiModelMultimodalCapabilities {

    public static final String TYPE_IMAGE = "image";
    public static final String TYPE_VIDEO = "video";
    public static final String TYPE_AUDIO = "audio";
    public static final String TYPE_FILE = "file";
    public static final String TRANSPORT_URL = "url";
    public static final String TRANSPORT_BASE64 = "base64";

    private static final int MAX_INPUT_COUNT = 8;
    private static final int MAX_BASE64_INPUT_BYTES = 10 * 1024 * 1024;
    private static final int MAX_TOTAL_BASE64_BYTES = 20 * 1024 * 1024;
    private static final Set<String> INPUT_TYPES = Set.of(
            TYPE_IMAGE, TYPE_VIDEO, TYPE_AUDIO, TYPE_FILE);
    private static final Set<String> TRANSPORTS = Set.of(TRANSPORT_URL, TRANSPORT_BASE64);

    private AiModelMultimodalCapabilities() {
    }

    public static void normalizeModel(AiModel model) {
        if (model == null) {
            return;
        }
        if (model.getMultimodalInputTypes() == null || model.getMultimodalInputTransports() == null) {
            throw new BusinessException(400, "多模态输入类型和传输方式必须使用数组与对象结构");
        }

        LinkedHashSet<String> normalizedTypes = new LinkedHashSet<>();
        for (String value : model.getMultimodalInputTypes()) {
            String type = normalize(value);
            if (!INPUT_TYPES.contains(type)) {
                throw new BusinessException(400, "不支持的多模态输入类型: " + value);
            }
            normalizedTypes.add(type);
        }
        if (!Integer.valueOf(1).equals(model.getModelType()) && !normalizedTypes.isEmpty()) {
            throw new BusinessException(400, "只有对话模型可以配置多模态输入能力");
        }

        Map<String, List<String>> normalizedTransports = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> entry : model.getMultimodalInputTransports().entrySet()) {
            String type = normalize(entry.getKey());
            if (!normalizedTypes.contains(type)) {
                throw new BusinessException(400, "传输方式包含未启用的输入类型: " + entry.getKey());
            }
            if (entry.getValue() == null) {
                throw new BusinessException(400, "多模态传输方式必须使用数组结构: " + type);
            }
            LinkedHashSet<String> transports = new LinkedHashSet<>();
            for (String value : entry.getValue()) {
                String transport = normalize(value);
                if (!TRANSPORTS.contains(transport)) {
                    throw new BusinessException(400, "不支持的多模态传输方式: " + value);
                }
                transports.add(transport);
            }
            if (transports.isEmpty()) {
                throw new BusinessException(400, "已启用的输入类型至少需要一种传输方式: " + type);
            }
            normalizedTransports.put(type, List.copyOf(transports));
        }
        if (!normalizedTransports.keySet().equals(normalizedTypes)) {
            throw new BusinessException(400, "每个多模态输入类型都必须配置且只能配置自己的传输方式");
        }

        model.setMultimodalInputTypes(List.copyOf(normalizedTypes));
        model.setMultimodalInputTransports(Map.copyOf(normalizedTransports));
        model.setSupportVision(normalizedTypes.contains(TYPE_IMAGE));
    }

    public static void validateInputs(AiModel model, List<AiMultimodalInputVO> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        requireConfiguredModel(model);
        if (inputs.size() > MAX_INPUT_COUNT) {
            throw new BusinessException(400, "单次最多添加 " + MAX_INPUT_COUNT + " 个多模态输入");
        }

        long totalBase64Bytes = 0;
        Set<String> ids = new LinkedHashSet<>();
        for (AiMultimodalInputVO input : inputs) {
            if (input == null) {
                throw new BusinessException(400, "多模态输入不能为空");
            }
            String id = requireText(input.getId(), "多模态输入缺少 ID");
            if (!ids.add(id)) {
                throw new BusinessException(400, "多模态输入 ID 重复: " + id);
            }
            String type = normalize(input.getInputType());
            String mimeType = normalizeMimeType(input.getMimeType());
            String transport = normalize(input.getTransport());
            if (!type.equals(inputTypeForMime(mimeType))) {
                throw new BusinessException(400, "多模态输入类型与 MIME 类型不一致: " + mimeType);
            }
            requireSupported(model, type, transport);

            input.setId(id);
            input.setName(StrUtil.blankToDefault(input.getName(), "attachment"));
            input.setInputType(type);
            input.setMimeType(mimeType);
            input.setTransport(transport);
            if (TRANSPORT_URL.equals(transport)) {
                validateUrlInput(input);
            } else {
                int decodedBytes = validateBase64Input(input);
                totalBase64Bytes += decodedBytes;
                if (totalBase64Bytes > MAX_TOTAL_BASE64_BYTES) {
                    throw new BusinessException(400, "Base64 多模态输入总大小不能超过 20MB");
                }
            }
        }
    }

    public static String validateUpload(AiModel model, String mimeType, String transport) {
        requireConfiguredModel(model);
        String normalizedMimeType = normalizeMimeType(mimeType);
        String type = inputTypeForMime(normalizedMimeType);
        requireSupported(model, type, normalize(transport));
        return type;
    }

    public static String inputTypeForMime(String mimeType) {
        String value = normalizeMimeType(mimeType);
        if (value.startsWith("image/")) {
            return TYPE_IMAGE;
        }
        if (value.startsWith("video/")) {
            return TYPE_VIDEO;
        }
        if (value.startsWith("audio/")) {
            return TYPE_AUDIO;
        }
        return TYPE_FILE;
    }

    private static void requireConfiguredModel(AiModel model) {
        if (model == null) {
            throw new BusinessException(404, "AI 模型不存在");
        }
        if (!Integer.valueOf(1).equals(model.getModelType()) || !Integer.valueOf(1).equals(model.getStatus())) {
            throw new BusinessException(400, "请选择已启用的对话模型");
        }
        if (model.getMultimodalInputTypes() == null || model.getMultimodalInputTransports() == null) {
            throw new BusinessException(500, "模型多模态能力配置的数据结构无效");
        }
    }

    private static void requireSupported(AiModel model, String type, String transport) {
        List<String> transports = model.getMultimodalInputTransports().get(type);
        if (!model.getMultimodalInputTypes().contains(type)
                || transports == null
                || !transports.contains(transport)) {
            throw new BusinessException(400, "当前模型不支持 " + type + " 的 " + transport + " 输入");
        }
    }

    private static void validateUrlInput(AiMultimodalInputVO input) {
        String url = requireText(input.getUrl(), "URL 多模态输入缺少地址");
        if (url.length() > 4096) {
            throw new BusinessException(400, "多模态输入 URL 过长");
        }
        try {
            URI uri = URI.create(url);
            if (!("http".equalsIgnoreCase(uri.getScheme()) || "https".equalsIgnoreCase(uri.getScheme()))
                    || StrUtil.isBlank(uri.getHost())) {
                throw new IllegalArgumentException("unsupported URL");
            }
        } catch (IllegalArgumentException e) {
            throw new BusinessException(400, "多模态输入必须使用有效的 HTTP/HTTPS URL");
        }
        if (StrUtil.isNotBlank(input.getData())) {
            throw new BusinessException(400, "URL 多模态输入不能同时携带 Base64 数据");
        }
        input.setUrl(url);
        input.setData(null);
    }

    private static int validateBase64Input(AiMultimodalInputVO input) {
        String data = requireText(input.getData(), "Base64 多模态输入缺少数据");
        if (StrUtil.isNotBlank(input.getUrl())) {
            throw new BusinessException(400, "Base64 多模态输入不能同时携带 URL");
        }
        byte[] decoded;
        try {
            decoded = Base64.getDecoder().decode(data);
        } catch (IllegalArgumentException e) {
            throw new BusinessException(400, "多模态输入不是有效的 Base64 数据");
        }
        if (decoded.length == 0 || decoded.length > MAX_BASE64_INPUT_BYTES) {
            throw new BusinessException(400, "单个 Base64 多模态输入大小必须在 1B 到 10MB 之间");
        }
        if (input.getSize() != null && input.getSize() != decoded.length) {
            throw new BusinessException(400, "多模态输入大小与 Base64 数据不一致");
        }
        input.setData(data);
        input.setUrl(null);
        input.setSize((long) decoded.length);
        return decoded.length;
    }

    private static String normalizeMimeType(String value) {
        String mimeType = normalize(value);
        if (StrUtil.isBlank(mimeType) || !mimeType.contains("/")) {
            throw new BusinessException(400, "多模态输入缺少有效的 MIME 类型");
        }
        int separator = mimeType.indexOf(';');
        return separator >= 0 ? mimeType.substring(0, separator).trim() : mimeType;
    }

    private static String requireText(String value, String message) {
        if (StrUtil.isBlank(value)) {
            throw new BusinessException(400, message);
        }
        return value.trim();
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }
}
