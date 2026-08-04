package com.stonewu.fusion.service.ai.comfyui.client;

import cn.hutool.core.util.StrUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import lombok.RequiredArgsConstructor;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/** Typed HTTP client for the ComfyUI v0.30.0 Native API contract. */
@Component
@RequiredArgsConstructor
public class ComfyUiNativeClient {

    public static final String DEFAULT_BASE_URL = "http://localhost:8188";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final int MAX_JSON_RESPONSE_BYTES = 16 * 1024 * 1024;
    private static final long MAX_OUTPUT_BYTES = 1024L * 1024L * 1024L;

    private final ObjectMapper objectMapper;

    private final OkHttpClient baseClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(10, TimeUnit.MINUTES)
            .writeTimeout(2, TimeUnit.MINUTES)
            .build();

    public ComfyUiConnectionResult testConnection(ApiConfig apiConfig) {
        JsonNode systemStats = getJson(apiConfig, "/system_stats", "读取系统信息");
        JsonNode features = getJson(apiConfig, "/features", "读取服务能力");
        String probeId = UUID.randomUUID().toString();
        Request request = request(apiConfig, "/api/jobs/" + probeId).get().build();
        boolean jobsApiSupported;
        try (Response response = client(apiConfig).newCall(request).execute()) {
            String body = readJsonBody(response);
            JsonNode errorResponse = objectMapper.readTree(body);
            jobsApiSupported = response.code() == 404
                    && errorResponse != null
                    && errorResponse.isObject()
                    && "Job not found".equals(errorResponse.path("error").asText());
        } catch (IOException e) {
            throw remoteError("检测 Jobs API", e);
        }
        if (!jobsApiSupported) {
            throw new BusinessException(400,
                    "目标 ComfyUI 不支持 /api/jobs/{id}，请升级到 v0.30.0 或更高版本");
        }
        String version = requiredText(systemStats.path("system"), "comfyui_version",
                "ComfyUI /system_stats 响应缺少 system.comfyui_version");
        return new ComfyUiConnectionResult(true, true, version, systemStats, features);
    }

    public JsonNode getNodeInfo(ApiConfig apiConfig, String nodeClass) {
        if (StrUtil.isBlank(nodeClass)) {
            throw new BusinessException(400, "节点 class_type 不能为空");
        }
        JsonNode response = getJson(
                apiConfig, "/object_info/" + encodePathSegment(nodeClass), "读取节点定义");
        if (!response.isObject()) {
            throw new BusinessException(502, "ComfyUI 节点定义响应必须是对象");
        }
        return response;
    }

    public JsonNode getModels(ApiConfig apiConfig, String folder) {
        if (StrUtil.isBlank(folder)) {
            throw new BusinessException(400, "模型目录不能为空");
        }
        JsonNode response = getJson(apiConfig, "/models/" + encodePathSegment(folder), "读取模型目录");
        if (!response.isArray()) {
            throw new BusinessException(502, "ComfyUI 模型目录响应必须是数组");
        }
        return response;
    }

    public ComfyUiUploadResult uploadImage(ApiConfig apiConfig,
                                           byte[] bytes,
                                           String fileName,
                                           String contentType,
                                           String subfolder) {
        if (bytes == null || bytes.length == 0) {
            throw new BusinessException(400, "上传到 ComfyUI 的图片不能为空");
        }
        String safeFileName = safeFileName(fileName);
        MultipartBody.Builder body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", safeFileName,
                        RequestBody.create(bytes, MediaType.get(normalizeContentType(contentType))))
                .addFormDataPart("type", "input")
                .addFormDataPart("overwrite", "false");
        if (StrUtil.isNotBlank(subfolder)) {
            body.addFormDataPart("subfolder", safeSubfolder(subfolder));
        }
        Request request = request(apiConfig, "/upload/image")
                .post(body.build())
                .build();
        JsonNode response = executeJson(apiConfig, request, "上传输入图片", 200);
        String name = requiredText(response, "name", "ComfyUI 上传响应缺少 name");
        String returnedSubfolder = requiredString(
                response, "subfolder", "ComfyUI 上传响应缺少 subfolder");
        String returnedType = requiredText(response, "type", "ComfyUI 上传响应缺少 type");
        if (!"input".equals(returnedType)) {
            throw new BusinessException(502, "ComfyUI 上传响应 type 必须是 input");
        }
        return new ComfyUiUploadResult(name, returnedSubfolder, returnedType);
    }

    public ComfyUiPromptResponse submitPrompt(ApiConfig apiConfig,
                                               ObjectNode prompt,
                                               String promptId) {
        if (prompt == null || prompt.isEmpty()) {
            throw new BusinessException(400, "ComfyUI prompt 不能为空");
        }
        String canonicalPromptId = validatePromptId(promptId);
        ObjectNode payload = objectMapper.createObjectNode();
        payload.set("prompt", prompt);
        payload.put("prompt_id", canonicalPromptId);
        Request request = request(apiConfig, "/prompt")
                .post(RequestBody.create(writeJson(payload), JSON))
                .build();
        JsonNode response = executeJson(apiConfig, request, "提交工作流", 200);
        String returnedPromptId = requiredText(response, "prompt_id", "ComfyUI 提交响应缺少 prompt_id");
        if (!canonicalPromptId.equals(returnedPromptId)) {
            throw new BusinessException(502, "ComfyUI 返回的 prompt_id 与请求不一致");
        }
        JsonNode numberNode = response.get("number");
        if (numberNode == null || !numberNode.isNumber()) {
            throw new BusinessException(502, "ComfyUI 提交响应缺少数值 number");
        }
        JsonNode nodeErrors = response.get("node_errors");
        if (nodeErrors == null || !nodeErrors.isObject()) {
            throw new BusinessException(502, "ComfyUI 提交响应缺少对象 node_errors");
        }
        return new ComfyUiPromptResponse(returnedPromptId, numberNode.asDouble(), nodeErrors);
    }

    public ComfyUiJobResult getJob(ApiConfig apiConfig, String promptId) {
        String canonicalPromptId = validatePromptId(promptId);
        JsonNode response = getJson(apiConfig, "/api/jobs/" + canonicalPromptId, "查询工作流任务");
        String id = requiredText(response, "id", "ComfyUI Job 响应缺少 id");
        String status = requiredText(response, "status", "ComfyUI Job 响应缺少 status");
        if (!id.equals(canonicalPromptId)) {
            throw new BusinessException(502, "ComfyUI Job ID 与请求不一致");
        }
        if (!isOfficialJobStatus(status)) {
            throw new BusinessException(502, "ComfyUI Job 返回未知状态: " + status);
        }
        JsonNode outputsNode = response.get("outputs");
        ObjectNode outputs;
        if (outputsNode instanceof ObjectNode objectNode) {
            outputs = objectNode;
        } else if ("pending".equals(status) || "in_progress".equals(status)) {
            outputs = objectMapper.createObjectNode();
        } else {
            throw new BusinessException(502, "ComfyUI 终态 Job 响应缺少对象 outputs");
        }
        JsonNode error = response.get("execution_error");
        return new ComfyUiJobResult(id, status, outputs, error, response);
    }

    public boolean cancelJob(ApiConfig apiConfig, String promptId) {
        Request request = request(apiConfig, "/api/jobs/" + validatePromptId(promptId) + "/cancel")
                .post(RequestBody.create("{}", JSON))
                .build();
        JsonNode response = executeJson(apiConfig, request, "取消工作流任务", 200);
        JsonNode cancelled = response.get("cancelled");
        if (cancelled == null || !cancelled.isBoolean()) {
            throw new BusinessException(502, "ComfyUI 取消响应缺少布尔值 cancelled");
        }
        return cancelled.asBoolean();
    }

    public ComfyUiDownloadedFile downloadOutput(ApiConfig apiConfig,
                                                 String filename,
                                                 String subfolder,
                                                 String type) {
        if (StrUtil.isBlank(filename)) {
            throw new BusinessException(400, "ComfyUI 输出缺少 filename");
        }
        if (subfolder == null) {
            throw new BusinessException(400, "ComfyUI 输出缺少 subfolder");
        }
        if (StrUtil.isBlank(type)) {
            throw new BusinessException(400, "ComfyUI 输出缺少 type");
        }
        String query = "/view?filename=" + encodeQuery(filename)
                + "&subfolder=" + encodeQuery(subfolder)
                + "&type=" + encodeQuery(type);
        Request request = request(apiConfig, query).get().build();
        Path tempFile = null;
        try (Response response = client(apiConfig).newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw httpError("下载工作流输出", response, readJsonBody(response));
            }
            ResponseBody body = response.body();
            if (body == null) {
                throw new BusinessException(502, "ComfyUI 输出响应为空");
            }
            long declaredLength = body.contentLength();
            if (declaredLength > MAX_OUTPUT_BYTES) {
                throw new BusinessException(502, "ComfyUI 输出超过 1GB 限制");
            }
            String extension = extensionOf(filename);
            tempFile = Files.createTempFile("afv-comfyui-output-", "." + extension);
            try (InputStream input = body.byteStream()) {
                copyBounded(input, tempFile, MAX_OUTPUT_BYTES);
            }
            long actualSize = Files.size(tempFile);
            if (actualSize <= 0) {
                throw new BusinessException(502, "ComfyUI 输出大小无效");
            }
            String responseType = body.contentType() != null
                    ? body.contentType().toString() : "application/octet-stream";
            return new ComfyUiDownloadedFile(tempFile, responseType, extension, actualSize);
        } catch (IOException e) {
            if (tempFile != null) {
                try {
                    Files.deleteIfExists(tempFile);
                } catch (IOException ignored) {
                    // Preserve the original transfer error.
                }
            }
            throw remoteError("下载工作流输出", e);
        } catch (RuntimeException e) {
            if (tempFile != null) {
                try {
                    Files.deleteIfExists(tempFile);
                } catch (IOException ignored) {
                    // Preserve the original validation error.
                }
            }
            throw e;
        }
    }

    public String resolveBaseUrl(ApiConfig apiConfig) {
        if (apiConfig == null) {
            throw new BusinessException(400, "ComfyUI API 配置不能为空");
        }
        String baseUrl = StrUtil.blankToDefault(apiConfig.getApiUrl(), DEFAULT_BASE_URL).trim();
        if (!baseUrl.startsWith("http://") && !baseUrl.startsWith("https://")) {
            throw new BusinessException(400, "ComfyUI 地址必须使用 HTTP 或 HTTPS");
        }
        return baseUrl.replaceAll("/+$", "");
    }

    private JsonNode getJson(ApiConfig apiConfig, String path, String operation) {
        Request request = request(apiConfig, path).get().build();
        return executeJson(apiConfig, request, operation, 200);
    }

    private JsonNode executeJson(ApiConfig apiConfig,
                                 Request request,
                                 String operation,
                                 int expectedStatus) {
        try (Response response = client(apiConfig).newCall(request).execute()) {
            String body = readJsonBody(response);
            if (response.code() != expectedStatus) {
                throw httpError(operation, response, body);
            }
            if (body.isBlank()) {
                throw new BusinessException(502, "ComfyUI " + operation + "响应为空");
            }
            return objectMapper.readTree(body);
        } catch (JsonProcessingException e) {
            throw new BusinessException(502, "ComfyUI " + operation + "返回了无效 JSON");
        } catch (IOException e) {
            throw remoteError(operation, e);
        }
    }

    private Request.Builder request(ApiConfig apiConfig, String path) {
        String url = path.startsWith("http://") || path.startsWith("https://")
                ? path : resolveBaseUrl(apiConfig) + (path.startsWith("/") ? path : "/" + path);
        Request.Builder builder = new Request.Builder().url(url).header("Accept", "application/json");
        if (apiConfig != null && StrUtil.isNotBlank(apiConfig.getApiKey())) {
            builder.header("Authorization", "Bearer " + apiConfig.getApiKey().trim());
        }
        return builder;
    }

    private OkHttpClient client(ApiConfig apiConfig) {
        return AiProxySupport.okHttpClient(baseClient, apiConfig);
    }

    private String readJsonBody(Response response) throws IOException {
        ResponseBody body = response.body();
        if (body == null) return "";
        long length = body.contentLength();
        if (length > MAX_JSON_RESPONSE_BYTES) {
            throw new BusinessException(502, "ComfyUI JSON 响应超过 16MB 限制");
        }
        return new String(readJsonBounded(body.byteStream(), MAX_JSON_RESPONSE_BYTES),
                StandardCharsets.UTF_8);
    }

    private BusinessException httpError(String operation, Response response, String body) {
        String detail = body == null || body.isBlank() ? "" : ": " + truncate(body, 2000);
        return new BusinessException(502,
                "ComfyUI " + operation + "失败，HTTP " + response.code() + detail);
    }

    private BusinessException remoteError(String operation, IOException error) {
        return new BusinessException(502,
                "ComfyUI " + operation + "请求异常: " + StrUtil.blankToDefault(error.getMessage(), "I/O error"));
    }

    private String requiredText(JsonNode node, String field, String message) {
        String value = requiredString(node, field, message);
        if (value.isBlank()) {
            throw new BusinessException(502, message);
        }
        return value.trim();
    }

    private String requiredString(JsonNode node, String field, String message) {
        JsonNode value = node == null ? null : node.get(field);
        if (value == null || !value.isTextual()) {
            throw new BusinessException(502, message);
        }
        return value.asText();
    }

    private String writeJson(JsonNode node) {
        try {
            return objectMapper.writeValueAsString(node);
        } catch (JsonProcessingException e) {
            throw new BusinessException(500, "序列化 ComfyUI 请求失败");
        }
    }

    private String validatePromptId(String promptId) {
        try {
            UUID parsed = UUID.fromString(promptId);
            String canonical = parsed.toString();
            if (!canonical.equals(promptId)) {
                throw new IllegalArgumentException("non-canonical UUID");
            }
            return canonical;
        } catch (RuntimeException e) {
            throw new BusinessException(400, "ComfyUI prompt_id 必须是 canonical UUID");
        }
    }

    static long copyBounded(InputStream input, Path target, long maxBytes) throws IOException {
        byte[] buffer = new byte[8192];
        long total = 0L;
        try (OutputStream output = Files.newOutputStream(target)) {
            int read;
            while ((read = input.read(buffer)) != -1) {
                if (total > maxBytes - read) {
                    throw new BusinessException(502, "ComfyUI 输出超过 1GB 限制");
                }
                output.write(buffer, 0, read);
                total += read;
            }
        }
        return total;
    }

    static byte[] readJsonBounded(InputStream input, int maxBytes) throws IOException {
        byte[] buffer = new byte[8192];
        int total = 0;
        try (ByteArrayOutputStream output = new ByteArrayOutputStream(Math.min(maxBytes, 8192))) {
            int read;
            while ((read = input.read(buffer)) != -1) {
                if (total > maxBytes - read) {
                    throw new BusinessException(502, "ComfyUI JSON 响应超过 16MB 限制");
                }
                output.write(buffer, 0, read);
                total += read;
            }
            return output.toByteArray();
        }
    }

    private boolean isOfficialJobStatus(String status) {
        return switch (status) {
            case "pending", "in_progress", "completed", "failed", "cancelled" -> true;
            default -> false;
        };
    }

    private String safeFileName(String value) {
        if (StrUtil.isBlank(value)) {
            throw new BusinessException(400, "上传文件名不能为空");
        }
        String fileName = value.trim();
        fileName = fileName.replace('\\', '_').replace('/', '_');
        if (fileName.contains("..") || fileName.isBlank()) {
            throw new BusinessException(400, "上传文件名不安全");
        }
        return fileName;
    }

    private String safeSubfolder(String value) {
        String subfolder = value.trim().replace('\\', '/');
        if (subfolder.startsWith("/") || subfolder.contains("..")) {
            throw new BusinessException(400, "ComfyUI 上传子目录不安全");
        }
        return subfolder;
    }

    private String normalizeContentType(String value) {
        if (StrUtil.isBlank(value)) {
            throw new BusinessException(400, "ComfyUI 上传图片缺少 Content-Type");
        }
        String contentType = value.split(";", 2)[0].trim();
        if (!contentType.startsWith("image/")) {
            throw new BusinessException(400, "ComfyUI /upload/image 只接受图片");
        }
        return contentType;
    }

    private String extensionOf(String filename) {
        int separator = filename.lastIndexOf('.');
        String extension = separator >= 0 ? filename.substring(separator + 1) : "bin";
        extension = extension.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", "");
        return extension.isBlank() || extension.length() > 10 ? "bin" : extension;
    }

    private String encodePathSegment(String value) {
        return encodeQuery(value).replace("+", "%20");
    }

    private String encodeQuery(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }

    private String truncate(String value, int maxLength) {
        return value.length() <= maxLength ? value : value.substring(0, maxLength) + "…";
    }
}
