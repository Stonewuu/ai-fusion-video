package com.stonewu.fusion.controller.storage;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.ai.AiModelMultimodalCapabilities;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.storage.StorageTypes;
import com.stonewu.fusion.service.system.SystemConfigService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * 通用文件上传 Controller
 */
@Tag(name = "文件上传")
@RestController
@RequestMapping("/api/storage")
@RequiredArgsConstructor
@Slf4j
public class FileUploadController {

    private static final long MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
    private static final Set<String> ALLOWED_IMAGE_TYPES = Set.of(
            "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"
    );
    private static final Map<String, String> ASSISTANT_UPLOAD_TYPES = Map.ofEntries(
            Map.entry("image/png", "png"),
            Map.entry("image/jpeg", "jpg"),
            Map.entry("image/jpg", "jpg"),
            Map.entry("image/webp", "webp"),
            Map.entry("image/gif", "gif"),
            Map.entry("video/mp4", "mp4"),
            Map.entry("video/webm", "webm"),
            Map.entry("video/quicktime", "mov"),
            Map.entry("video/mpeg", "mpeg"),
            Map.entry("audio/mpeg", "mp3"),
            Map.entry("audio/mp4", "m4a"),
            Map.entry("audio/wav", "wav"),
            Map.entry("audio/x-wav", "wav"),
            Map.entry("audio/ogg", "ogg"),
            Map.entry("audio/flac", "flac"),
            Map.entry("audio/aac", "aac"),
            Map.entry("application/pdf", "pdf"),
            Map.entry("text/plain", "txt"),
            Map.entry("text/markdown", "md"),
            Map.entry("text/csv", "csv"),
            Map.entry("application/json", "json")
    );

    private final MediaStorageService mediaStorageService;
    private final StorageConfigService storageConfigService;
    private final SystemConfigService systemConfigService;
    private final AiModelService aiModelService;

    @PostMapping("/upload")
    @Operation(summary = "上传文件")
    public CommonResult<String> upload(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "subDir", defaultValue = "uploads") String subDir) {

        if (file.isEmpty()) {
            throw new BusinessException("文件不能为空");
        }
        if (file.getSize() > MAX_FILE_SIZE) {
            throw new BusinessException("文件大小不能超过 100MB");
        }

        String contentType = file.getContentType();
        if (contentType == null || !ALLOWED_IMAGE_TYPES.contains(contentType.toLowerCase())) {
            throw new BusinessException("仅支持图片格式：PNG, JPEG, WebP, GIF");
        }

        try {
            String ext = getExtension(file.getOriginalFilename());
            String url = mediaStorageService.storeBytes(file.getBytes(), subDir, ext);
            log.info("[FileUpload] 上传成功: size={}KB, url={}", file.getSize() / 1024, url);
            return CommonResult.success(url);
        } catch (IOException e) {
            log.error("[FileUpload] 上传失败", e);
            throw new BusinessException("上传失败: " + e.getMessage());
        }
    }

    @PostMapping("/assistant-upload")
    @Operation(summary = "上传助手多模态输入")
    public CommonResult<String> uploadAssistantInput(
            @RequestParam("file") MultipartFile file,
            @RequestParam("modelId") Long modelId,
            @RequestParam(value = "transport", defaultValue = "url") String transport) {
        if (file.isEmpty()) {
            throw new BusinessException("文件不能为空");
        }
        if (file.getSize() > MAX_FILE_SIZE) {
            throw new BusinessException("文件大小不能超过 100MB");
        }

        String contentType = normalizeContentType(file.getContentType());
        String extension = ASSISTANT_UPLOAD_TYPES.get(contentType);
        if (extension == null) {
            throw new BusinessException("不支持的助手输入格式: " + contentType);
        }
        AiModel model = aiModelService.getById(modelId);
        String normalizedTransport = transport.trim().toLowerCase(Locale.ROOT);
        String inputType = AiModelMultimodalCapabilities.validateUpload(
                model, contentType, normalizedTransport);
        if (AiModelMultimodalCapabilities.TRANSPORT_URL.equals(normalizedTransport)) {
            requirePublicUploadStorage();
        }

        try {
            String storedUrl = mediaStorageService.storeBytes(
                    file.getBytes(), "assistant/" + inputType, extension);
            if (AiModelMultimodalCapabilities.TRANSPORT_BASE64.equals(normalizedTransport)) {
                log.info("[FileUpload] 助手回显资源上传成功: modelId={}, type={}, size={}KB, url={}",
                        modelId, inputType, file.getSize() / 1024, storedUrl);
                return CommonResult.success(storedUrl);
            }
            String publicUrl = systemConfigService.resolvePublicUrl(storedUrl);
            if (publicUrl == null) {
                throw new BusinessException("URL 输入需要公开可访问的对象存储，或在系统设置中配置后端资源公网地址");
            }
            log.info("[FileUpload] 助手输入上传成功: modelId={}, type={}, size={}KB, url={}",
                    modelId, inputType, file.getSize() / 1024, publicUrl);
            return CommonResult.success(publicUrl);
        } catch (IOException e) {
            log.error("[FileUpload] 助手输入上传失败", e);
            throw new BusinessException("上传失败: " + e.getMessage());
        }
    }

    private void requirePublicUploadStorage() {
        StorageConfig config = storageConfigService.getDefaultConfig();
        boolean localStorage = config == null || !StorageTypes.isS3Like(config.getType());
        if (localStorage && StrUtil.isBlank(systemConfigService.getPublicResourceBaseUrl())) {
            throw new BusinessException("本地存储用于 URL 输入时，必须先在系统设置中配置后端资源公网地址");
        }
    }

    private String normalizeContentType(String contentType) {
        if (contentType == null) {
            return "application/octet-stream";
        }
        int separator = contentType.indexOf(';');
        String value = separator >= 0 ? contentType.substring(0, separator) : contentType;
        return value.trim().toLowerCase(Locale.ROOT);
    }

    private String getExtension(String filename) {
        if (filename == null) return "png";
        int dotIndex = filename.lastIndexOf('.');
        return dotIndex >= 0 ? filename.substring(dotIndex + 1) : "png";
    }
}
