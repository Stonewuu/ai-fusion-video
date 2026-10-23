package com.stonewu.fusion.controller.storage;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.ai.AiModelMultimodalCapabilities;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.system.SystemConfigService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Locale;
import java.util.Set;

/**
 * 现有上传入口沿用返回字符串的契约，产物统一登记为受权限控制的媒体引用。
 */
@Tag(name = "文件上传")
@RestController
@RequestMapping("/api/storage")
@RequiredArgsConstructor
public class FileUploadController {
    private static final Set<String> IMAGE_TYPES = Set.of("image/png", "image/jpeg", "image/jpg", "image/webp",
            "image/gif");
    private final MediaIntakeService intake;
    private final CanvasActorResolver actors;
    private final SystemConfigService system;
    private final AiModelService models;

    @PostMapping("/upload")
    @Operation(summary = "上传图片")
    public CommonResult<String> upload(@RequestParam("file") MultipartFile file,
                                       @RequestParam String requestId) throws IOException {
        validateFile(file);
        if (!IMAGE_TYPES.contains(mime(file))) {
            throw MediaException.invalid("仅支持图片格式：PNG, JPEG, WebP, GIF");
        }
        return CommonResult.success(receive(file, requestId, Set.of(MediaKind.IMAGE)));
    }

    @PostMapping("/assistant-upload")
    @Operation(summary = "上传助手多模态输入")
    public CommonResult<String> uploadAssistantInput(@RequestParam("file") MultipartFile file,
                                                     @RequestParam Long modelId, @RequestParam String requestId,
                                                     @RequestParam(defaultValue = "url") String transport) throws IOException {
        validateFile(file);
        String mode = transport.trim().toLowerCase(Locale.ROOT);
        String kind = AiModelMultimodalCapabilities.validateUpload(models.getById(modelId), mime(file), mode);
        if (AiModelMultimodalCapabilities.TRANSPORT_URL.equals(mode)
                && (system.getPublicResourceBaseUrl() == null || system.getPublicResourceBaseUrl().isBlank())) {
            throw MediaException.invalid("URL 输入需要先配置后端资源公网地址");
        }
        return CommonResult.success(receive(file, requestId, switch (kind) {
            case "image" -> Set.of(MediaKind.IMAGE);
            case "video" -> Set.of(MediaKind.VIDEO);
            case "audio" -> Set.of(MediaKind.AUDIO);
            case "file" -> Set.of(MediaKind.TEXT, MediaKind.DOCUMENT);
            default -> throw MediaException.invalid("模型未登记此媒体类型");
        }));
    }

    private String receive(MultipartFile file, String requestId, Set<MediaKind> kinds) throws IOException {
        var context = ResourceCommandContext.user(actors.current(), null);
        try (var stream = file.getInputStream()) {
            return MediaReference.of(
                    intake.uploadChecked(stream, file.getOriginalFilename(), mime(file), requestId, null, context,
                            kinds).id());
        }
    }

    private void validateFile(MultipartFile file) {
        if (file.isEmpty()) {
            throw MediaException.invalid("文件不能为空");
        }
        if (file.getSize() > MediaIntakeService.MAX_UPLOAD_BYTES) {
            throw new MediaException(413, "MEDIA_TOO_LARGE", "文件大小不能超过 100MB");
        }
    }

    private String mime(MultipartFile file) {
        String value = file.getContentType();
        if (value == null) {
            throw MediaException.invalid("必须声明文件类型");
        }
        return value.split(";", 2)[0].trim().toLowerCase(Locale.ROOT);
    }
}
