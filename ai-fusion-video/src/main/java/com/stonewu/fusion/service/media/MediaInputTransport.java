package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextData.Claim;
import com.stonewu.fusion.service.generation.job.RemoteGenerationJob;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.system.SystemConfigService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

/**
 * 业务中保存资源标识，仅在实际模型调用边界转换为声明支持的传输形式。
 */
@Component
@RequiredArgsConstructor
public class MediaInputTransport {
    private final MediaResourceQueries resources;
    private final MediaContentReader reader;
    private final MediaAccessTickets tickets;
    private final SystemConfigService system;

    public String dataUri(String reference, CanvasActor actor, MediaKind kind, long limit) {
        String id = MediaReference.id(reference);
        var media = resources.get(id, actor);
        if (media.content().kind() != kind) {
            throw MediaException.invalid("模型输入的媒体类型不符");
        }
        try (var file = new TemporaryFile(Files.createTempFile("afv-model-input-", ".input"))) {
            reader.copy(id, actor, file.path(), limit);
            return "data:" + media.content().mimeType() + ";base64," + Base64.getEncoder()
                    .encodeToString(Files.readAllBytes(file.path()));
        } catch (IOException failure) {
            var error = new MediaException(502, "MEDIA_INPUT_FAILED", "模型输入媒体读取失败");
            error.initCause(failure);
            throw error;
        }
    }

    public String generationUrl(String reference, CanvasActor actor, MediaKind kind, RemoteGenerationJob.Claim claim) {
        String id = providerResource(reference, actor, kind);
        return publicBase() + tickets.generation(id, actor, claim).url();
    }

    public String agentUrl(String reference, CanvasActor actor, MediaKind kind, AgentRunContext run) {
        String id = providerResource(reference, actor, kind);
        return publicBase() + tickets.agent(id, actor, run).url();
    }

    public String canvasTextUrl(String reference, CanvasActor actor, Claim claim) {
        String id = providerResource(reference, actor, MediaKind.IMAGE);
        return publicBase() + tickets.canvasText(id, actor, claim).url();
    }

    private String providerResource(String reference, CanvasActor actor, MediaKind kind) {
        String id = MediaReference.id(reference);
        var media = resources.get(id, actor);
        if (media.content().kind() != kind) {
            throw MediaException.invalid("模型输入的媒体类型不符");
        }
        return id;
    }

    private String publicBase() {
        String base = system.getPublicResourceBaseUrl();
        if (base == null || base.isBlank()) {
            throw MediaException.invalid("此模型需要 URL 输入，请先配置后端资源公网地址");
        }
        return base;
    }

    private record TemporaryFile(Path path) implements AutoCloseable {
        @Override
        public void close() throws IOException {
            Files.deleteIfExists(path);
        }
    }
}
