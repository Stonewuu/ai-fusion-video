package com.stonewu.fusion.service.media;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaGenerationOrigin;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaResource;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@Service
@RequiredArgsConstructor
public class GenerationMediaService {
    private final GenerationExecutionContexts executions;
    private final MediaIntakeService intake;
    private final MediaResourceQueries queries;
    private final ProviderMediaFetcher fetcher;
    private final MediaProbe probe;

    @CacheEvict(value = "mediaResource", allEntries = true)
    public String image(ImageTask task, String slot, String source, ApiConfig config) {
        return source(GenerationKind.IMAGE, task.getId(), slot, source, config, MediaKind.IMAGE);
    }

    @CacheEvict(value = "mediaResource", allEntries = true)
    public String video(VideoTask task, String slot, String source, ApiConfig config, MediaKind kind) {
        return source(GenerationKind.VIDEO, task.getId(), slot, source, config, kind);
    }

    @CacheEvict(value = "mediaResource", allEntries = true)
    public String file(GenerationKind kind, long taskId, String slot, Path file,
                       MediaKind expected) throws IOException {
        var execution = executions.get(kind, taskId);
        if (execution.status() != 1) {
            throw new MediaException(409, "GENERATION_FENCED", "生成任务已停止");
        }
        var facts = probe.inspect(file, "application/octet-stream");
        if (facts.kind() != expected) {
            throw MediaException.invalid("生成结果的实际媒体类型不符合输出定义");
        }
        String name = switch (expected) {
            case IMAGE -> "生成图片";
            case VIDEO -> "生成视频";
            case AUDIO -> "生成音频";
            default -> throw MediaException.invalid("生成输出类别未登记");
        };
        var resource = intake.generation(file, name + extension(facts.mimeType()), facts.mimeType(), slot,
                execution.projectId(), execution.context(), new MediaGenerationOrigin(kind, taskId));
        return MediaReference.of(resource.id());
    }

    private String source(GenerationKind kind, long taskId, String slot, String source, ApiConfig config,
                          MediaKind expected) {
        var execution = executions.get(kind, taskId);
        if (execution.status() != 1) {
            throw new MediaException(409, "GENERATION_FENCED", "生成任务已停止");
        }
        if (MediaReference.managed(source)) {
            MediaResource resource = queries.get(MediaReference.id(source), execution.context().actor());
            if (!"READY".equals(resource.state()) || resource.content().kind() != expected) {
                throw MediaException.invalid("生成媒体引用类型或状态无效");
            }
            return source;
        }
        try (var temporary = new TemporaryFile(Files.createTempFile("afv-provider-result-", ".input"))) {
            fetcher.copy(source, config, temporary.path());
            return file(kind, taskId, slot, temporary.path(), expected);
        } catch (IOException failure) {
            var error = new MediaException(502, "GENERATION_MEDIA_FAILED", "生成结果接收失败，请查看任务详情");
            error.initCause(failure);
            throw error;
        }
    }

    private record TemporaryFile(Path path) implements AutoCloseable {
        @Override
        public void close() throws IOException {
            Files.deleteIfExists(path);
        }
    }

    private String extension(String mime) {
        return switch (mime) {
            case "image/png" -> ".png";
            case "image/jpeg" -> ".jpg";
            case "image/webp" -> ".webp";
            case "image/gif" -> ".gif";
            case "video/mp4" -> ".mp4";
            case "video/webm" -> ".webm";
            case "video/quicktime" -> ".mov";
            case "video/x-matroska" -> ".mkv";
            case "audio/mpeg" -> ".mp3";
            case "audio/wav" -> ".wav";
            case "audio/mp4" -> ".m4a";
            default -> throw MediaException.invalid("生成输出格式未登记");
        };
    }
}
