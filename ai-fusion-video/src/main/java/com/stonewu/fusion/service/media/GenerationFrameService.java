package com.stonewu.fusion.service.media;

import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.video.VideoFrameExtractor;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@Service
@RequiredArgsConstructor
public class GenerationFrameService {
    private final GenerationExecutionContexts executions;
    private final MediaContentReader content;
    private final GenerationMediaService output;
    private final VideoFrameExtractor extractor;

    @CacheEvict(value = "mediaResource", allEntries = true)
    public Frames extract(VideoTask task, VideoItem item, boolean first, boolean last) {
        if (!first && !last) {
            return new Frames(null, null);
        }
        var execution = executions.running(GenerationKind.VIDEO, task.getId());
        try (var directory = new TemporaryDirectory(Files.createTempDirectory("afv-generation-frames-"))) {
            Path input = directory.path().resolve("input.video");
            content.copy(MediaReference.id(item.getVideoUrl()), execution.context().actor(), input,
                    MediaIntakeService.MAX_RESULT_BYTES);
            var frames = extractor.extract(input, directory.path(), first, last);
            return new Frames(
                    first ? output.file(GenerationKind.VIDEO, task.getId(), "item-" + item.getId() + "-firstFrame",
                            frames.firstFrame(), MediaKind.IMAGE) : null,
                    last ? output.file(GenerationKind.VIDEO, task.getId(), "item-" + item.getId() + "-lastFrame",
                            frames.lastFrame(), MediaKind.IMAGE) : null);
        } catch (IOException failure) {
            var error = new MediaException(502, "VIDEO_FRAME_EXTRACTION_FAILED", "视频已保留，首尾帧提取未完成");
            error.initCause(failure);
            throw error;
        }
    }

    public record Frames(String firstFrameUrl, String lastFrameUrl) {
    }

    private record TemporaryDirectory(Path path) implements AutoCloseable {
        @Override
        public void close() throws IOException {
            for (String name : new String[]{"first.jpg", "last.jpg", "input.video"}) {
                Files.deleteIfExists(path.resolve(name));
            }
            Files.delete(path);
        }
    }
}
