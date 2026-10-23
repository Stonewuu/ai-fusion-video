package com.stonewu.fusion.service.media.preview;

import com.stonewu.fusion.mapper.media.MediaPreviewJobMapper;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaResource;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

import static com.stonewu.fusion.service.media.preview.MediaPreviewData.*;

@Component
@RequiredArgsConstructor
public class MediaPreviewQueries {
    private final MediaPreviewJobMapper previews;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;
    private final MediaJobRepository jobs;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View get(String id, CanvasActor actor) {
        var source = source(id, actor, false);
        var current = previews.current(id, source.version(), PROFILE, false);
        if (current == null) {
            return new View(id, source.version(), PROFILE, null, "ABSENT", null, null, null, null, sample(source),
                    false);
        }
        return view(current, actor);
    }

    public MediaResource source(String id, CanvasActor actor, boolean lock) {
        var source = resources.get(id, lock).resource();
        access.ready(source, actor);
        var facts = source.content();
        if (!List.of(MediaKind.IMAGE, MediaKind.VIDEO).contains(facts.kind()) || resources.previewParent(id)
                .isPresent()) {
            throw new MediaException(422, "MEDIA_PREVIEW_UNSUPPORTED", "此素材类型不提供图片预览");
        }
        if (facts.width() == null || facts.height() == null || facts.width() <= 0 || facts.height() <= 0
                || (long) facts.width() * facts.height() > MAX_PIXELS) {
            throw new MediaException(422, "MEDIA_PREVIEW_DIMENSIONS", "预览需要有效的画面尺寸，单帧最多 6400 万像素");
        }
        sample(source);
        return source;
    }

    public View view(String jobId, CanvasActor actor) {
        var row = previews.selectById(jobId);
        if (row == null) {
            throw MediaException.invalid("预览任务关联不存在");
        }
        var details = new Details(row.getSourceResourceId(), row.getSourceVersion(), row.getSampleAtMs(),
                row.getOutputResourceId());
        var source = source(details.sourceId(), actor, false);
        if (source.version() != details.version()) {
            throw new MediaException(410, "MEDIA_PREVIEW_EXPIRED", "原素材版本已变化，请重新读取预览");
        }
        var job = jobs.get(jobId);
        MediaResource output = null;
        if (job.status().equals("SUCCEEDED")) {
            output = resources.get(details.outputId(), false).resource();
            access.ready(output, actor);
        }
        return new View(source.id(), source.version(), PROFILE, job.id(), job.status(), job.error(),
                output == null ? null : output.id(), output == null ? null : output.content().width(),
                output == null ? null : output.content().height(), details.sampleAtMs(),
                List.of("FAILED", "CANCELLED").contains(job.status()));
    }

    public static long sample(MediaResource source) {
        if (source.content().kind() == MediaKind.IMAGE) {
            return 0;
        }
        Long duration = source.content().durationMs();
        if (duration == null || duration <= 0) {
            throw new MediaException(422, "MEDIA_PREVIEW_DURATION", "原视频缺少有效时长，请重新登记媒体");
        }
        return Math.min(1000, duration / 10);
    }

    private record Details(String sourceId, int version, long sampleAtMs, String outputId) {
    }
}
