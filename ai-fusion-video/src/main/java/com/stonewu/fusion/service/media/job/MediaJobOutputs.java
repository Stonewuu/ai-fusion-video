package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaProbe;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaJobOrigin;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;

@Component
@RequiredArgsConstructor
public class MediaJobOutputs {
    private final MediaJobRepository jobs;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;
    private final MediaIntakeService intake;
    private final MediaProbe probe;

    public Optional<String> existing(MediaJob.Claim claim, String slot, MediaKind kind) {
        var job = jobs.check(claim);
        var result = resources.request(job.context().actor().userId(), "job:" + job.id() + ":" + slot);
        if (result.isEmpty() || !result.get().resource().state().equals("READY")) {
            return Optional.empty();
        }
        var resource = result.get().resource();
        access.ready(resource, job.context().actor());
        if (resource.content().kind() != kind) {
            throw MediaException.invalid("媒体任务产物类型不符");
        }
        return Optional.of(MediaReference.of(resource.id()));
    }

    public String store(MediaJob.Claim claim, String slot, Path file, String mime, MediaKind kind) throws IOException {
        var job = jobs.check(claim);
        var facts = probe.inspect(file, mime);
        if (facts.kind() != kind) {
            throw MediaException.invalid("媒体任务产物类型不符");
        }
        String name = switch (kind) {
            case VIDEO -> "合成视频.mp4";
            case IMAGE -> "处理图片.png";
            case AUDIO -> "生成音轨";
            default -> throw MediaException.invalid("媒体任务产物类型未登记");
        };
        return MediaReference.of(intake.job(file, name, facts.mimeType(), slot, job.projectId(), job.context(),
                new MediaJobOrigin(claim)).id());
    }
}
