package com.stonewu.fusion.service.media.preview;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.media.MediaContentReader;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.job.MediaProcessRunner;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

@Component
@RequiredArgsConstructor
@Slf4j
public class MediaPreviewWorker {
    private final MediaPreviewJobs previews;
    private final MediaJobRepository jobs;
    private final MediaContentReader content;
    private final MediaProcessRunner processes;
    private final MediaPreviewIntake intake;
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(2);
    private final Set<MediaJob.Claim> active = ConcurrentHashMap.newKeySet();
    private final String owner = "media-preview-" + UUID.randomUUID();
    @Value("${video.compose.ffmpeg-path:ffmpeg}")
    private String ffmpeg = "ffmpeg";

    @Scheduled(fixedDelay = 500)
    public void dispatch() {
        while (slots.tryAcquire()) {
            Optional<MediaJob> next;
            try {
                next = previews.claim(owner);
            } catch (RuntimeException failure) {
                slots.release();
                throw failure;
            }
            if (next.isEmpty()) {
                slots.release();
                return;
            }
            var claim = next.get().claim();
            active.add(claim);
            try {
                executor.execute(() -> {
                    try {
                        process(claim);
                    } finally {
                        active.remove(claim);
                        slots.release();
                    }
                });
            } catch (RuntimeException failure) {
                active.remove(claim);
                slots.release();
                if (!executor.isShutdown()) {
                    throw failure;
                }
                return;
            }
        }
    }

    @Scheduled(fixedDelay = 20000)
    public void heartbeat() {
        active.forEach(jobs::heartbeat);
    }

    public boolean runOne() {
        var next = previews.claim(owner);
        if (next.isEmpty()) {
            return false;
        }
        var claim = next.get().claim();
        active.add(claim);
        try {
            process(claim);
        } finally {
            active.remove(claim);
        }
        return true;
    }

    private void process(MediaJob.Claim claim) {
        Path directory = null;
        try {
            var job = previews.checked(claim);
            var existing = previews.existing(claim);
            if (existing.isPresent()) {
                previews.complete(claim, existing.get());
                return;
            }
            var input = previews.input(job);
            directory = Files.createTempDirectory("afv-media-preview-");
            Path source = directory.resolve("source.media"), output = directory.resolve("preview.png");
            content.copy(input.source().id(), job.context().actor(), source, MediaIntakeService.MAX_RESULT_BYTES);
            previews.checked(claim);
            var command = new ArrayList<>(
                    List.of(ffmpeg, "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe"));
            if (input.source().content().kind() == MediaKind.VIDEO) {
                command.addAll(List.of("-ss", Double.toString(input.sampleAtMs() / 1000.0)));
            }
            command.addAll(List.of("-i", source.toString(), "-frames:v", "1", "-an", "-vf",
                    "scale=w='min(640,iw)':h='min(480,ih)':force_original_aspect_ratio=decrease,setsar=1",
                    output.toString()));
            processes.run(command, directory, output, () -> previews.checked(claim));
            var result = intake.store(previews.checked(claim), output);
            previews.complete(claim, result.id());
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
            if (failure instanceof DataAccessException) {
                log.warn("预览接收状态将由原租约恢复 [jobId={}]", claim.id(), failure);
                return;
            }
            try {
                if (failure instanceof MediaException media && media.getErrorCode()
                        .equals("MEDIA_INGEST_IN_PROGRESS")) {
                    previews.waitForIngest(claim);
                } else {
                    boolean denied = failure instanceof BusinessException business && List.of(403, 404, 410)
                            .contains(business.getCode())
                            || failure instanceof MediaException media && media.getErrorCode()
                            .equals("MEDIA_JOB_FENCED");
                    log.warn("媒体预览生成未完成 [jobId={}]", claim.id(), failure);
                    previews.stop(claim, denied,
                            denied ? "原素材、权限或任务状态已变化，预览生成已停止" : "预览生成失败，请重试或检查原素材与媒体处理配置");
                }
            } catch (RuntimeException unavailable) {
                log.warn("预览任务收尾尚未保存，将由原租约恢复 [jobId={}]", claim.id(), unavailable);
            }
        } finally {
            if (directory != null) {
                try (var paths = Files.walk(directory)) {
                    for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                        Files.deleteIfExists(path);
                    }
                } catch (Exception cleanup) {
                    log.warn("预览临时文件清理未完成 [jobId={}]", claim.id(), cleanup);
                }
            }
        }
    }

    @PreDestroy
    public void stop() {
        executor.shutdownNow();
    }
}
