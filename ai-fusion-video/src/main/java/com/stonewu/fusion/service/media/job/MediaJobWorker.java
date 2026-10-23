package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.model.MediaException;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

@Component
@RequiredArgsConstructor
@Slf4j
public class MediaJobWorker {
    private final MediaJobRepository jobs;
    private final ComposeVideoProcessor video;
    private final ComposeBgmProcessor bgm;
    private final MediaJobCompletion completion;
    private final EpisodeAudioJobs audio;
    private final String instance = "media-" + UUID.randomUUID();
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final AtomicBoolean busy = new AtomicBoolean();
    private final AtomicReference<MediaJob.Claim> active = new AtomicReference<>();

    @Scheduled(fixedDelay = 1000)
    public void dispatch() {
        if (!busy.compareAndSet(false, true)) {
            return;
        }
        try {
            executor.execute(() -> {
                try {
                    runOne();
                } finally {
                    busy.set(false);
                }
            });
        } catch (RuntimeException failure) {
            busy.set(false);
            if (!executor.isShutdown()) {
                throw failure;
            }
        }
    }

    @Scheduled(fixedDelay = 20000)
    public void heartbeat() {
        var claim = active.get();
        if (claim != null) {
            jobs.heartbeat(claim);
        }
    }

    public boolean runOne() {
        var claimed = jobs.claim(instance);
        if (claimed.isEmpty()) {
            return false;
        }
        var job = claimed.get();
        var claim = job.claim();
        active.set(claim);
        try (var directory = new WorkDirectory()) {
            if (job.cancelRequested()) {
                bgm.observeCancellation(claim);
                return true;
            }
            video.prepare(claim, directory.path);
            if (bgm.process(claim, directory.path)) {
                completion.complete(claim);
            }
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return true; // 关闭进程后由新租约恢复。
            }
            if (failure instanceof MediaException media && media.getErrorCode().equals("MEDIA_INGEST_IN_PROGRESS")) {
                jobs.waitForProvider(claim, 10);
                return true;
            }
            String status = "FAILED";
            String message = failure.getMessage() == null ? "媒体处理失败" : failure.getMessage();
            var current = jobs.get(job.id());
            boolean unresolved = !audio.unresolved(job.id()).isEmpty();
            if (unresolved && failure instanceof MediaException media && media.getErrorCode()
                    .equals("MEDIA_AUDIO_QUERY_FAILED")) {
                try {
                    audio.waitNext(claim, "查询暂不可用，稍后查询原音频任务", 3, true);
                    return true;
                } catch (RuntimeException unavailable) {
                    message = unavailable.getMessage();
                }
            }
            if (unresolved) {
                status = "NEEDS_REVIEW";
                message = "原音频请求尚未确认结束，额度保留。" + message;
            } else if (current.cancelRequested()) {
                status = "CANCELLED";
                message = "已停止后续处理；没有仍待核实的音频请求";
            } else if (failure instanceof CanvasException canvas && canvas.getErrorCode() == CanvasErrorCode.VERSION_CONFLICT) {
                status = "CONFLICT";
                message = "合成期间分镜已修改，成片已保留，没有覆盖当前版本";
            } else if (audio.hasSuccessfulResult(
                    job.id()) && !(failure instanceof MediaException media && media.getErrorCode()
                    .equals("MEDIA_AUDIO_PROVIDER_FAILED"))) {
                status = "NEEDS_REVIEW";
                message = "原音频响应已保存，可核实后继续登记。" + message;
            } else if (failure instanceof MediaException media && (media.getErrorCode()
                    .equals("MEDIA_SUBMISSION_UNKNOWN")
                    || media.getErrorCode().equals("MEDIA_PROVIDER_WAIT_EXPIRED"))) {
                status = "NEEDS_REVIEW";
            }
            log.warn("媒体任务结束: id={}, status={}, reason={}", job.id(), status, message);
            if (completion.recordFailure(claim, message)) {
                if (!status.equals("NEEDS_REVIEW")) {
                    completion.cleanBusinessStatus(claim);
                }
                completion.terminal(claim, status, message);
            }
        } finally {
            active.compareAndSet(claim, null);
        }
        return true;
    }

    @PreDestroy
    public void stop() {
        executor.shutdownNow();
    }

    private static final class WorkDirectory implements AutoCloseable {
        final Path path = Files.createTempDirectory("afv-compose-").toRealPath();

        WorkDirectory() throws IOException {
        }

        @Override
        public void close() throws IOException {
            try (var paths = Files.walk(path)) {
                for (Path item : paths.sorted(Comparator.reverseOrder()).toList()) {
                    if (!item.toAbsolutePath().normalize().startsWith(path)) {
                        throw new IOException("媒体临时目录范围无效");
                    }
                    Files.deleteIfExists(item);
                }
            }
        }
    }
}
