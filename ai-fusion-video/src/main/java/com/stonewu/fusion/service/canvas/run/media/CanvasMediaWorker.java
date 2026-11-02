package com.stonewu.fusion.service.canvas.run.media;

import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
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
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

@Component
@RequiredArgsConstructor
@Slf4j
public class CanvasMediaWorker {
    private final CanvasMediaJobs tasks;
    private final MediaJobRepository jobs;
    private final CanvasLocalMediaProcessor local;
    private final CanvasAudioProcessor audio;
    private final String owner = "canvas-media-" + UUID.randomUUID();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(4);
    private final Map<String, MediaJob.Claim> active = new ConcurrentHashMap<>();

    @Scheduled(fixedDelay = 500)
    public void dispatch() {
        if (!slots.tryAcquire()) {
            return;
        }
        try {
            var job = tasks.claim(owner);
            if (job.isEmpty()) {
                slots.release();
                return;
            }
            active.put(job.get().id(), job.get().claim());
            workers.submit(() -> step(job.get()));
        } catch (RuntimeException failure) {
            slots.release();
            if (!workers.isShutdown()) {
                log.warn("画布媒体领取失败", failure);
            }
        }
    }

    @Scheduled(fixedDelay = 20000)
    public void heartbeat() {
        active.values().forEach(tasks::heartbeat);
    }

    private void step(MediaJob job) {
        try (var directory = new WorkDirectory()) {
            var input = tasks.input(job);
            if (CanvasMediaExecutor.AUDIO.contains(input.operation())) {
                if (!job.cancelRequested()) {
                    try {
                        tasks.check(job.claim(), false);
                    } catch (BusinessException denied) {
                        if (denied.getCode() != 403 && denied.getCode() != 404 && denied.getCode() != 410) {
                            throw denied;
                        }
                        job = jobs.requestStop(job.claim());
                    }
                }
                audio.step(job, input, directory.path);
            } else if (job.cancelRequested()) {
                tasks.finish(job.claim(), "CANCELLED", "本地处理已停止", null, false);
            } else {
                tasks.check(job.claim(), false);
                var output = local.process(job, input, directory.path);
                tasks.finish(job.claim(), "SUCCEEDED", null, output, false);
            }
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
            try {
                if (failure instanceof MediaException media && media.getErrorCode()
                        .equals("MEDIA_INGEST_IN_PROGRESS")) {
                    tasks.waitNext(job.claim(), "恢复原任务媒体接收", false);
                    return;
                }
                var current = jobs.get(job.id());
                var intent = jobs.submission(job.id(), "audio");
                boolean unknown = intent.isPresent() && tasks.received(job.id()) == null;
                if (unknown && intent.get().providerId() != null && current.deadline().isAfter(jobs.now())
                        && failure instanceof BusinessException denied
                        && (denied.getCode() == 403 || denied.getCode() == 404 || denied.getCode() == 410)) {
                    jobs.requestStop(job.claim());
                    tasks.waitNext(job.claim(), "停止后续处理，继续查询原音频任务", false);
                    return;
                }
                if (unknown && intent.get().providerId() != null && current.deadline().isAfter(jobs.now())
                        && !(failure instanceof BusinessException)) {
                    tasks.waitNext(job.claim(), "查询暂不可用，稍后查询原音频任务", true);
                    return;
                }
                String status = unknown ? "UNKNOWN_SUBMISSION" : current.cancelRequested() ? "CANCELLED" : "FAILED";
                String message = failure.getMessage() == null ? "媒体处理失败" : failure.getMessage();
                tasks.finish(job.claim(), status, (unknown ? "远端结果未知：" : "")
                        + message.substring(0, Math.min(message.length(), 2000)), null, false);
            } catch (RuntimeException fenced) {
                log.debug("画布媒体任务已停止或已接管 [jobId={}]", job.id(), fenced);
            }
        } finally {
            active.remove(job.id(), job.claim());
            slots.release();
        }
    }

    @PreDestroy
    public void close() {
        workers.shutdownNow();
    }

    private static final class WorkDirectory implements AutoCloseable {
        final Path path = Files.createTempDirectory("afv-canvas-media-").toRealPath();

        WorkDirectory() throws IOException {
        }

        @Override
        public void close() throws IOException {
            try (var paths = Files.walk(path)) {
                for (var item : paths.sorted(Comparator.reverseOrder()).toList()) {
                    if (!item.toAbsolutePath().normalize().startsWith(path)) {
                        throw new IOException("媒体临时文件超出当前任务目录");
                    }
                    Files.deleteIfExists(item);
                }
            }
        }
    }
}
