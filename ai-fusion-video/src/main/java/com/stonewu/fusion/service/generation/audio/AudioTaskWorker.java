package com.stonewu.fusion.service.generation.audio;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.google.GoogleRequestUncertainException;
import com.stonewu.fusion.service.ai.google.GoogleAudioProtocol;
import com.stonewu.fusion.entity.generation.AudioTask;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.UncheckedIOException;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

@Component
@RequiredArgsConstructor
@Slf4j
public class AudioTaskWorker {
    private final AudioTaskRepository tasks;
    private final AudioTaskRunner runner;
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(4);
    private final Map<String, AudioTask> active = new ConcurrentHashMap<>();

    @Scheduled(fixedDelay = 1000)
    public void dispatch() {
        while (slots.tryAcquire()) {
            try {
                var claimed = tasks.claim(UUID.randomUUID().toString());
                if (claimed.isEmpty()) {
                    slots.release();
                    return;
                }
                var task = claimed.get();
                active.put(task.getId(), task);
                workers.execute(() -> execute(task));
            } catch (RuntimeException failure) {
                slots.release();
                log.warn("音频任务调度失败: {}", failure.getClass().getSimpleName());
                return;
            }
        }
    }

    @Scheduled(fixedDelay = 15000)
    public void heartbeat() {
        for (var task : active.values()) {
            tasks.heartbeat(task);
        }
    }

    private void execute(AudioTask task) {
        try {
            if (task.getOutputJson() != null) {
                tasks.finish(task, "COMPLETED", runner.save(task), null);
            } else {
                tasks.received(task, runner.submit(task));
            }
        } catch (GoogleAudioProtocol.InvalidOutput failure) {
            tasks.finish(task, "NO_OUTPUT", null, failure.getMessage());
        } catch (GoogleRequestUncertainException failure) {
            tasks.finish(task, "UNKNOWN", null, "调用已中断，请检查生成记录后再试");
        } catch (UncheckedIOException failure) {
            tasks.finish(task, "FAILED", null, task.getOutputJson() != null
                    ? "结果保存失败，可重试保存" : "参考素材无法读取，请重新选择");
        } catch (RuntimeException failure) {
            var providerFailure = GenerationProviderFailure.find(failure);
            String message = providerFailure == null
                    ? failure instanceof BusinessException ? failure.getMessage()
                            : task.getOutputJson() != null ? "结果保存失败，可重试保存" : "音频生成未完成，请稍后重试"
                    : providerFailure.reason();
            tasks.finish(task, "FAILED", null, message);
            log.warn("Google 音频生成失败: task={}, type={}", task.getId(), failure.getClass().getSimpleName());
        } finally {
            active.remove(task.getId());
            slots.release();
        }
    }

    @PreDestroy
    public void close() {
        workers.shutdownNow();
    }
}
