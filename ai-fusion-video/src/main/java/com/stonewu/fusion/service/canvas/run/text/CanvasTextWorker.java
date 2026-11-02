package com.stonewu.fusion.service.canvas.run.text;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.agentscope.message.AgentModelHttpFailure;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import static com.stonewu.fusion.service.canvas.run.text.CanvasTextData.Claim;
import static com.stonewu.fusion.service.canvas.run.text.CanvasTextData.Job;

@Component
@RequiredArgsConstructor
@Slf4j
public class CanvasTextWorker {
    private final CanvasTextJobs tasks;
    private final CanvasTextRepository repository;
    private final CanvasRunRepository runs;
    private final CanvasTextModels models;
    private final CanvasTextProcessor processor;
    private final String owner = "canvas-text-" + UUID.randomUUID();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(4);
    private final Map<String, Claim> active = new ConcurrentHashMap<>();

    @Scheduled(fixedDelay = 500)
    public void dispatch() {
        if (!slots.tryAcquire()) {
            return;
        }
        try {
            var claimed = tasks.claim(owner);
            if (claimed.isEmpty()) {
                slots.release();
                return;
            }
            var job = claimed.get();
            active.put(job.id(), job.claim());
            workers.submit(() -> step(job));
        } catch (RuntimeException failure) {
            slots.release();
            if (!workers.isShutdown()) {
                log.warn("文本任务领取暂未完成: {}", failure.getClass().getSimpleName());
            }
        }
    }

    @Scheduled(fixedDelay = 20000)
    public void heartbeat() {
        active.values().forEach(tasks::heartbeat);
    }

    private void step(Job job) {
        try {
            if (job.result() != null) {
                publish(job.claim(), job);
                return;
            }
            if (job.startedAt() != null) {
                tasks.finish(job.claim(), "UNKNOWN_SUBMISSION", "远端结果未知，原文本请求不会自动重发",
                        false);
                return;
            }
            if (cancelled(job)) {
                tasks.finish(job.claim(), "CANCELLED", "尚未发送的文本任务已停止", false);
                return;
            }
            var run = tasks.check(job.claim(), false);
            try (var model = models.provision(job.input())) {
                var messages = processor.messages(job, run);
                tasks.begin(job.claim());
                var output = processor.generate(model, messages);
                tasks.received(job.claim(), output);
                publish(job.claim(), repository.get(job.id(), false));
            }
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
            try {
                var current = repository.get(job.id(), false);
                if (current.startedAt() != null && current.result() == null && AgentModelHttpFailure.status(
                        failure) != null) {
                    var rejected = GenerationProviderFailure.find(failure);
                    tasks.finish(job.claim(), cancelled(current) ? "CANCELLED" : "FAILED",
                            rejected != null ? rejected.getMessage() : failure.getMessage(), false);
                    return;
                }
                boolean uncertain = current.startedAt() != null && current.result() == null;
                String state = uncertain ? "UNKNOWN_SUBMISSION" : cancelled(current) ? "CANCELLED"
                        : "FAILED";
                String message = uncertain ? "远端结果未知，文本调用未取得完整响应，已结束本地等待"
                        : state.equals("CANCELLED") ? "文本任务已停止"
                        : failure instanceof BusinessException ? failure.getMessage() : "文本处理未完成，请检查模型配置后重新确认";
                tasks.finish(job.claim(), state, message, false);
            } catch (RuntimeException fenced) {
                log.debug("文本任务已被接管或暂停 [jobId={}]", job.id());
            }
        } finally {
            active.remove(job.id(), job.claim());
            slots.release();
        }
    }

    private void publish(Claim claim, Job job) {
        if (cancelled(job)) {
            tasks.finish(claim, "CANCELLED", "原文本调用已结束，取消后不发布结果", false);
        } else if (!job.result().path("complete").asBoolean() && !job.resultAccepted()) {
            tasks.finish(claim, "FAILED", "模型返回了截断或不完整的文本，未继续执行后续任务", false);
        } else {
            tasks.finish(claim, "SUCCEEDED", null, false);
        }
    }

    private boolean cancelled(Job job) {
        return job.cancelRequested() || runs.run(job.runId(), false).cancelRequested() || !job.deadlineAt()
                .isAfter(Instant.now());
    }

    @PreDestroy
    public void close() {
        workers.shutdownNow();
    }
}
