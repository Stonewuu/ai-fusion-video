package com.stonewu.fusion.service.canvas.run.mcp;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Claim;
import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Job;

@Component
@RequiredArgsConstructor
@Slf4j
public class CanvasMcpWorker {
    private final CanvasMcpJobs tasks;
    private final CanvasMcpRepository repository;
    private final CanvasMcpContracts contracts;
    private final CanvasMcpProcessor processor;
    private final CanvasRunRepository runs;
    private final ObjectMapper mapper;
    private final String owner = "canvas-mcp-" + UUID.randomUUID();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(4);
    private final Map<String, Claim> active = new ConcurrentHashMap<>();

    @Scheduled(fixedDelay = 500)
    public void dispatch() {
        if (!slots.tryAcquire()) {
            return;
        }
        try {
            var next = tasks.claim(owner);
            if (next.isEmpty()) {
                slots.release();
                return;
            }
            var job = next.get();
            active.put(job.id(), job.claim());
            workers.submit(() -> execute(job));
        } catch (RuntimeException failure) {
            slots.release();
            if (!workers.isShutdown()) {
                log.warn("MCP 任务领取暂未完成: {}", failure.getClass().getSimpleName());
            }
        }
    }

    @Scheduled(fixedDelay = 20000)
    public void heartbeat() {
        active.values().forEach(tasks::heartbeat);
    }

    private void execute(Job job) {
        try {
            if (job.result() != null) {
                publish(job.claim(), job);
                return;
            }
            if (job.startedAt() != null) {
                tasks.finish(job.claim(), "UNKNOWN_SUBMISSION",
                        "原 MCP 请求已发送但没有完整响应，请核实原调用；不会自动重发", true);
                return;
            }
            if (cancelled(job)) {
                tasks.finish(job.claim(), "CANCELLED", "未发送的 MCP 调用已停止", false);
                return;
            }
            tasks.check(job.claim());
            try (var client = contracts.open(job.input().contract(), job.userId())) {
                var arguments = mapper.convertValue(job.input().arguments(), new TypeReference<Map<String, Object>>() {
                });
                tasks.begin(job.claim());
                var timeout = Duration.ofSeconds(job.input().contract().definition().timeoutSeconds());
                var remaining = Duration.between(Instant.now(), job.deadlineAt());
                if (remaining.compareTo(timeout) < 0) {
                    timeout = remaining.isNegative() ? Duration.ofMillis(1) : remaining;
                }
                var result = client.invoke(arguments, timeout, Map.of("aiFusionVideo.canvasCallId", job.id()))
                        .block(timeout.plusSeconds(1));
                tasks.received(job.claim(), processor.response(job.input().contract().definition(), result));
                publish(job.claim(), repository.get(job.id(), false));
            }
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
            log.warn("MCP 调用未完成 [jobId={}]: {}", job.id(), failure.getClass().getSimpleName());
            try {
                var current = repository.get(job.id(), false);
                boolean unknown = current.startedAt() != null && current.result() == null;
                String state = unknown ? "UNKNOWN_SUBMISSION" : cancelled(
                        current) ? "CANCELLED" : current.result() != null ? "NEEDS_REVIEW" : "FAILED";
                String message = unknown ? "MCP 调用缺少完整响应，请核实原请求；系统不会重新提交"
                        : state.equals("CANCELLED") ? "MCP 后续执行已停止"
                        : failure instanceof BusinessException ? failure.getMessage() : "MCP 执行未完成，请核对操作契约和远端服务";
                tasks.finish(job.claim(), state, message, unknown);
            } catch (RuntimeException fenced) {
                log.debug("MCP 任务已被其他工作者接管 [jobId={}]", job.id());
            }
        } finally {
            active.remove(job.id(), job.claim());
            slots.release();
        }
    }

    private void publish(Claim claim, Job job) {
        if (cancelled(job)) {
            tasks.finish(claim, "CANCELLED", "原 MCP 调用已返回，取消后不发布结果", false);
        } else if (!job.result().path("valid").asBoolean()) {
            tasks.finish(claim, "FAILED", job.result().path("message").asText("MCP 输出未通过校验"), false);
        } else {
            tasks.finish(claim, "SUCCEEDED", null, false);
        }
    }

    private boolean cancelled(Job job) {
        return job.cancelled() || runs.run(job.runId(), false).cancelRequested() || !job.deadlineAt()
                .isAfter(Instant.now());
    }

    @PreDestroy
    public void close() {
        workers.shutdownNow();
    }
}
