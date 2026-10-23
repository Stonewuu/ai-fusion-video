package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategyRouter;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategyRouter;
import com.stonewu.fusion.service.media.model.MediaException;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

/**
 * 每次领取最多提交一次或查询一次；等待记录留在数据库，不占用线程。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class RemoteGenerationWorker {
    private final GenerationJobRepository jobs;
    private final GenerationAttemptRepository attempts;
    private final GenerationJobLifecycle lifecycle;
    private final GenerationExecutionIdentity identity;
    private final GenerationModelSnapshot models;
    private final GenerationProviderInputs inputs;
    private final GenerationResultIntake intake;
    private final ImageGenerationStrategyRouter imageProtocols;
    private final VideoGenerationStrategyRouter videoProtocols;
    private final String instance = "generation-" + UUID.randomUUID();
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(16);
    private final Set<RemoteGenerationJob.Claim> active = ConcurrentHashMap.newKeySet();

    @Scheduled(fixedDelay = 500)
    public void dispatch() {
        while (slots.tryAcquire()) {
            Optional<RemoteGenerationJob> claimed;
            try {
                claimed = jobs.claim(instance);
            } catch (RuntimeException failure) {
                slots.release();
                throw failure;
            }
            if (claimed.isEmpty()) {
                slots.release();
                return;
            }
            var job = claimed.get();
            active.add(job.claim());
            try {
                executor.execute(() -> {
                    try {
                        execute(job);
                    } finally {
                        active.remove(job.claim());
                        slots.release();
                    }
                });
            } catch (RuntimeException stopped) {
                active.remove(job.claim());
                slots.release();
                if (!executor.isShutdown()) {
                    throw stopped;
                }
                return;
            }
        }
    }

    @Scheduled(fixedDelay = 20_000)
    public void heartbeat() {
        active.forEach(jobs::heartbeat);
    }

    /**
     * 供真实数据库恢复测试手动驱动一次领取。
     */
    public boolean runOne() {
        var claimed = jobs.claim(instance);
        if (claimed.isEmpty()) {
            return false;
        }
        var claim = claimed.get().claim();
        active.add(claim);
        try {
            execute(claimed.get());
        } finally {
            active.remove(claim);
        }
        return true;
    }

    private void execute(RemoteGenerationJob initial) {
        var claim = initial.claim();
        try (var ignored = GenerationLeaseScope.enter(claim)) {
            var submission = lifecycle.start(claim);
            var job = jobs.get(claim.id());
            if (!identity.permitted(submission) && !job.cancelRequested()) {
                lifecycle.requestStop(claim);
                job = jobs.get(job.id());
            }
            boolean preparing = job.phase().equals("NEW") || job.phase().equals("PREPARING");
            if (preparing && job.cancelRequested()) {
                lifecycle.fail(claim, "CANCELLED", "CANCELLED_BEFORE_SUBMISSION", "已取消排队，尚未提交供应商");
                return;
            }
            if (jobs.now().isAfter(job.deadline()) && !job.phase().equals("INGESTING")) {
                lifecycle.fail(claim, preparing ? "FAILED" : "NEEDS_REVIEW", "GENERATION_DEADLINE_EXCEEDED",
                        preparing ? "准备超过等待期限，未提交供应商" : "等待超过期限，供应商任务仍需核实；没有自动重发");
                return;
            }
            ApiConfig api = models.forExecution(submission);
            if (preparing) {
                submit(job, submission, api);
                job = jobs.get(job.id());
                if (job.phase().equals("POLLING")) {
                    jobs.waitForNextPoll(claim, 2, "POLLING", null, null);
                    return;
                }
            } else if (job.phase().equals("SUBMITTING")) {
                var attempt = attempts.get(job.activeAttemptId());
                if (attempt.acknowledged()) {
                    job = attempts.applyReceipt(claim);
                } else if (attempt.providerTaskId() == null) {
                    lifecycle.fail(claim, "UNKNOWN_SUBMISSION", "UNKNOWN_SUBMISSION",
                            "提交意图已保存，但未取得回执；请核实供应商后再决定是否重试");
                    return;
                } else {
                    // 客户端预先分配的任务 ID 只能用于查询对账，不能假设重复 POST 幂等。
                    var result = query(job, submission, api);
                    attempts.acknowledge(attempt.id(), GenerationProviderReceipt.accepted(attempt.providerTaskId()));
                    job = attempts.applyReceipt(claim);
                    if (result.state() == RemoteGenerationResult.State.PENDING) {
                        jobs.waitForNextPoll(claim, result.nextPollSeconds(), "POLLING", null, null);
                        return;
                    }
                    jobs.remoteResult(claim, result);
                    job = jobs.get(job.id());
                }
            }
            if (job.phase().equals("POLLING")) {
                if (job.cancelRequested() && !job.cancelSent() && requestRemoteCancel(job, submission, api)) {
                    lifecycle.cancellationSent(claim);
                }
                var result = query(job, submission, api);
                if (result.state() == RemoteGenerationResult.State.PENDING) {
                    jobs.waitForNextPoll(claim, result.nextPollSeconds(), "POLLING", null, null);
                    return;
                }
                jobs.remoteResult(claim, result);
                job = jobs.get(job.id());
            }
            if (!job.phase().equals("INGESTING") || job.result() == null) {
                throw new IllegalStateException("未登记的生成执行阶段");
            }
            if (job.cancelRequested() || job.result().state() == RemoteGenerationResult.State.CANCELLED) {
                lifecycle.fail(claim, "CANCELLED", "GENERATION_CANCELLED",
                        "本地执行已取消，供应商已结束；远端已产生的费用不会自动退回");
            } else if (job.result().state() == RemoteGenerationResult.State.FAILED) {
                lifecycle.fail(claim, "FAILED", "PROVIDER_FAILED", job.result().error());
            } else {
                lifecycle.succeed(claim, intake.receive(job, submission, api));
            }
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
            handleFailure(claim, failure);
        }
    }

    private void submit(RemoteGenerationJob job, GenerationSubmission submission, ApiConfig api) {
        attempts.prepare(job.claim());
        job = jobs.get(job.id());
        PreparedGenerationCall prepared;
        if (submission.kind() == GenerationKind.IMAGE) {
            var task = inputs.image(job, submission, api, true);
            var protocol = imageProtocols.resolve(submission.model(), api);
            prepared = protocol.prepare(task, submission.model(), api);
        } else {
            var task = inputs.video(job, submission, api, true);
            var protocol = videoProtocols.resolve(submission.model());
            prepared = protocol.prepare(task, submission.model(), api);
        }
        try (prepared) {
            verifyBeforeSubmission(job, submission);
            var attempt = attempts.begin(job.claim(), prepared.intent());
            var receipt = prepared.submit();
            attempts.acknowledge(attempt.id(), receipt);
            attempts.applyReceipt(job.claim());
        }
    }

    private void verifyBeforeSubmission(RemoteGenerationJob job, GenerationSubmission submission) {
        if (!identity.permitted(submission)) {
            lifecycle.requestStop(job.claim());
            throw new GenerationJobException(403, "GENERATION_SCOPE_REVOKED",
                    "生成准备期间权限或发起运行状态已变化，未提交供应商");
        }
        if (jobs.now().isAfter(job.deadline())) {
            throw new GenerationJobException(409, "GENERATION_DEADLINE_EXCEEDED", "准备超过等待期限，未提交供应商");
        }
        models.forExecution(submission);
    }

    private RemoteGenerationResult query(RemoteGenerationJob job, GenerationSubmission submission, ApiConfig api) {
        if (submission.kind() == GenerationKind.IMAGE) {
            return imageProtocols.resolve(submission.model(), api)
                    .query(job.providerTaskId(), inputs.image(job, submission, api, false), submission.model(), api);
        }
        return videoProtocols.resolve(submission.model())
                .query(job.providerTaskId(), inputs.video(job, submission, api, false), submission.model(), api);
    }

    private boolean requestRemoteCancel(RemoteGenerationJob job, GenerationSubmission submission, ApiConfig api) {
        if (submission.kind() == GenerationKind.IMAGE) {
            return imageProtocols.resolve(submission.model(), api)
                    .cancel(job.providerTaskId(), inputs.image(job, submission, api, false), submission.model(), api);
        }
        return videoProtocols.resolve(submission.model())
                .cancel(job.providerTaskId(), inputs.video(job, submission, api, false), submission.model(), api);
    }

    private void handleFailure(RemoteGenerationJob.Claim claim, Exception failure) {
        try {
            var current = jobs.get(claim.id());
            String code = failure instanceof GenerationJobException known ? known.getErrorCode() : failure instanceof MediaException media ? media.getErrorCode() : "GENERATION_EXECUTION_ERROR";
            if (code.equals("GENERATION_LEASE_LOST")) {
                // 取消也会拒绝继续处理，但原工作者仍可在真实租约下结束尚未发送/远端已结束的任务。
                boolean noRemotePending = current.phase().equals("NEW") || current.phase()
                        .equals("PREPARING") || current.phase().equals("INGESTING") && current.result() != null;
                if (current.cancelRequested() && noRemotePending) {
                    lifecycle.fail(claim, "CANCELLED", "GENERATION_CANCELLED",
                            "已取消后续处理，没有待确认的远端生成请求");
                }
                return;
            }
            String message = failure.getMessage() == null ? "生成任务执行异常" : failure.getMessage();
            if (current.phase().equals("SUBMITTING")) {
                var intent = attempts.get(current.activeAttemptId());
                if (intent.providerTaskId() != null || intent.acknowledged()) {
                    jobs.waitForNextPoll(claim, 5, "SUBMITTING", code, "正在根据已保存的提交记录核实受理状态");
                } else {
                    lifecycle.fail(claim, "UNKNOWN_SUBMISSION", "UNKNOWN_SUBMISSION",
                            "请求结果未确认，不能自动重发：" + message);
                }
            } else if (code.equals("GENERATION_LATE_RECEIPT")) {
                lifecycle.fail(claim, "NEEDS_REVIEW", code, message);
            } else if (current.phase().equals("NEW") || current.phase().equals("PREPARING")) {
                lifecycle.fail(claim, current.cancelRequested() ? "CANCELLED" : "FAILED", code, message);
            } else if (code.equals("PROVIDER_TASK_UNKNOWN") || code.equals(
                    "GENERATION_PROVIDER_CHANGED") || code.equals("GENERATION_PROVIDER_UNAVAILABLE") || code.equals(
                    "GENERATION_MODEL_DISABLED")) {
                lifecycle.fail(claim, "NEEDS_REVIEW", code, message);
            } else if (current.consecutiveErrors() < 8) {
                int delay = Math.min(60, 1 << Math.min(6, current.consecutiveErrors() + 1));
                jobs.waitForNextPoll(claim, delay, current.phase(), code, message);
            } else {
                lifecycle.fail(claim, current.phase().equals("INGESTING") ? "FAILED" : "NEEDS_REVIEW", code, message);
            }
        } catch (GenerationJobException lost) {
            if (!lost.getErrorCode().equals("GENERATION_LEASE_LOST")) {
                log.warn("生成任务状态记录失败: id={}, code={}", claim.id(), lost.getErrorCode());
            }
        } catch (Exception recordFailure) {
            log.warn("生成状态暂未提交，将由到期租约恢复: id={}", claim.id(), recordFailure);
        }
    }

    @PreDestroy
    public void stop() {
        executor.shutdownNow();
    }
}
