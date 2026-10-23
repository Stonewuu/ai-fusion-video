package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.generation.GenerationAttemptEntity;
import com.stonewu.fusion.mapper.generation.GenerationAttemptMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;

import java.util.Optional;

import com.stonewu.fusion.service.canvas.CanvasJson;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Objects;
import java.util.UUID;

@Repository
@RequiredArgsConstructor
public class GenerationAttemptRepository {
    private final GenerationAttemptMapper attempts;
    private final RemoteGenerationJobMapper remoteJobs;
    private final CanvasJson json;
    private final GenerationJobRepository jobs;

    /**
     * 先登记准备阶段，输入票据绑定这次尝试；此时尚未发出付费请求。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Attempt prepare(RemoteGenerationJob.Claim claim) {
        var job = jobs.leased(claim, false);
        if (job.phase().equals("PREPARING") && job.activeAttemptId() != null) {
            return get(job.activeAttemptId());
        }
        if (!job.phase().equals("NEW") || job.activeAttemptId() != null) {
            throw new GenerationJobException(409, "GENERATION_ALREADY_SUBMITTED",
                    "此尝试已提交，请读取原回执或核实受理状态");
        }
        requireNoLateReceipt(job.id());
        String id = UUID.randomUUID().toString();
        int attempt = attempts.nextNumber(job.id());
        attempts.insertPrepared(id, job.id(), attempt);
        remoteJobs.prepareAttempt(id, job.id());
        return get(id);
    }

    /**
     * 准备和验证完成后才提交付费意图，之后恢复只能查询原回执，不能再次发送。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Attempt begin(RemoteGenerationJob.Claim claim, GenerationPreparedRequest prepared) {
        var job = jobs.leased(claim, false);
        if (job.phase().equals("NEW")) {
            prepare(claim);
        }
        job = jobs.leased(claim, false);
        if (!job.phase().equals("PREPARING") || job.activeAttemptId() == null) {
            throw new GenerationJobException(409, "GENERATION_ALREADY_SUBMITTED",
                    "此尝试已提交，请读取原回执或核实受理状态");
        }
        requireNoLateReceipt(job.id());
        int changed = attempts.beginSubmission(prepared.payload() == null ? null : json.write(prepared.payload()),
                prepared.providerTaskId(), job.activeAttemptId());
        if (changed != 1) {
            throw GenerationJobException.fenced();
        }
        remoteJobs.beginSubmission(prepared.providerTaskId(), job.id());
        return get(job.activeAttemptId());
    }

    private void requireNoLateReceipt(String jobId) {
        var acknowledged = attempts.lockRetriedAcknowledgements(jobId);
        if (acknowledged.stream().anyMatch(Objects::nonNull)) {
            throw new GenerationJobException(409, "GENERATION_LATE_RECEIPT",
                    "已收到先前请求的迟到回执，不能再次提交，请重新核实");
        }
    }

    public Attempt get(String id) {
        return Optional.ofNullable(attempts.selectById(id)).map(this::read)
                .orElseThrow(() -> new GenerationJobException(404, "GENERATION_ATTEMPT_NOT_FOUND", "提交意图不存在"));
    }

    private Attempt read(GenerationAttemptEntity row) {
        String request = row.getRequestJson();
        String result = row.getResultJson();
        return new Attempt(row.getId(), row.getJobId(), row.getAttemptNo(),
                request == null ? null : json.read(request, ObjectNode.class), row.getProviderTaskId(),
                result == null ? null : json.read(result, RemoteGenerationResult.class),
                row.getAcknowledgedAt() != null, row.getSubmittedAt() != null, row.getResolution());
    }

    public List<Summary> forJob(String jobId) {
        return attempts.selectList(new LambdaQueryWrapper<GenerationAttemptEntity>()
                        .select(GenerationAttemptEntity::getId, GenerationAttemptEntity::getAttemptNo,
                                GenerationAttemptEntity::getProviderTaskId, GenerationAttemptEntity::getSubmittedAt,
                                GenerationAttemptEntity::getAcknowledgedAt, GenerationAttemptEntity::getResolution)
                        .eq(GenerationAttemptEntity::getJobId, jobId).orderByAsc(GenerationAttemptEntity::getAttemptNo))
                .stream().map(row -> new Summary(row.getId(), row.getAttemptNo(), row.getProviderTaskId(),
                        row.getSubmittedAt() != null, row.getAcknowledgedAt() != null, row.getResolution())).toList();
    }

    /**
     * 失去租约的请求仍可补记不可变回执；不会写结果条目、业务对象或继续执行。
     */
    @Transactional(propagation = Propagation.REQUIRES_NEW, isolation = Isolation.READ_COMMITTED)
    public void acknowledge(String id, GenerationProviderReceipt receipt) {
        attempts.lock(id);
        var previous = get(id);
        if (!previous.submitted()) {
            throw GenerationJobException.invalid("尚未提交的准备记录不能接收供应商回执");
        }
        if (previous.providerTaskId() != null && !previous.providerTaskId().equals(receipt.providerTaskId())) {
            throw new GenerationJobException(409, "GENERATION_RECEIPT_CONFLICT", "供应商回执与已存标识不一致");
        }
        if (previous.acknowledged()) {
            if (!Objects.equals(previous.providerTaskId(), receipt.providerTaskId())
                    || !json.hash(previous.result()).equals(json.hash(receipt.result()))) {
                throw new GenerationJobException(409, "GENERATION_RECEIPT_CONFLICT", "供应商回执不能被替换");
            }
            return;
        }
        attempts.acknowledge(receipt.providerTaskId(), receipt.result() == null ? null : json.write(receipt.result()),
                id);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public RemoteGenerationJob applyReceipt(RemoteGenerationJob.Claim claim) {
        var job = jobs.leased(claim, true);
        var attempt = get(job.activeAttemptId());
        if (!attempt.acknowledged()) {
            throw new GenerationJobException(409, "UNKNOWN_SUBMISSION", "请求已经发出，但尚未取得受理回执");
        }
        remoteJobs.applyReceipt(attempt.providerTaskId(),
                attempt.result() == null ? null : json.write(attempt.result()),
                attempt.result() == null ? "POLLING" : "INGESTING", attempt.result() == null, job.id());
        return jobs.get(job.id());
    }

    public record Attempt(String id, String jobId, int number, ObjectNode request, String providerTaskId,
                          RemoteGenerationResult result, boolean acknowledged, boolean submitted, String resolution) {
    }

    public record Summary(String id, int number, String providerTaskId, boolean submitted, boolean acknowledged,
                          String resolution) {
    }
}
