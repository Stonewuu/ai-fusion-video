package com.stonewu.fusion.service.execution.recovery;

import com.stonewu.fusion.mapper.execution.ExecutionRecoveryDecisionMapper;
import com.stonewu.fusion.entity.execution.ExecutionRecoveryDecisionEntity;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.*;

@Service
@RequiredArgsConstructor
public class ExecutionRecoveryService {
    private final ExecutionRecoveryDecisionMapper decisions;
    private final UserMapper users;
    private final CanvasJson json;
    private final ExecutionRecoveryPolicy policy;
    private final ExecutionRecoveryTargets targets;
    private final ExecutionRecoveryCloser closer;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public ExecutionRecoveryData.Page list(long admin, int page) {
        policy.requireAdmin(admin, false);
        if (page < 0 || page > 1000000) {
            throw MediaException.invalid("任务页码无效");
        }
        var rows = targets.candidates(page);
        return new ExecutionRecoveryData.Page(rows.stream().limit(50).map(targets::get).map(policy::view).toList(),
                rows.size() > 50, page);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View get(Kind kind, String id, long admin) {
        policy.requireAdmin(admin, false);
        return policy.view(targets.get(new Ref(kind, id)));
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public List<History> history(long admin, int page) {
        policy.requireAdmin(admin, false);
        if (page < 0 || page > 1000000) {
            throw MediaException.invalid("核实记录页码无效");
        }
        return decisions.selectPage(new Page<ExecutionRecoveryDecisionEntity>(page + 1, 50, false),
                        Wrappers.<ExecutionRecoveryDecisionEntity>lambdaQuery()
                                .orderByDesc(ExecutionRecoveryDecisionEntity::getCreatedAt,
                                        ExecutionRecoveryDecisionEntity::getId))
                .getRecords().stream()
                .map(row -> new History(row.getId(), row.getAdminUserId(), Kind.valueOf(row.getExecutionKind()),
                        row.getJobId(), row.getActorUserId(), row.getNote(),
                        row.getCreatedAt().toInstant(ZoneOffset.UTC))).toList();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"canvasRun", "canvasTextJob", "canvasMcpJob", "canvasMediaJob", "mediaJob", "imageTask", "videoTask", "imageItems", "videoItems", "comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public Receipt confirm(Kind kind, String id, Decision decision, long admin) {
        if (decision.adminUserId() == null || decision.adminUserId() != admin) {
            throw new MediaException(409, "RECOVERY_IDENTITY_CHANGED", "管理员账号已变化，请重新核实当前任务");
        }
        if (!decision.providerFinished() || decision.note() == null || decision.note().isBlank() || decision.note()
                .length() > 2000
                || decision.requestId() == null || !decision.requestId().matches("[A-Za-z0-9._:-]{1,128}")) {
            throw MediaException.invalid("请确认所有原远端调用均已结束，并填写核实依据");
        }
        policy.requireAdmin(admin, false);
        var observed = targets.get(new Ref(kind, id));
        users.lockPair(admin, observed.actor().userId());
        policy.requireAdmin(admin, true);
        String hash = json.hash(Map.of("kind", kind, "id", id, "decision", decision));
        var prior = decisions.lockRequest(admin, decision.requestId());
        if (prior != null) {
            if (!hash.equals(prior.getRequestHash())) {
                throw new MediaException(409, "RECOVERY_REQUEST_CONFLICT", "同一核实请求不能用于不同的决定");
            }
            return json.read(prior.getReceiptJson(), Receipt.class);
        }
        var target = targets.lock(observed);
        if (decision.expectedRevision() == null || target.revision() != decision.expectedRevision()) {
            throw new MediaException(409, "RECOVERY_VERSION_CONFLICT", "任务有新状态或回执，请刷新后重新核实");
        }
        var facts = policy.view(target);
        if (!facts.factsHash().equals(decision.expectedFactsHash())) {
            throw new MediaException(409, "RECOVERY_EVIDENCE_CHANGED", "原调用有新的受理或终态证据，请刷新并重新核实");
        }
        if (!facts.canResolve()) {
            throw new MediaException(409, "RECOVERY_OWNER_AVAILABLE",
                    "此任务仍可由原发起人处理，或已经结束；管理员不能代为重试或采用内容");
        }
        String decisionId = UUID.randomUUID().toString();
        closer.close(target, admin, decisionId, decision.note().trim());
        var current = targets.get(new Ref(kind, id));
        var receipt = new Receipt(decisionId, kind, id, current.status(), current.revision());
        decisions.record(ExecutionRecoveryDecisionEntity.builder()
                .id(decisionId)
                .adminUserId(admin)
                .requestId(decision.requestId())
                .requestHash(hash)
                .executionKind(kind.name())
                .jobId(id)
                .expectedRevision(target.revision())
                .actorUserId(target.actor().userId())
                .actorTeamId(target.actor().teamId())
                .projectId(target.projectId())
                .canvasId(target.canvasId())
                .factsJson(json.write(facts))
                .note(decision.note().trim())
                .receiptJson(json.write(receipt)).build());
        return receipt;
    }
}
