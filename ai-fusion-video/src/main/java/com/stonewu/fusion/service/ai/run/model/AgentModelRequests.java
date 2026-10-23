package com.stonewu.fusion.service.ai.run.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.enums.ai.AgentModelCallStatus;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.repository.ai.AgentModelCallUsageRepository;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.model.AiModelRequestOptions;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.model.ChatUsage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentModelRequestEntity;
import com.stonewu.fusion.mapper.ai.AgentModelRequestMapper;
import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Objects;
import java.util.Optional;

/**
 * 模型调用在网络订阅前受理；与画布文本任务共用模型额度，租约丢失不能释放未知的远端调用。
 */
@Repository
@RequiredArgsConstructor
@Slf4j
public class AgentModelRequests {
    private final AgentModelRequestMapper requests;
    private final AgentModelCallUsageMapper usageRows;
    private final ExecutionCoordinationMapper coordination;
    private final AgentRunMapper runs;
    private final AgentModelCallUsageRepository usages;
    private final CanvasAccessPolicy access;
    private final CanvasDocumentRepository canvases;
    private final CanvasJson json;
    private final AiModelMetadataResolver modelMetadata;
    private final AiModelMapper models;
    private final AiProviderService providers;
    private final ObjectMapper mapper;

    public record Claim(String id, String runId, String owner, long epoch) {
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<Claim> admit(String id, AiModel model, RuntimeContext runtime, String fingerprint) {
        var context = runtime == null ? null : runtime.get(AgentRunContext.class);
        var user = runtime == null ? null : runtime.get(AuthenticatedUserContext.class);
        if (context == null || user == null || !Objects.equals(runtime.getUserId(), user.userId().toString())) {
            throw inactive();
        }
        var run = runs.selectByRunIdForUpdate(context.runId());
        var now = runs.selectDatabaseNow();
        if (run == null || !Objects.equals(run.getUserId(), user.userId()) || !"RUNNING".equals(run.getStatus())
                || !Objects.equals(run.getOwnerInstanceId(),
                context.ownerInstanceId()) || run.getOwnerEpoch() != context.ownerEpoch()
                || run.getCancelRequestedAt() != null || run.getLeaseUntil() == null || !run.getLeaseUntil()
                .isAfter(now)
                || !run.getDeadlineAt().isAfter(now)) {
            throw inactive();
        }
        CanvasActor actor = new CanvasActor(user.userId(), run.getActorTeamId());
        access.requireActorScope(actor);
        if (run.getCanvasScopeJson() != null) {
            var scope = json.read(run.getCanvasScopeJson(), CanvasAgentScope.class);
            var canvas = canvases.require(scope.canvasId(), false);
            if (!Objects.equals(canvas.projectId(), run.getProjectId())) {
                throw inactive();
            }
            if (scope.nodeRunId() == null) {
                access.requireView(canvas, actor);
            } else {
                access.requireExecute(canvas, actor);
            }
        }
        // 不允许在同一对话的原请求仍不确定时继续产生新的模型调用。
        long uncertain = requests.uncertainForConversation(run.getConversationId());
        if (uncertain != 0) {
            throw new BusinessException(409, "原模型调用仍需核实，请在任务详情中处理未完成的调用");
        }
        var currentModel = models.selectById(model.getId());
        if (currentModel != null && model.getConfig() != null) {
            currentModel = AiModelRequestOptions.withReasoningEffort(currentModel,
                    AiModelRequestOptions.reasoningEffort(json.read(model.getConfig(), JsonNode.class)), mapper);
        }
        if (fingerprint == null || currentModel == null || !fingerprint.equals(
                providers.agentScopeModelFingerprint(currentModel))) {
            throw new BusinessException(409, "模型或供应商配置在执行期间变化，已停止新增调用，请重新确认运行");
        }
        coordination.lockGenerationDispatch();
        var configured = models.selectOne(
                new LambdaQueryWrapper<AiModel>().select(AiModel::getId, AiModel::getMaxConcurrency)
                        .eq(AiModel::getId, model.getId()).eq(AiModel::getStatus, 1));
        if (configured == null || configured.getMaxConcurrency() == null || configured.getMaxConcurrency() < 1
                || model.getMaxConcurrency() == null || model.getMaxConcurrency() < 1) {
            throw new BusinessException(409, "模型已停用或并发配置不可用");
        }
        long busy = requests.countHeldForModel(model.getId());
        if (busy >= Math.min(configured.getMaxConcurrency(), model.getMaxConcurrency())) {
            return Optional.empty();
        }
        if (requests.exists(new LambdaQueryWrapper<AgentModelRequestEntity>().eq(AgentModelRequestEntity::getId, id))) {
            throw new BusinessException(409, "模型调用已受理，不能再次发送");
        }
        usages.startCall(run.getRunId(), id, modelMetadata.resolve(model).modelProtocol(), model.getCode());
        requests.admit(id, run.getRunId(), model.getId(), context.ownerInstanceId(), context.ownerEpoch());
        return Optional.of(new Claim(id, run.getRunId(), context.ownerInstanceId(), context.ownerEpoch()));
    }

    @Transactional
    public void completed(Claim claim, ChatUsage usage) {
        String status = lock(claim);
        if (status.equals("COMPLETED") || status.equals("CANCELLED")) {
            return;
        }
        // 即使任务已失去租约，也只登记这个原调用的用量和远端结束证据，不发布业务结果。
        usageRows.completeObserved(usage == null ? null : usage.getInputTokens(),
                usage == null ? null : usage.getOutputTokens(),
                usage == null ? null : usage.getCachedTokens(), usage == null ? null : json.write(usage), claim.runId(),
                claim.id());
        requests.completed(claim.id());
    }

    @Transactional
    public void failed(Claim claim) {
        String status = lock(claim);
        if (status.equals("COMPLETED") || status.equals("CANCELLED")) {
            return;
        }
        usages.failCall(claim.runId(), claim.id(), AgentModelCallStatus.FAILED);
        // COMPLETED describes the request lifecycle; the call and Agent run remain FAILED.
        requests.completed(claim.id());
    }

    @Transactional
    public void uncertain(Claim claim, boolean cancelled) {
        if (!lock(claim).equals("RUNNING")) {
            return;
        }
        usages.failCall(claim.runId(), claim.id(),
                cancelled ? AgentModelCallStatus.CANCELLED : AgentModelCallStatus.FAILED);
        requests.uncertain(claim.id());
    }

    private String lock(Claim claim) {
        String status = requests.lockRequest(claim.id(), claim.runId(), claim.owner(), claim.epoch());
        if (status == null) {
            throw inactive();
        }
        return status;
    }

    @Scheduled(fixedDelay = 5000)
    public void reconcile() {
        try {
            requests.reconcile();
        } catch (RuntimeException failure) {
            log.warn("智能体模型调用核对暂未完成: {}", failure.getClass().getSimpleName());
        }
    }

    private BusinessException inactive() {
        return new BusinessException(409, "智能体运行身份已失效，不能发起或改写模型调用");
    }
}
