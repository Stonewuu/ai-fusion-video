package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.entity.generation.RemoteGenerationJobEntity;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowCapabilities;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeGeneration;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.VideoModelRules;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategyRouter;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategyRouter;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 原页面、画布和 Agent 共用的受理事务：任务、条目、远端记录和投递一起提交。
 */
@Service
@RequiredArgsConstructor
public class GenerationSubmissionService {
    private final GenerationSubmissionMapper submissions;
    private final RemoteGenerationJobMapper remoteJobs;
    private final CanvasJson json;
    private final GenerationExecutionContexts contexts;
    private final GenerationModelSnapshot models;
    private final GenerationModelCapabilityService capabilities;
    private final ComfyUiWorkflowService workflows;
    private final ImageGenerationService images;
    private final VideoGenerationService videos;
    private final GenerationJobRepository jobs;
    private final GenerationOutbox outbox;
    private final GenerationRequestRepository requests;
    private final ImageGenerationStrategyRouter imageProtocols;
    private final VideoGenerationStrategyRouter videoProtocols;
    private final GenerationInputVersions inputVersions;
    private final CanvasAgentNodeGeneration agentNodes;

    @CacheEvict(value = {"imageTask", "imageItems"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public GenerationSubmission image(ImageTask task, String requestId, ResourceCommandContext context,
                                      String nodeRunId) {
        return submitImage(task, requestId, context, nodeRunId, null, null);
    }

    @CacheEvict(value = {"imageTask", "imageItems"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public GenerationSubmission imageFrozen(ImageTask task, String requestId, ResourceCommandContext context,
                                            String nodeRunId, AiModel model, String apiHash) {
        return submitImage(task, requestId, context, nodeRunId, model, apiHash);
    }

    private GenerationSubmission submitImage(ImageTask task, String requestId, ResourceCommandContext context,
                                             String nodeRunId, AiModel frozen, String apiHash) {
        String hash = json.hash(Map.of("input", task, "nodeRunId", nodeRunId == null ? "" : nodeRunId));
        contexts.prepare(task, context);
        var existing = existing(requestId, hash, GenerationKind.IMAGE, context);
        if (existing != null) {
            return existing;
        }
        validateCommon(task.getPrompt(), task.getCount());
        var model = frozen == null ? freeze(task.getModelId(), GenerationKind.IMAGE, task.getWorkflowVersionId())
                : models.frozen(frozen, GenerationKind.IMAGE, apiHash);
        if (!model.getId().equals(task.getModelId())) {
            throw GenerationJobException.invalid("冻结模型与生成输入不符");
        }
        task.setWorkflowVersionId(version(model, task.getWorkflowVersionId()));
        capabilities.validateImageTask(model, task);
        nodeRunId = agentNodes.admit(context, GenerationKind.IMAGE, model, task.getCount(), null, nodeRunId);
        inputVersions.freeze(task, context.actor());
        task.setExecutionContextJson(json.write(GenerationExecutionIdentity.worker(context)));
        task.setTaskId(UUID.randomUUID().toString());
        task.setStatus(0);
        task.setSuccessCount(0);
        task.setId(null);
        images.create(task);
        String submissionId = insert(GenerationKind.IMAGE, task.getId(), task.getTaskId(), task.getProjectId(),
                requestId, hash, context, nodeRunId, model);
        var itemIds = new ArrayList<Long>();
        for (int index = 0; index < task.getCount(); index++) {
            var item = images.createItem(ImageItem.builder().taskId(task.getId()).status(0).build());
            itemIds.add(item.getId());
        }
        int batch = imageProtocols.resolve(model, models.requireApi(model.getApiConfigId()))
                .batchSize(task, model, models.requireApi(model.getApiConfigId()));
        insertJobs(submissionId, model.getId(), itemIds, batch, json.object(task), context, task.getProjectId());
        accepted(submissionId);
        return jobs.submission(submissionId);
    }

    @CacheEvict(value = {"videoTask", "videoItems"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public GenerationSubmission video(VideoTask task, String requestId, ResourceCommandContext context,
                                      String nodeRunId) {
        return submitVideo(task, requestId, context, nodeRunId, null, null);
    }

    @CacheEvict(value = {"videoTask", "videoItems"}, allEntries = true)
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public GenerationSubmission videoFrozen(VideoTask task, String requestId, ResourceCommandContext context,
                                            String nodeRunId, AiModel model, String apiHash) {
        return submitVideo(task, requestId, context, nodeRunId, model, apiHash);
    }

    private GenerationSubmission submitVideo(VideoTask task, String requestId, ResourceCommandContext context,
                                             String nodeRunId, AiModel frozen, String apiHash) {
        String hash = json.hash(Map.of("input", task, "nodeRunId", nodeRunId == null ? "" : nodeRunId));
        contexts.prepare(task, context);
        var existing = existing(requestId, hash, GenerationKind.VIDEO, context);
        if (existing != null) {
            return existing;
        }
        var model = frozen == null ? freeze(task.getModelId(), GenerationKind.VIDEO, task.getWorkflowVersionId())
                : models.frozen(frozen, GenerationKind.VIDEO, apiHash);
        var modelConfig = capabilities.getMergedModelConfig(model);
        var rules = VideoModelRules.from(modelConfig);
        validateCommon(task.getPrompt(), task.getCount(),
                rules != null && !Boolean.TRUE.equals(rules.promptRequired()));
        if (!model.getId().equals(task.getModelId())) {
            throw GenerationJobException.invalid("冻结模型与生成输入不符");
        }
        task.setWorkflowVersionId(version(model, task.getWorkflowVersionId()));
        if (task.getWatermark() == null && modelConfig.getBool("supportWatermark", true)) {
            task.setWatermark(false);
        }
        if (task.getGenerateAudio() == null
                && capabilities.getMergedModelConfig(model).getBool("supportGenerateAudio", true)) {
            task.setGenerateAudio(
                    rules != null && rules.defaultGenerateAudio() != null ? rules.defaultGenerateAudio() : true);
        }
        capabilities.validateVideoTask(model, task);
        Integer budgetDuration = rules == null ? task.getDuration() : Integer.valueOf(
                VideoModelRules.budgetSeconds(modelConfig, task.getDuration()));
        nodeRunId = agentNodes.admit(context, GenerationKind.VIDEO, model, task.getCount(), budgetDuration, nodeRunId);
        inputVersions.freeze(task, context.actor());
        inputVersions.validateVideoMedia(task, modelConfig, context.actor());
        task.setExecutionContextJson(json.write(GenerationExecutionIdentity.worker(context)));
        task.setTaskId(UUID.randomUUID().toString());
        task.setStatus(0);
        task.setSuccessCount(0);
        task.setId(null);
        videos.create(task);
        String submissionId = insert(GenerationKind.VIDEO, task.getId(), task.getTaskId(), task.getProjectId(),
                requestId, hash, context, nodeRunId, model);
        var itemIds = new ArrayList<Long>();
        for (int index = 0; index < task.getCount(); index++) {
            var item = videos.createItem(VideoItem.builder().taskId(task.getId()).status(0).build());
            itemIds.add(item.getId());
        }
        int batch = videoProtocols.resolve(model).batchSize(task, model, models.requireApi(model.getApiConfigId()));
        insertJobs(submissionId, model.getId(), itemIds, batch, json.object(task), context, task.getProjectId());
        accepted(submissionId);
        return jobs.submission(submissionId);
    }

    private GenerationSubmission existing(String requestId, String hash, GenerationKind kind,
                                          ResourceCommandContext context) {
        requests.lock(requestId, context);
        // Agent 按原工具调用身份复用结果，不能通过改写请求编码重复发起付费生成。
        var row = "agent".equals(context.origin())
                ? submissions.selectAgentInvocation(requests.scope(context), context.agentRunId(), context.toolCallId())
                : submissions.selectOne(new LambdaQueryWrapper<GenerationSubmissionEntity>()
                .select(GenerationSubmissionEntity::getId, GenerationSubmissionEntity::getRequestHash,
                        GenerationSubmissionEntity::getKind)
                .eq(GenerationSubmissionEntity::getScopeKey, requests.scope(context))
                .eq(GenerationSubmissionEntity::getRequestId, requestId));
        if (row == null) {
            requests.requireNotWithdrawn(requestId, context);
            return null;
        }
        if (!row.getRequestHash().equals(hash) || !row.getKind().equals(kind.name())) {
            throw new GenerationJobException(409, "GENERATION_IDEMPOTENCY_CONFLICT", "此请求标识已用于不同的生成内容");
        }
        var previous = jobs.submission(row.getId());
        contexts.requireRead(previous.kind(), previous.taskId(), context.actor());
        return previous;
    }

    private String insert(GenerationKind kind, long taskId, String publicId, Long projectId, String requestId,
                          String hash, ResourceCommandContext context, String nodeRunId, AiModel model) {
        String id = UUID.randomUUID().toString();
        String apiHash = models.fingerprint(models.requireApi(model.getApiConfigId()));
        submissions.insertQueued(GenerationSubmissionEntity.builder().id(id).requestId(requestId)
                .scopeKey(requests.scope(context)).requestHash(hash).kind(kind.name()).taskId(taskId)
                .publicTaskId(publicId)
                .actorUserId(context.actor().userId()).actorTeamId(context.actor().teamId()).projectId(projectId)
                .canvasNodeRunId(nodeRunId).contextJson(json.write(context)).modelSnapshotJson(json.write(model))
                .apiConfigId(model.getApiConfigId()).apiConfigHash(apiHash).build());
        return id;
    }

    private void insertJobs(String submissionId, long modelId, List<Long> itemIds, int batchSize, ObjectNode original,
                            ResourceCommandContext context, Long projectId) {
        if (batchSize < 1 || batchSize > itemIds.size()) {
            throw GenerationJobException.invalid("协议返回了无效的批量大小");
        }
        for (int start = 0; start < itemIds.size(); start += batchSize) {
            var group = itemIds.subList(start, Math.min(itemIds.size(), start + batchSize));
            var input = original.deepCopy();
            input.put("count", group.size());
            var values = input.putArray("_resultItemIds");
            group.forEach(values::add);
            insertJob(submissionId, modelId, group.getFirst(), start, input, context, projectId);
        }
    }

    private void insertJob(String submissionId, long modelId, long itemId, int index, ObjectNode input,
                           ResourceCommandContext context, Long projectId) {
        remoteJobs.insertQueued(RemoteGenerationJobEntity.builder().id(UUID.randomUUID().toString())
                .submissionId(submissionId).itemId(itemId).itemIndex(index).modelId(modelId)
                .actorUserId(context.actor().userId()).projectId(projectId).inputJson(json.write(input)).build());
    }

    private void accepted(String id) {
        outbox.enqueue("generation:" + id + ":1", "GENERATION_ACCEPTED", id, GenerationOutbox.state(id, 1, "QUEUED"));
    }

    private AiModel freeze(Long modelId, GenerationKind kind, Long versionId) {
        return versionId == null ? models.freeze(modelId, kind) : models.freeze(modelId, kind, versionId);
    }

    private Long version(AiModel model, Long requested) {
        if (model.getComfyuiWorkflowId() == null) {
            if (requested != null) {
                throw GenerationJobException.invalid("普通模型不能指定 ComfyUI 工作流版本");
            }
            return null;
        }
        var workflow = workflows.requireWorkflow(model.getComfyuiWorkflowId());
        if (!Integer.valueOf(1).equals(workflow.getStatus())) {
            throw GenerationJobException.invalid("ComfyUI 工作流已停用");
        }
        Long frozenVersion = Boolean.TRUE.equals(model.getGenerationConfigSnapshot())
                ? JSONUtil.parseObj(model.getConfig()).getLong(ComfyUiWorkflowCapabilities.VERSION_KEY) : null;
        if (frozenVersion != null && requested != null && !frozenVersion.equals(requested)) {
            throw GenerationJobException.invalid("工作流版本与已冻结的模型能力不一致");
        }
        Long id = frozenVersion != null ? frozenVersion : requested == null ? workflow.getActiveVersionId() : requested;
        if (id == null) {
            throw GenerationJobException.invalid("ComfyUI 工作流没有发布版本");
        }
        var version = workflows.requireVersion(id);
        if (!Boolean.TRUE.equals(version.getPublished()) || !version.getWorkflowId().equals(workflow.getId())) {
            throw GenerationJobException.invalid("ComfyUI 版本未发布或不属于所选模型");
        }
        return id;
    }

    private void validateCommon(String prompt, Integer count) {
        validateCommon(prompt, count, false);
    }

    private void validateCommon(String prompt, Integer count, boolean mediaPromptOptional) {
        if (!mediaPromptOptional && (prompt == null || prompt.isBlank()) || prompt != null && prompt.length() > 100_000) {
            throw GenerationJobException.invalid("请填写有效的创作要求，最多 100000 字符");
        }
        if (count == null || count < 1 || count > 32) {
            throw GenerationJobException.invalid("一次生成数量必须为 1 至 32");
        }
    }

}
