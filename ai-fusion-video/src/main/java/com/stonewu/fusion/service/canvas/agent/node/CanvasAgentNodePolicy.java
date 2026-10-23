package com.stonewu.fusion.service.canvas.agent.node;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.resource.ResourceMutationObserver;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeScope.Entry;
import static com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeScope.key;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 同一业务事务内执行；老工具、子 Agent 和结果写回也必须遵守专业节点的范围与版本。
 */
@Component
@RequiredArgsConstructor
public class CanvasAgentNodePolicy implements ResourceMutationObserver {
    private final CanvasAgentNodeRepository jobs;
    private final AgentRunRepository agents;
    private final ResourceQueryService resources;
    private final CanvasJson json;

    public CanvasAgentNodeRepository.Job job(ResourceCommandContext context, boolean lock) {
        if (context.agentRunId() == null) {
            return null;
        }
        var run = agents.findRun(context.agentRunId());
        if (run == null || run.getCanvasScopeJson() == null) {
            return null;
        }
        var scope = json.read(run.getCanvasScopeJson(), CanvasAgentScope.class);
        if (scope.nodeRunId() == null) {
            return null;
        }
        if (!Objects.equals(scope.canvasId(), context.canvasId()) || !Objects.equals(run.getUserId(),
                context.actor().userId())
                || !Objects.equals(run.getActorTeamId(), context.actor().teamId())) {
            throw new BusinessException(403, "专业智能体的身份或画布范围不一致");
        }
        var job = jobs.find(scope.nodeRunId(), lock);
        if (job == null || job.cancelRequested()) {
            throw new BusinessException(409, "专业智能体任务不存在或已停止");
        }
        return job;
    }

    @Override
    public void before(ResourceCommand command, ResourceCommandContext context) {
        var job = job(context, true);
        if (job == null) {
            return;
        }
        var scope = job.scope();
        if (command instanceof ResourceCreateCommand create) {
            create(scope, command.resourceType(), create.fields());
        } else if (command instanceof AssetBatchCreateCommand batch) {
            if (!scope.createTypes().contains(batch.resourceType())) {
                deny();
            }
            if (batch.resourceType() == ASSET_ITEM) {
                writable(scope, ASSET, Objects.requireNonNull(batch.assetId()));
            }
        } else if (command instanceof StoryboardBatchPatchCommand batch) {
            for (var entry : batch.entries()) {
                writable(scope, entry.resourceType(), entry.resourceId());
            }
        } else {
            var encoded = json.object(command);
            if (!encoded.path("resourceId").isIntegralNumber()) {
                deny();
            }
            writable(scope, command.resourceType(), encoded.path("resourceId").asLong());
        }
    }

    private void create(CanvasAgentNodeScope scope, CanvasValueType type, ObjectNode fields) {
        if (!scope.createTypes().contains(type)) {
            deny();
        }
        switch (type) {
            case SCRIPT_EPISODE -> writable(scope, SCRIPT, fields.path("scriptId").asLong());
            case SCRIPT_SCENE -> writable(scope, SCRIPT_EPISODE, fields.path("episodeId").asLong());
            case STORYBOARD_EPISODE -> {
                writable(scope, STORYBOARD, fields.path("storyboardId").asLong());
                if (scope.agentType().equals("episode_storyboard_writer")) {
                    var source = scope.resources().get(key(SCRIPT_EPISODE, fields.path("scriptEpisodeId").asLong()));
                    if (source == null || source.deleted()) {
                        deny();
                    }
                }
            }
            case STORYBOARD_SCENE -> writable(scope, STORYBOARD_EPISODE, fields.path("episodeId").asLong());
            case SHOT -> writable(scope, STORYBOARD_SCENE, fields.path("storyboardSceneId").asLong());
            case ASSET_ITEM -> writable(scope, ASSET, fields.path("assetId").asLong());
            case ASSET -> {
            }
            default -> deny();
        }
    }

    public void writable(CanvasAgentNodeScope scope, CanvasValueType type, long id) {
        var entry = scope.resources().get(key(type, id));
        if (entry == null || !entry.writable() || entry.deleted()) {
            deny();
        }
    }

    @Override
    public void validateAndApply(ResourceMutation mutation, ResourceCommandContext context) {
        var job = job(context, true);
        if (job == null) {
            return;
        }
        var entries = new LinkedHashMap<>(job.scope().resources());
        for (var change : mutation.changes()) {
            String key = key(change.resourceType(), change.resourceId());
            var old = entries.get(key);
            boolean created = old == null && change.version() == 0 && !change.deleted();
            if (created) {
                if (!job.scope().createTypes().contains(change.resourceType())) {
                    deny();
                }
                // 集合命令可一次创建多层子对象；父对象必须来自原范围或本次同一事务的已创建对象。
                if (change.resourceType() != ASSET) {
                    var value = resources.get(change.resourceType(), change.resourceId(), context.actor());
                    create(job.scope().withResources(entries), change.resourceType(), value.data());
                }
            } else if (old == null || old.deleted()) {
                deny();
            } else if (change.version() != old.version() + 1) {
                throw CanvasRunException.conflict(
                        "专业智能体执行期间业务内容已被其他操作修改，已保留新版本，请重新确认输入范围");
            }
            entries.put(key,
                    new Entry(change.resourceType(), change.resourceId(), change.version(), created || old.writable(),
                            true, change.deleted()));
        }
        if (entries.size() > 3000) {
            throw CanvasRunException.invalid("本次专业智能体已达到 3000 个业务对象上限，请拆分创作");
        }
        jobs.scope(job.nodeRunId(), job.scope().withResources(entries));
    }

    private void deny() {
        throw new BusinessException(403, "此业务修改超出了专业画布节点确认的输入范围");
    }
}
