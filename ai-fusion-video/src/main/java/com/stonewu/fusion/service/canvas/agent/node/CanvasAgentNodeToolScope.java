package com.stonewu.fusion.service.canvas.agent.node;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.AgentToolInvocationPolicy;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class CanvasAgentNodeToolScope implements AgentToolInvocationPolicy {
    private final CanvasAgentNodePolicy policy;
    private final ResourceQueryService resources;
    private final CanvasJson json;
    private final GenerationSubmissionMapper submissions;

    @Override
    public void before(String toolName, Map<String, Object> input, ToolExecutionContext context) {
        if (context.getCanvasScope() == null || context.getCanvasScope().nodeRunId() == null) {
            return;
        }
        var job = policy.job(context.requireMutationContext(), false);
        if (job == null) {
            throw new BusinessException(403, "专业节点的持久化授权不存在");
        }
        if (!CanvasAgentNodeTypes.permittedTool(toolName)) {
            throw new BusinessException(403, "此工具尚未登记专业画布节点的作用域协议");
        }
        if (json.write(input).length() > 512 * 1024) {
            throw new BusinessException(400, "专业任务的单次工具输入超过 512 KiB，请拆分修改");
        }
        if (toolName.equals("list_project_assets") && !input.containsKey("projectId")) {
            throw new BusinessException(400, "画布专业节点查询资产时必须明确使用当前 projectId");
        }
        check(toolName, json.object(input), context, 0);
        if (toolName.equals("get_generation_task") || toolName.equals("await_generation_task")) {
            long count = submissions.selectCount(new LambdaQueryWrapper<GenerationSubmissionEntity>()
                    .eq(GenerationSubmissionEntity::getId, input.get("submissionId"))
                    .eq(GenerationSubmissionEntity::getCanvasNodeRunId, job.nodeRunId()));
            if (count != 1) {
                throw new BusinessException(403, "只能查询本专业节点实际提交的生成任务");
            }
        }
    }

    private void check(String name, JsonNode input, ToolExecutionContext context, int depth) {
        if (depth > 24) {
            throw new BusinessException(400, "工具参数嵌套过深");
        }
        if (input.isArray()) {
            for (var value : input) {
                check(name, value, context, depth + 1);
            }
        } else if (input.isObject()) {
            input.fields().forEachRemaining(field -> {
                String key = field.getKey();
                if (key.equals("projectId") && !field.getValue().isNull()) {
                    if (!field.getValue().isIntegralNumber() || !Objects.equals(context.getCanvasScope().projectId(),
                            field.getValue().asLong())) {
                        throw new BusinessException(403, "工具不能跨越当前画布的项目范围");
                    }
                }
                var type = type(name, key.endsWith("Ids") ? key.substring(0, key.length() - 1) : key);
                if (type != null && !field.getValue().isNull()) {
                    if (name.equals("storyboard_asset_preprocessor") && key.equals(
                            "scriptEpisodeIds") && field.getValue()
                            .isTextual()) {
                        String[] ids = field.getValue().asText().split(",", -1);
                        if (ids.length == 0 || ids.length > 200) {
                            throw new BusinessException(400, "子资产预处理每次最多处理 200 集");
                        }
                        for (String id : ids) {
                            try {
                                read(type, json.object(Map.of("id", Long.parseLong(id.strip()))).path("id"), context);
                            } catch (NumberFormatException invalid) {
                                throw new BusinessException(400, "分集列表必须是逗号分隔的正整数 ID");
                            }
                        }
                    } else if (field.getValue().isArray()) {
                        for (var id : field.getValue()) {
                            read(type, id, context);
                        }
                    } else {
                        read(type, field.getValue(), context);
                    }
                } else {
                    check(name, field.getValue(), context, depth + 1);
                }
            });
        }
    }

    private void read(CanvasValueType type, JsonNode value, ToolExecutionContext context) {
        if (!value.isIntegralNumber() || value.asLong() <= 0) {
            throw new BusinessException(400, "业务 ID 必须为正整数");
        }
        var resource = resources.get(type, value.asLong(), context.requireMutationContext().actor());
        if (!Objects.equals(resource.header().projectId(), context.getCanvasScope().projectId())) {
            throw new BusinessException(403, "业务对象不属于当前画布项目");
        }
    }

    private CanvasValueType type(String tool, String name) {
        return switch (name) {
            case "scriptId", "sourceScriptId" -> SCRIPT;
            case "scriptEpisodeId", "sourceEpisodeId" -> SCRIPT_EPISODE;
            case "scriptSceneId", "sourceSceneId" -> SCRIPT_SCENE;
            case "storyboardId" -> STORYBOARD;
            case "storyboardEpisodeId" -> STORYBOARD_EPISODE;
            case "storyboardSceneId" -> STORYBOARD_SCENE;
            case "shotId", "storyboardItemId" -> SHOT;
            case "assetId", "characterAssetId", "sceneAssetId", "propAssetId" -> ASSET;
            case "assetItemId" -> ASSET_ITEM;
            case "episodeId" -> tool.contains("storyboard") ? STORYBOARD_EPISODE : SCRIPT_EPISODE;
            case "sceneId" -> tool.contains("storyboard") ? STORYBOARD_SCENE : SCRIPT_SCENE;
            case "itemId" -> tool.contains("asset") ? ASSET_ITEM : tool.contains("storyboard") ? SHOT : null;
            default -> null;
        };
    }
}
