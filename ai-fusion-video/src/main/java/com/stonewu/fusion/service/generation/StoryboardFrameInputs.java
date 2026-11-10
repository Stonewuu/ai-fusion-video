package com.stonewu.fusion.service.generation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.resource.ResourceMutations;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardFields;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/** 只读取并固定业务输入；生成受理在此事务结束后执行。 */
@Component
@RequiredArgsConstructor
public class StoryboardFrameInputs {
    private final ResourceQueryService resources;
    private final AgentRunMapper runs;

    public boolean appliesTo(ToolExecutionContext context) {
        var invocation = context.requireMutationContext();
        if (!"agent".equals(invocation.origin())) {
            return false;
        }
        var run = runs.selectByRunId(invocation.agentRunId());
        if (run == null) {
            throw new IllegalStateException("智能体运行不存在");
        }
        return "storyboard_frame_executor".equals(run.getAgentType());
    }

    public record Reference(String name, String role, String imageUrl) {
    }

    public record Prepared(List<Reference> references, int shotVersion, int sceneAssetVersion) {
        public String prompt(String body) {
            List<String> guide = new ArrayList<>();
            for (int index = 0; index < references.size(); index++) {
                var ref = references.get(index);
                guide.add("图片" + (index + 1) + "：" + ref.role() + "「" + ref.name() + "」。");
            }
            return "参考素材：\n" + String.join("\n", guide)
                    + "\n场景图固定地点、布局与环境；人物图固定身份与服装；道具图固定形状、材质和数量。"
                    + "只绘制正文要求入画的主体，移除设定图白底，不照搬参考图的姿势、景别和构图。\n\n画面要求：\n" + body;
        }
    }

    @Transactional(readOnly = true)
    public Prepared prepare(long shotId, int expectedVersion, int expectedSceneAssetVersion,
                            String frameType, CanvasActor actor) {
        var shot = resources.get(SHOT, shotId, actor);
        ResourceMutations.requireVersion(shot.header(), expectedVersion);
        ObjectNode data = shot.data();
        if (!data.hasNonNull("sceneAssetItemId")) {
            throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "请先为镜头关联场景并生成场景参考图");
        }
        Map<String, Reference> references = new LinkedHashMap<>();
        if ("last".equals(frameType) && data.hasNonNull("firstFrameImageUrl")) {
            add(references, "本镜头首帧", "首帧，保留人物、场景与道具，按正文推进到结尾状态", data.path("firstFrameImageUrl").asText());
        }
        long sceneId = StoryboardFields.id(data.get("sceneAssetItemId"), "sceneAssetItemId");
        var scene = resources.get(ASSET_ITEM, sceneId, actor);
        ResourceMutations.requireVersion(scene.header(), expectedSceneAssetVersion);
        asset(references, scene.data(), shot.header().projectId(), "scene", "场景", actor);
        for (String key : List.of("characterIds", "propIds")) {
            for (JsonNode id : data.path(key)) {
                var item = resources.get(ASSET_ITEM, StoryboardFields.id(id, key), actor);
                asset(references, item.data(), shot.header().projectId(),
                        key.equals("characterIds") ? "character" : "prop",
                        key.equals("characterIds") ? "人物" : "道具", actor);
            }
        }
        return new Prepared(List.copyOf(references.values()), expectedVersion, expectedSceneAssetVersion);
    }

    private void asset(Map<String, Reference> references, ObjectNode item, Long projectId,
                       String type, String role, CanvasActor actor) {
        var asset = resources.get(ASSET, item.path("assetId").longValue(), actor);
        if (!Objects.equals(asset.header().projectId(), projectId) || !type.equals(asset.data().path("type").asText())) {
            throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "镜头关联资产的项目或类型不匹配");
        }
        String name = item.path("name").asText();
        String image = item.path("imageUrl").asText("");
        if (image.isBlank()) {
            throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "请先生成" + role + "「" + name + "」的参考图");
        }
        add(references, name, role, image);
    }

    private void add(Map<String, Reference> references, String name, String role, String image) {
        references.putIfAbsent(image, new Reference(name, role, image));
    }
}
