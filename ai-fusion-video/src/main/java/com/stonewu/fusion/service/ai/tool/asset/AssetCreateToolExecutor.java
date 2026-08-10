package com.stonewu.fusion.service.ai.tool.asset;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.asset.AssetService;
import com.stonewu.fusion.service.project.ProjectService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 创建单个资产工具（create_asset）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AssetCreateToolExecutor implements ToolExecutor {

    private final AssetService assetService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "create_asset";
    }

    @Override
    public String getDisplayName() {
        return "创建资产";
    }

    @Override
    public String getToolDescription() {
        return """
                创建新资产（角色/场景/道具），用于管理项目中的各种创作元素。
                创建时应尽可能补充完整的信息，特别是：
                - 角色（character）：外貌（appearance）、性格（personality）、年龄（age）、身高体型等
                - 场景（scene）：地点描述、时间、氛围、空间布局等
                - 道具（prop）：用途、外观特征、尺寸材质等

                这些信息应填入 properties 字段（key-value 键值对），方便后续生图时使用。

                **重要**：
                1. name 不要重复，创建前建议先用 list_project_assets 查看已有资产
                2. 如需批量创建，请使用 batch_create_assets 工具
                3. 创建资产后会自动生成一个与资产同名的初始子资产（itemType=initial），无需手动创建

                **角色命名规范**：
                - 每个角色资产只使用一个名称，以该角色在剧情中最主要/最常出现的名字为准
                - 禁止用"/"或其他分隔符连接多个名称（如"王德发/花非烟"是错误的）
                - 穿越、重生、变身等导致角色名称变化的情况，只取在剧情中出场最多的那个名字
                - 如果某个角色前后各占一半，以剧本开篇使用的名字为准
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {
                            "type": "integer",
                            "description": "项目ID（必填）"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["character", "scene", "prop"],
                            "description": "资产类型（必填）"
                        },
                        "name": {
                            "type": "string",
                            "description": "资产名称（必填），不要与已有资产重复"
                        },
                        "description": {
                            "type": "string",
                            "description": "资产描述（可选），详细描述该资产的特征"
                        },
                        "properties": {
                            "type": "object",
                            "additionalProperties": { "type": "string" },
                            "description": "自定义属性键值对（可选），例如 appearance=黑发高个、age=25"
                        },
                        "initialItem": {
                            "type": "object",
                            "description": "初始子资产内容（可选），用于填充自动创建的初始子资产",
                            "properties": {
                                "name": { "type": "string", "description": "子资产名称，应使用具体有意义的名称（如角色名、场景名等）" },
                                "itemType": { "type": "string", "description": "子资产类型：original/front/side/back/detail/expression/pose/variant" },
                                "imageUrl": { "type": "string", "description": "子资产图片URL" },
                                "properties": { "type": "object", "additionalProperties": { "type": "string" }, "description": "子资产自定义属性" }
                            }
                        }
                    },
                    "required": ["type", "name", "projectId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long projectId = params.getLong("projectId");
            String type = params.getStr("type");
            String name = params.getStr("name");

            if (projectId == null || type == null || name == null) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "缺少必要参数: projectId, type, name").toString();
            }

            Long userId = context.getUserId();

            if (!projectService.canAccessProject(projectId, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "无权访问该项目").toString();
            }

            Asset asset = Asset.builder()
                    .projectId(projectId)
                    .type(type)
                    .name(name)
                    .description(params.getStr("description"))
                    .properties(params.containsKey("properties") ? params.getJSONObject("properties").toString() : null)
                    .sourceType(2)
                    .userId(userId)
                    .build();

            Asset saved = assetService.create(asset);

            // 如果传入了 initialItem，用其数据更新自动创建的初始子资产
            JSONObject initialItemData = params.getJSONObject("initialItem");
            if (initialItemData != null) {
                List<AssetItem> items = assetService.listItems(saved.getId());
                if (!items.isEmpty()) {
                    AssetItem first = items.get(0);
                    if (initialItemData.getStr("name") != null) {
                        first.setName(initialItemData.getStr("name"));
                    }
                    if (initialItemData.getStr("itemType") != null) {
                        first.setItemType(initialItemData.getStr("itemType"));
                    }
                    if (initialItemData.getStr("imageUrl") != null) {
                        first.setImageUrl(initialItemData.getStr("imageUrl"));
                    }
                    if (initialItemData.containsKey("properties")) {
                        first.setProperties(initialItemData.getJSONObject("properties").toString());
                    }
                    assetService.updateItem(first);
                }
            }

            return JSONUtil.createObj()
                    .set("assetId", saved.getId())
                    .set("type", type)
                    .set("name", name)
                    .set("message", String.format("资产 \"%s\" 创建成功", name)).toString();
        } catch (Exception e) {
            log.error("创建资产失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "创建失败: " + e.getMessage()).toString();
        }
    }
}
