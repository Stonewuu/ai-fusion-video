package com.stonewu.fusion.service.ai.tool.asset;

import cn.hutool.json.JSONArray;
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

import java.util.ArrayList;
import java.util.List;

/**
 * 批量创建资产工具（batch_create_assets）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class BatchCreateAssetsToolExecutor implements ToolExecutor {

    /**
     * 单次批量创建资产的最大数量
     */
    private static final int MAX_BATCH_SIZE = 10;

    private final AssetService assetService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "batch_create_assets";
    }

    @Override
    public String getDisplayName() {
        return "批量创建资产";
    }

    @Override
    public String getToolDescription() {
        return String.format(
                """
                        从剧本中提取角色、场景、道具等信息并批量创建资产。

                        **使用流程**：
                        1. 先使用 list_project_assets 查看已有资产
                        2. 调用 query_asset_metadata 获取各类型资产的属性字段定义及填写指南（fieldDescription）
                        3. 调用此工具创建缺失的资产（工具自动按 project_id + type + name 去重，已有资产会被复用）
                        4. 创建完成后，可使用 add_asset_item 为资产添加参考图

                        **注意事项**：
                        - 单次最多创建 %d 个资产，超出请分次调用
                        - assets 数组中每项必须包含 type 和 name
                        - 支持的资产类型：character（角色）、scene（场景）、prop（道具）
                        - properties 必须包含 query_asset_metadata 返回的**所有** fieldKey，每个 key 都必须填写，确实无法确定的填'无'。填写内容严格遵循对应 fieldDescription 中的要求
                        - 每个资产必须提供 initialItem 填充初始子资产的 name 和 properties（properties 同样须包含所有 fieldKey，无值填'无'）

                        ## 角色资产命名规范（最高优先级）

                        - 每个角色资产只使用一个名称，以该角色在剧情中最主要/最常出现的名字为准
                        - 禁止用"/"或其他分隔符连接多个名称（如"王德发/花非烟"是错误的，应只写"王德发"或只写"花非烟"）
                        - 穿越、重生、变身、性别转换等情节导致角色名称变化时，只用剧中出场最多的那个名字
                        - 如果前后名字出场各占一半，以剧本开篇使用的名字为准
                        - 在 description 中可以补充说明角色的其他名字/身份（如"王德发，穿越后名花非烟"）

                        ## 角色类资产 description 描述规范

                        描述要求突出角色特色，有细节质感，使用中文 80-150 字：
                        - 面部：脸型、五官特征（如高挺鼻梁、深邃眼窝、薄唇等）
                        - 眼睛：形状、大小（禁止描写眼睛颜色）
                        - 头发：颜色、长度、发型、发质（如蓬松卷发、挑染银灰等）
                        - 体型：身高感、体态、肩宽、腰线
                        - 皮肤：只写质感和独特标记（光滑/粗糙、雀斑/疤痕/纹身），禁止描述肤色
                        - 服装：款式、材质、配色、细节
                        - 鞋子：必填，款式、颜色、材质
                        - 配饰：耳钉、项链、手表等

                        非人类角色（动物、神话生物等）不受上述模板限制，以名称/物种名开头自由描述核心特征。

                        **描述禁忌**：
                        - 禁写表情、姿态、动作、情绪
                        - 禁写背景、环境、道具、光影、画风
                        - 禁写身体部位颜色（肤色/唇色/眼色/脸色），头发和服装颜色可写
                        - 禁写抽象气质（如"气场强大"、"成熟稳重的气息"）
                        - 禁用不确定词（或/可能/也许/大概）
                        - 服装发型须符合故事年代设定
                        - 原文有描述以原文优先（颜色禁忌仍需遵守）

                        场景/道具等其他类型按实际特征描述即可。

                        返回值：created（新建）、existing（复用）、allAssets（全部）
                        """,
                MAX_BATCH_SIZE);
    }

    @Override
    public String getParametersSchema() {
        return String.format(
                """
                        {
                            "type": "object",
                            "properties": {
                                "projectId": {
                                    "type": "integer",
                                    "description": "项目ID（必填）"
                                },
                                "assets": {
                                    "type": "array",
                                    "maxItems": %d,
                                    "description": "资产列表（最多%d个，超出请分次调用）",
                                    "items": {
                                        "type": "object",
                                        "properties": {
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
                                                "description": "资产描述（必填）。角色类资产须撰写可视化静态外貌描述（80-150字中文），禁止描述表情、姿态、动作、情绪（如微笑、困惑、奔跑等都不允许）。场景/道具等按实际特征描述"
                                            },
                                            "properties": {
                                                "type": "object",
                                                "additionalProperties": { "type": "string" },
                                                "description": "属性键值对（必填），必须包含 query_asset_metadata 返回的所有 fieldKey，每个都要填写，无值填'无'。填写内容遵循对应的 fieldDescription"
                                            },
                                            "initialItem": {
                                                "type": "object",
                                                "description": "初始子资产内容（必填），用于填充自动创建的初始子资产。name 填写具体有意义的名称（如角色名、场景名等）",
                                                "properties": {
                                                    "name": { "type": "string", "description": "子资产名称，应使用具体有意义的名称（如'张三'、'桌子'等）" },
                                                    "itemType": { "type": "string" },
                                                    "imageUrl": { "type": "string" },
                                                    "properties": {
                                                        "type": "object",
                                                        "additionalProperties": { "type": "string" },
                                                        "description": "子资产属性（必填），必须包含 query_asset_metadata 返回的所有 fieldKey，每个都要填写，无值填'无'。填写规则同主资产，遵循对应的 fieldDescription"
                                                    }
                                                },
                                                "required": ["name", "properties"]
                                            }
                                        },
                                        "required": ["type", "name", "initialItem"]
                                    }
                                }
                            },
                            "required": ["projectId", "assets"]
                        }
                        """,
                MAX_BATCH_SIZE, MAX_BATCH_SIZE);
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long projectId = params.getLong("projectId");
            JSONArray assetsArray = params.getJSONArray("assets");

            if (projectId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 projectId").toString();
            }
            if (assetsArray == null || assetsArray.isEmpty()) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 assets 数组").toString();
            }
            if (assetsArray.size() > MAX_BATCH_SIZE) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", String.format("单次最多创建 %d 个资产，当前传入 %d 个。请分多次调用。",
                                MAX_BATCH_SIZE, assetsArray.size()))
                        .toString();
            }

            Long userId = context.getUserId();

            if (!projectService.canAccessProject(projectId, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "无权访问该项目").toString();
            }

            List<JSONObject> created = new ArrayList<>();
            List<JSONObject> existing = new ArrayList<>();

            for (int i = 0; i < assetsArray.size(); i++) {
                JSONObject data = assetsArray.getJSONObject(i);
                String type = data.getStr("type");
                String name = data.getStr("name");

                if (type == null || name == null || name.isBlank()) {
                    continue;
                }

                // 去重检查
                Asset found = assetService.findByProjectTypeAndName(projectId, type, name);
                if (found != null) {
                    existing.add(JSONUtil.createObj()
                            .set("type", type).set("name", name).set("assetId", found.getId()));
                } else {
                    Asset asset = Asset.builder()
                            .projectId(projectId)
                            .type(type)
                            .name(name)
                            .description(data.getStr("description"))
                            .properties(
                                    data.containsKey("properties") ? data.getJSONObject("properties").toString() : null)
                            .sourceType(2)
                            .userId(userId)
                            .build();
                    Asset saved = assetService.create(asset);

                    // 处理 initialItem
                    JSONObject initialItemData = data.getJSONObject("initialItem");
                    if (initialItemData != null) {
                        List<AssetItem> items = assetService.listItems(saved.getId());
                        if (!items.isEmpty()) {
                            AssetItem first = items.get(0);
                            if (initialItemData.getStr("name") != null)
                                first.setName(initialItemData.getStr("name"));
                            if (initialItemData.getStr("itemType") != null)
                                first.setItemType(initialItemData.getStr("itemType"));
                            if (initialItemData.getStr("imageUrl") != null)
                                first.setImageUrl(initialItemData.getStr("imageUrl"));
                            if (initialItemData.containsKey("properties"))
                                first.setProperties(initialItemData.getJSONObject("properties").toString());
                            assetService.updateItem(first);
                        }
                    }

                    created.add(JSONUtil.createObj()
                            .set("type", type).set("name", name).set("assetId", saved.getId()));
                }
            }

            List<JSONObject> allAssets = new ArrayList<>(created);
            allAssets.addAll(existing);

            return JSONUtil.createObj()
                    .set("created", created)
                    .set("existing", existing)
                    .set("allAssets", allAssets)
                    .set("createdCount", created.size())
                    .set("existingCount", existing.size())
                    .set("message", String.format("新建 %d 个资产，复用 %d 个已有资产", created.size(), existing.size()))
                    .toString();
        } catch (Exception e) {
            log.error("批量创建资产失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "创建失败: " + e.getMessage()).toString();
        }
    }
}
