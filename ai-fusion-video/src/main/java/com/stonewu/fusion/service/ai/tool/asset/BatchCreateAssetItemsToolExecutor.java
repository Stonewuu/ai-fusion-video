package com.stonewu.fusion.service.ai.tool.asset;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.asset.AssetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * 批量创建子资产工具执行器
 * <p>
 * 提供 batch_create_asset_items 工具：为指定主资产批量创建子资产变体。
 * 用于分镜 pipeline 中发现剧本描述的角色变体没有对应子资产时批量创建。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class BatchCreateAssetItemsToolExecutor implements ToolExecutor {

    /**
     * 单次批量创建子资产的最大数量
     */
    private static final int MAX_BATCH_SIZE = 5;

    /**
     * 来源类型：AI生成
     */
    private static final int SOURCE_AI_GENERATED = 2;

    private final AssetService assetService;

    @Override
    public String getToolName() {
        return "batch_create_asset_items";
    }

    @Override
    public String getDisplayName() {
        return "批量创建子资产";
    }

    @Override
    public String getToolDescription() {
        return String.format("""
                为指定主资产批量创建子资产变体。
                每个主资产创建时会自动生成一个**初始子资产**（默认变体），图片挂在子资产上而非主资产。
                新建子资产变体代表角色/场景在**外观上有显著变化**的其他版本，每个子资产对应一张独立的参考图片。

                **何时需要创建新子资产（外观发生显著变化）**：
                - 身体状态显著改变：如严重受伤（手臂打石膏、脸上有伤疤）、残疾、怀孕等
                - 年龄阶段变化：如"少年时期的张三"、"老年的张三"
                - 服装/造型大幅变化：如"穿婚纱的李梅"、"穿军装的张三"、"剃了光头的王五"
                - 场景外观显著变化：如"被火烧毁的咖啡厅"、"夜晚的咖啡厅"、"雪后的校园"
                - 道具外观显著变化：如"破碎的花瓶"、"生锈的铁门"

                **何时不需要创建子资产（直接复用初始子资产或已有合适的子资产）**：
                - 表情变化：如"微笑的张三"、"愤怒的张三"、"哭泣的李梅" → 直接复用初始子资产
                - 心理/情绪状态：如"紧张的张三"、"失落的李梅"、"兴奋的王五" → 直接复用初始子资产
                - 简单动作/姿态：如"奔跑的张三"、"坐着的李梅" → 直接复用初始子资产
                - 光影/氛围变化：如"昏暗灯光下的教室" → 直接复用场景的初始子资产
                - 仅对话/台词不同：角色在说不同的话 → 直接复用初始子资产

                **判断原则**：想象一下，如果需要画一张新的参考图才能表达这个变化，就创建子资产；
                如果原有的参考图已经足够表达（表情、情绪可以在分镜描述中用文字说明），就直接复用初始子资产。

                ## 角色类子资产描述规范（description 字段）

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

                **重要规则**：
                1. 单次调用最多创建 %d 个子资产。如果需要创建更多，请分多次调用
                2. 工具自动按 assetId + name 去重，已存在的同名子资产会被复用
                3. 子资产的 properties 仅填写该子资产特有的属性值，不要重复主资产已有的 properties
                4. 调用前应先使用 query_asset_items 查看已有子资产，避免重复创建
                5. 宁可少创建，不要多创建。对于不确定是否需要的变体，优先复用初始子资产

                返回值：
                - created：本次新建的子资产
                - existing：已存在被复用的子资产
                - allItems：全部子资产（新建+复用）
                """, MAX_BATCH_SIZE);
    }

    @Override
    public String getParametersSchema() {
        return String.format("""
                {
                    "type": "object",
                    "properties": {
                        "assetId": {
                            "type": "integer",
                            "description": "主资产ID（必填）"
                        },
                        "items": {
                            "type": "array",
                            "maxItems": %d,
                            "description": "子资产列表（最多%d个，超出请分次调用）。注意：仅当角色/场景的外观有显著变化时才创建，表情、情绪、简单动作变化不需要创建子资产",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "子资产名称（必填），应体现外观上的显著变化。好的示例：'手臂打石膏的张三'、'穿婚纱的李梅'、'被火烧毁的咖啡厅'。不好的示例（不应创建子资产）：'微笑的张三'、'紧张的李梅'"
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "子资产描述（可选），详细描述该变体在外观上与主资产的区别，禁止描述表情、情绪、动作"
                                    },
                                    "itemType": {
                                        "type": "string",
                                        "description": "子资产类型（可选），如 variant（外观变体）、costume（服装变化）、age（年龄变化）、damaged（损坏状态）等，默认为 variant"
                                    },
                                    "properties": {
                                        "type": "object",
                                        "additionalProperties": { "type": "string" },
                                        "description": "子资产属性键值对（必填），必须包含 query_asset_metadata 返回的所有 fieldKey，每个都要填写，无值填'无'。仅填写与主资产不同的值即可，相同的也需列出"
                                    }
                                },
                                "required": ["name"]
                            }
                        }
                    },
                    "required": ["assetId", "items"]
                }
                """, MAX_BATCH_SIZE, MAX_BATCH_SIZE);
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long assetId = params.getLong("assetId");
            JSONArray itemsArray = params.getJSONArray("items");

            // 参数校验
            if (assetId == null) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "缺少必要参数: assetId").toString();
            }
            if (itemsArray == null || itemsArray.isEmpty()) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "缺少必要参数: items 不能为空").toString();
            }
            if (itemsArray.size() > MAX_BATCH_SIZE) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", String.format("单次最多创建 %d 个子资产，当前传入 %d 个。请分多次调用。",
                                MAX_BATCH_SIZE, itemsArray.size())).toString();
            }

            Long userId = context.getUserId();

            Asset asset = assetService.getById(assetId);
            if (!assetService.canAccessAsset(asset, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "未找到ID为 " + assetId + " 的资产或无权访问").toString();
            }

            // 查询已有子资产（用于去重）
            List<AssetItem> existingItems = assetService.listItems(assetId);

            List<JSONObject> created = new ArrayList<>();
            List<JSONObject> existing = new ArrayList<>();

            for (int i = 0; i < itemsArray.size(); i++) {
                JSONObject itemData = itemsArray.getJSONObject(i);
                String name = itemData.getStr("name");
                if (StrUtil.isBlank(name)) {
                    continue;
                }

                // 去重检查
                AssetItem found = existingItems.stream()
                        .filter(e -> name.equals(e.getName()))
                        .findFirst().orElse(null);

                if (found != null) {
                    existing.add(JSONUtil.createObj()
                            .set("assetItemId", found.getId())
                            .set("name", found.getName())
                            .set("itemType", found.getItemType()));
                    log.info("子资产已存在，复用: assetId={}, name={}, itemId={}", assetId, name, found.getId());
                    continue;
                }

                // 创建新子资产
                AssetItem newItem = AssetItem.builder()
                        .assetId(assetId)
                        .name(name)
                        .itemType(itemData.getStr("itemType", "variant"))
                        .sourceType(SOURCE_AI_GENERATED)
                        .sortOrder(existingItems.size() + i)
                        .properties(itemData.containsKey("properties") ? itemData.getJSONObject("properties").toString() : null)
                        .build();

                AssetItem saved = assetService.createItem(newItem);

                created.add(JSONUtil.createObj()
                        .set("assetItemId", saved.getId())
                        .set("name", name)
                        .set("itemType", newItem.getItemType()));
            }

            // 合并结果
            List<JSONObject> allItems = new ArrayList<>(created);
            allItems.addAll(existing);

            return JSONUtil.createObj()
                    .set("assetId", assetId)
                    .set("assetName", asset.getName())
                    .set("created", created)
                    .set("existing", existing)
                    .set("allItems", allItems)
                    .set("createdCount", created.size())
                    .set("existingCount", existing.size())
                    .set("message", String.format("新建 %d 个子资产，复用 %d 个已有子资产",
                            created.size(), existing.size()))
                    .toString();
        } catch (Exception e) {
            log.error("批量创建子资产失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "创建失败: " + e.getMessage()).toString();
        }
    }
}
