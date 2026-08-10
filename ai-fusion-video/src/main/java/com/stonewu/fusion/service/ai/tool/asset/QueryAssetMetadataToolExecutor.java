package com.stonewu.fusion.service.ai.tool.asset;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.asset.AssetMetadataRegistry;
import com.stonewu.fusion.service.asset.AssetMetadataRegistry.FieldDef;
import com.stonewu.fusion.service.asset.AssetMetadataRegistry.FieldOption;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 查询资产属性元数据工具（query_asset_metadata）
 * <p>
 * 告知 AI 各类型资产（character/scene/prop 等）允许的 properties 字段定义，
 * 包括字段名、中文标签、类型、选项列表和是否必填。
 * <p>
 * 复用 {@link AssetMetadataRegistry} 中的字段定义。
 */
@Component
@Slf4j
public class QueryAssetMetadataToolExecutor implements ToolExecutor {

    @Override
    public String getToolName() {
        return "query_asset_metadata";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询资产属性定义";
    }

    @Override
    public String getToolDescription() {
        return """
                查询指定资产类型允许的 properties 字段定义。

                **重要**：创建或更新资产的 properties 前，必须先调用此工具获取允许的字段列表。
                properties 中的 key 必须是本工具返回的 fieldKey，不允许使用未定义的 key。
                对于 fieldType 为 select 的字段，value 必须是 options 中的 value 值。

                返回每个字段的：
                - fieldKey：英文标识（用作 properties 的 key）
                - fieldLabel：中文显示名
                - fieldType：字段类型（text/select/number）
                - fieldDescription：填写指南（描述该字段应如何填写、格式要求和禁忌等）
                - options：select 类型的预设选项列表
                - required：是否必填
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "assetType": {
                            "type": "string",
                            "enum": ["character", "scene", "prop"],
                            "description": "资产类型"
                        }
                    },
                    "required": ["assetType"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            String assetType = params.getStr("assetType");

            if (assetType == null || assetType.isBlank()) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少必要参数: assetType").toString();
            }

            List<FieldDef> fields = AssetMetadataRegistry.getFields(assetType);
            if (fields == null) {
                return JSONUtil.createObj()
                        .set("status", "error")
                        .set("message", "不支持的资产类型: " + assetType
                                + "，支持的类型: " + String.join(", ", AssetMetadataRegistry.getSupportedTypes()))
                        .toString();
            }

            JSONArray fieldsArray = new JSONArray();
            for (FieldDef field : fields) {
                JSONObject fieldObj = JSONUtil.createObj()
                        .set("fieldKey", field.key())
                        .set("fieldLabel", field.label())
                        .set("fieldType", field.type())
                        .set("required", field.required());

                if (field.description() != null) {
                    fieldObj.set("fieldDescription", field.description());
                }

                if (field.options() != null && !field.options().isEmpty()) {
                    JSONArray optionsArray = new JSONArray();
                    for (FieldOption option : field.options()) {
                        optionsArray.add(JSONUtil.createObj()
                                .set("value", option.value())
                                .set("label", option.label()));
                    }
                    fieldObj.set("options", optionsArray);
                }
                fieldsArray.add(fieldObj);
            }

            return JSONUtil.createObj()
                    .set("assetType", assetType)
                    .set("totalFields", fields.size())
                    .set("fields", fieldsArray)
                    .set("message", String.format("资产类型 %s 共有 %d 个可设置的属性字段", assetType, fields.size()))
                    .toString();

        } catch (Exception e) {
            log.error("查询资产元数据失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
