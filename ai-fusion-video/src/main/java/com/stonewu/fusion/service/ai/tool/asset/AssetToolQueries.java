package com.stonewu.fusion.service.ai.tool.asset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.resource.AssetCommandSupport;
import com.stonewu.fusion.service.resource.AssetFields;
import com.stonewu.fusion.service.resource.AssetResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.system.SystemConfigService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;

@Component
@RequiredArgsConstructor
public class AssetToolQueries {
    private final AssetResourceQueryService assets;
    private final ResourceQueryService resources;
    private final SystemConfigService configuration;
    private final CanvasJson json;

    public String execute(String operation, String input, ToolExecutionContext invocation) {
        try {
            ObjectNode params = json.read(input, ObjectNode.class);
            var actor = invocation.requireMutationContext().actor();
            ObjectNode result;
            if (operation.equals("get_asset") || operation.equals("get_asset_item")) {
                boolean item = operation.equals("get_asset_item");
                String key = item ? "assetItemId" : "assetId";
                result = resources.get(item ? ASSET_ITEM : ASSET, AssetCommandSupport.id(params.get(key), key), actor).data();
                decorateDetail(result);
            } else if (operation.equals("list_project_assets")) {
                result = assets.summaryPage(optionalId(params.get("projectId")), text(params, "type"),
                        text(params, "keyword"), number(params, "page", 1), number(params, "pageSize", 50), actor);
                result.path("assets").forEach(row -> decorateSummary((ObjectNode) row));
                result.put("view", "summary");
                result.put("detailsTool", "get_asset");
            } else if (operation.equals("query_asset_items")) {
                result = itemDirectories(params, actor);
            } else {
                throw AssetFields.invalid("资产查询工具未登记");
            }
            result.put("status", "success");
            return json.write(result);
        } catch (CanvasException failure) {
            return json.write(Map.of("status", "error", "message", failure.getMessage(),
                    "errorCode", failure.getErrorCode(), "details", failure.getDetails()));
        } catch (RuntimeException failure) {
            return json.write(Map.of("status", "error", "message",
                    failure.getMessage() == null ? "资产查询失败" : failure.getMessage()));
        }
    }

    private ObjectNode itemDirectories(ObjectNode params, CanvasActor actor) {
        int modes = (params.has("assetIds") ? 1 : 0) + (params.has("assetId") ? 1 : 0)
                + (params.has("assetName") ? 1 : 0);
        if (modes != 1) {
            throw AssetFields.invalid("请选择 assetIds、assetId 或 assetName 中的一种查询方式");
        }
        int page = number(params, "page", 1);
        int size = number(params, "pageSize", 20);
        if (params.has("assetName")) {
            ObjectNode result = assets.summaryPage(AssetCommandSupport.id(params.get("projectId"), "projectId"),
                    null, text(params, "assetName"), page, size, actor);
            result.path("assets").forEach(row -> decorateSummary((ObjectNode) row));
            result.put("view", "asset_candidates");
            result.put("nextStep", "使用候选 assetId 调用 query_asset_items，读取子资产目录");
            return result;
        }
        List<Long> ids = new ArrayList<>();
        if (params.has("assetIds")) {
            JsonNode values = params.get("assetIds");
            if (!values.isArray() || values.isEmpty() || values.size() > 10) {
                throw AssetFields.invalid("每次查询 1 至 10 个资产");
            }
            var seen = new HashSet<Long>();
            for (JsonNode value : values) {
                long id = AssetCommandSupport.id(value, "assetIds");
                if (!seen.add(id)) {
                    throw AssetFields.invalid("资产 ID 不能重复");
                }
                ids.add(id);
            }
        } else {
            ids.add(AssetCommandSupport.id(params.get("assetId"), "assetId"));
        }
        List<ObjectNode> rows = assets.itemSummaryPages(ids, page, size, actor);
        rows.forEach(this::decorateSummary);
        ObjectNode result = json.object(Map.of("total", rows.size(), "assets", rows));
        result.put("view", "summary").put("detailsTool", "get_asset_item");
        return result;
    }

    private Long optionalId(JsonNode value) {
        return value == null || value.isNull() ? null : AssetCommandSupport.id(value, "projectId");
    }

    private int number(ObjectNode params, String key, int defaultValue) {
        if (!params.has(key)) {
            return defaultValue;
        }
        JsonNode value = params.get(key);
        if (!value.isIntegralNumber() || !value.canConvertToInt()) {
            throw AssetFields.invalid(key + " 必须为整数");
        }
        return value.intValue();
    }

    private String text(ObjectNode params, String key) {
        if (!params.has(key)) {
            return null;
        }
        if (!params.get(key).isTextual() || params.get(key).textValue().isBlank()) {
            throw AssetFields.invalid(key + " 必须为非空文本");
        }
        return params.get(key).textValue();
    }

    private void decorateSummary(ObjectNode item) {
        item.put(item.has("assetId") ? "assetItemId" : "assetId", item.path("id").longValue());
        if (item.has("hasImage")) {
            item.put("hasImage", item.path("hasImage").asBoolean());
        }
        if (item.has("items")) {
            item.path("items").forEach(child -> decorateSummary((ObjectNode) child));
        }
    }

    private void decorateDetail(ObjectNode item) {
        item.put(item.has("assetId") ? "assetItemId" : "assetId", item.path("id").longValue());
        for (String key : List.of("coverUrl", "imageUrl", "thumbnailUrl")) {
            if (item.hasNonNull(key)) {
                String stored = item.path(key).textValue();
                item.put(key + "Stored", stored);
                item.put(key, MediaReference.managed(stored) ? stored : configuration.resolvePublicUrl(stored));
            }
        }
    }
}
