package com.stonewu.fusion.service.ai.tool.asset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
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
            if (operation.equals("get_asset")) {
                result = resources.get(ASSET, AssetCommandSupport.id(params.get("assetId"), "assetId"), actor).data();
                decorate(result);
            } else {
                List<ObjectNode> rows;
                if (operation.equals("list_project_assets")) {
                    rows = assets.list(optionalId(params.get("projectId")), text(params, "type"), null, true, actor);
                } else if (operation.equals("query_asset_items")) {
                    int modes = (params.has("assetIds") ? 1 : 0) + (params.has("assetId") ? 1 : 0) + (params.has(
                            "assetName") ? 1 : 0);
                    if (modes != 1) {
                        throw AssetFields.invalid("请明确选择 assetIds、assetId 或 assetName 查询方式");
                    }
                    if (params.has("assetIds")) {
                        JsonNode ids = params.get("assetIds");
                        if (!ids.isArray() || ids.isEmpty() || ids.size() > 10) {
                            throw AssetFields.invalid("每次查询 1 至 10 个资产");
                        }
                        rows = new ArrayList<>();
                        var seen = new HashSet<Long>();
                        for (JsonNode id : ids) {
                            long value = AssetCommandSupport.id(id, "assetIds");
                            if (!seen.add(value)) {
                                throw AssetFields.invalid("资产 ID 不能重复");
                            }
                            rows.add(assets.tree(value, actor));
                        }
                    } else if (params.has("assetId")) {
                        rows = List.of(assets.tree(AssetCommandSupport.id(params.get("assetId"), "assetId"), actor));
                    } else {
                        rows = assets.list(AssetCommandSupport.id(params.get("projectId"), "projectId"), null,
                                text(params, "assetName"), true, actor);
                    }
                } else {
                    throw AssetFields.invalid("资产查询工具未登记");
                }
                rows.forEach(this::decorate);
                result = json.object(Map.of("total", rows.size(), "assets", rows));
            }
            result.put("status", "success");
            return json.write(result);
        } catch (CanvasException failure) {
            return json.write(
                    Map.of("status", "error", "message", failure.getMessage(), "errorCode", failure.getErrorCode(),
                            "details", failure.getDetails()));
        } catch (RuntimeException failure) {
            return json.write(Map.of("status", "error", "message",
                    failure.getMessage() == null ? "资产查询失败" : failure.getMessage()));
        }
    }

    private Long optionalId(JsonNode value) {
        return value == null || value.isNull() ? null : AssetCommandSupport.id(value, "projectId");
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

    private void decorate(ObjectNode item) {
        item.put(item.has("assetId") ? "assetItemId" : "assetId", item.path("id").longValue());
        for (String key : List.of("coverUrl", "imageUrl", "thumbnailUrl")) {
            if (item.hasNonNull(key)) {
                String stored = item.path(key).textValue();
                item.put(key + "Stored", stored);
                item.put(key, (MediaReference.managed(stored) ? stored : configuration.resolvePublicUrl(stored)));
            }
        }
        if (item.has("items")) {
            item.put("itemCount", item.path("items").size());
            for (JsonNode child : item.path("items")) {
                decorate((ObjectNode) child);
            }
        }
    }
}
