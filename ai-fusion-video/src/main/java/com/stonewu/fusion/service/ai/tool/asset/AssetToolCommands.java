package com.stonewu.fusion.service.ai.tool.asset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.*;
import com.stonewu.fusion.service.resource.model.AssetBatchCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourceDeleteCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;

@Component
@RequiredArgsConstructor
public class AssetToolCommands {
    private final ResourceCommandService commands;
    private final ResourceQueryService queries;
    private final CanvasJson json;

    public String execute(String operation, String input, ToolExecutionContext invocation) {
        try {
            var actor = invocation.requireMutationContext();
            ObjectNode values = json.read(input, ObjectNode.class);
            String request = ResourceToolRequests.requestId(json, actor);
            var receipt = switch (operation) {
                case "create_asset" -> {
                    Long project = optionalId(values.remove("projectId"), "projectId");
                    values.put("sourceType", 2);
                    yield commands.execute(new ResourceCreateCommand(request, ASSET, project, values), actor);
                }
                case "add_asset_item" -> {
                    AssetCommandSupport.id(values.get("assetId"), "assetId");
                    AssetCommandSupport.version(values.get("parentVersion"), "parentVersion");
                    if (!values.hasNonNull("imageUrl") || values.path("imageUrl").asText().isBlank()) {
                        throw AssetFields.invalid("必须指定图片 URL");
                    }
                    if (!values.has("itemType")) {
                        values.put("itemType", "original");
                    }
                    values.put("sourceType", 2);
                    yield commands.execute(new ResourceCreateCommand(request, ASSET_ITEM, null, values), actor);
                }
                case "update_asset", "update_asset_item", "update_asset_image" -> {
                    boolean root = operation.equals("update_asset");
                    String key = root ? "assetId" : operation.equals("update_asset_image") ? "itemId" : "assetItemId";
                    long id = AssetCommandSupport.id(values.remove(key), key);
                    int version = AssetCommandSupport.version(values.remove("expectedVersion"), "expectedVersion");
                    Integer parent = root ? null : AssetCommandSupport.version(values.remove("expectedParentVersion"),
                            "expectedParentVersion");
                    if (operation.equals("update_asset_image")) {
                        long assetId = AssetCommandSupport.id(values.remove("assetId"), "assetId");
                        if (queries.get(ASSET_ITEM, id, actor.actor()).data().path("assetId").longValue() != assetId) {
                            throw AssetFields.invalid("子资产不属于指定主资产");
                        }
                        if (!values.hasNonNull("imageUrl") || values.path("imageUrl").asText().isBlank()) {
                            throw AssetFields.invalid("必须指定图片 URL");
                        }
                        values.put("sourceType", 2);
                    }
                    if (root && values.has("properties")) {
                        values.set("propertyChanges", values.remove("properties"));
                    }
                    yield commands.patch(
                            new ResourcePatchCommand(request, root ? ASSET : ASSET_ITEM, id, version, values, parent),
                            actor);
                }
                case "delete_asset_resource" -> {
                    CanvasValueType type = switch (values.path("resourceType").asText()) {
                        case "asset" -> ASSET;
                        case "item" -> ASSET_ITEM;
                        default -> throw AssetFields.invalid("resourceType 仅支持 asset 或 item");
                    };
                    Integer parent = type == ASSET_ITEM ? AssetCommandSupport.version(
                            values.get("expectedParentVersion"), "expectedParentVersion") : null;
                    yield commands.execute(new ResourceDeleteCommand(request, type,
                                    AssetCommandSupport.id(values.get("resourceId"), "resourceId"),
                                    AssetCommandSupport.version(values.get("expectedVersion"), "expectedVersion"), parent),
                            actor);
                }
                case "batch_create_assets", "batch_create_asset_items" ->
                        commands.execute(batch(operation, values, request), actor);
                default -> throw AssetFields.invalid("资产写入工具未登记");
            };
            ObjectNode result = receipt.result().deepCopy();
            aliases(result);
            if (operation.equals("delete_asset_resource") && values.path("resourceType").asText().equals("item")) {
                result.remove("assetId");
                result.put("assetItemId", receipt.result().path("id").longValue());
            }
            result.put("status", "success").put("commandId", receipt.commandId());
            result.set("changes", json.object(Map.of("changes", receipt.changes())).get("changes"));
            return json.write(result);
        } catch (CanvasException failure) {
            return json.write(
                    Map.of("status", "error", "message", failure.getMessage(), "errorCode", failure.getErrorCode(),
                            "details", failure.getDetails()));
        } catch (RuntimeException failure) {
            return json.write(Map.of("status", "error", "message",
                    failure.getMessage() == null ? "资产命令失败" : failure.getMessage()));
        }
    }

    private AssetBatchCreateCommand batch(String operation, ObjectNode params, String requestId) {
        boolean root = operation.equals("batch_create_assets");
        JsonNode list = params.get(root ? "assets" : "items");
        if (list == null || !list.isArray() || list.isEmpty() || list.size() > 10) {
            throw AssetFields.invalid("每批必须提供 1 至 10 个条目");
        }
        if (!params.path("reuseExisting").isBoolean()) {
            throw AssetFields.invalid("必须明确 reuseExisting：复用已有对象或创建独立副本");
        }
        List<ObjectNode> entries = new ArrayList<>();
        for (JsonNode value : list) {
            if (!(value instanceof ObjectNode object)) {
                throw AssetFields.invalid("资产条目必须为对象");
            }
            ObjectNode entry = object.deepCopy();
            entry.put("sourceType", 2);
            entries.add(entry);
        }
        Long parent = root ? null : AssetCommandSupport.id(params.get("assetId"), "assetId");
        Integer version = root ? null : AssetCommandSupport.version(params.get("expectedParentVersion"),
                "expectedParentVersion");
        return new AssetBatchCreateCommand(requestId, root ? ASSET : ASSET_ITEM,
                optionalId(params.get("projectId"), "projectId"),
                parent, version, params.get("reuseExisting").booleanValue(), entries);
    }

    private Long optionalId(JsonNode value, String key) {
        return value == null || value.isNull() ? null : AssetCommandSupport.id(value, key);
    }

    private void aliases(ObjectNode item) {
        if (item.has("id")) {
            item.put(item.has("assetId") ? "assetItemId" : "assetId", item.path("id").longValue());
        }
        for (String key : List.of("items", "created", "existing", "allAssets", "allItems")) {
            for (JsonNode child : item.path(key)) {
                aliases((ObjectNode) child);
            }
        }
    }
}
