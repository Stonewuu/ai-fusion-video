package com.stonewu.fusion.repository.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.AssetFields;
import com.stonewu.fusion.service.resource.ResourceMutations;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import com.stonewu.fusion.service.resource.model.ResourceMutation;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper.NewResource;
import com.stonewu.fusion.mapper.asset.AssetMapper;
import com.stonewu.fusion.mapper.asset.AssetItemMapper;
import com.stonewu.fusion.mapper.team.TeamMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.common.BusinessException;
import org.springframework.stereotype.Repository;

import java.util.*;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Repository
@RequiredArgsConstructor
public class AssetMutationRepository {
    private final ResourceMutationMapper mutations;
    private final AssetMapper assets;
    private final AssetItemMapper items;
    private final TeamMapper teams;
    private final UserMapper users;
    private final ResourceReadRepository resources;
    private final AssetFields fields;
    private final CanvasJson json;

    public ChangeSet lock(CanvasValueType type, long id) {
        var first = resources.require(type, id, false);
        if (first.projectId() != null) {
            resources.require(PROJECT, first.projectId(), true);
        }
        long assetId = type == ASSET ? id : resources.data(first).path("assetId").longValue();
        var asset = resources.require(ASSET, assetId, true);
        return new ChangeSet(asset, resources.require(type, id, true), asset.projectId());
    }

    public ResourceMutation createAsset(Long projectId, ObjectNode input, CanvasActor actor) {
        lockCreation(projectId, actor);
        Map<String, Object> ownership = new LinkedHashMap<>();
        ownership.put("projectId", projectId);
        ownership.put("userId", actor.userId());
        if (projectId != null) {
            var project = resources.data(resources.require(PROJECT, projectId, true));
            ownership.put("ownerType", project.path("ownerType").intValue());
            ownership.put("ownerId", project.path("ownerId").longValue());
        } else {
            ownership.put("ownerType", actor.teamId() == null ? 1 : 2);
            ownership.put("ownerId", actor.teamId() == null ? actor.userId() : actor.teamId());
        }
        ObjectNode draft = input.deepCopy();
        ObjectNode initial = json.object(Map.of());
        if (draft.has("initialItem")) {
            if (!(draft.remove("initialItem") instanceof ObjectNode object)) {
                throw AssetFields.invalid("initialItem 必须为对象");
            }
            initial = object;
        }
        Map<String, Object> assetValues = fields.values(ASSET, draft, true);
        fields.validateProperties((String) assetValues.get("type"), draft.get("properties"));
        Map<String, Object> itemValues = fields.values(ASSET_ITEM, initial, true);
        fields.validateProperties((String) assetValues.get("type"), initial.get("properties"));
        itemValues.putIfAbsent("itemType", "initial");
        if (!itemValues.containsKey("name")) {
            itemValues.put("name", assetValues.get("name"));
        }
        itemValues.putIfAbsent("sourceType", assetValues.getOrDefault("sourceType", 1));
        itemValues.put("sortOrder", 0);
        if (!itemValues.containsKey("imageUrl") && assetValues.containsKey("coverUrl")) {
            itemValues.put("imageUrl", assetValues.get("coverUrl"));
        }
        if (itemValues.containsKey("imageUrl")) {
            assetValues.put("coverUrl", itemValues.get("imageUrl"));
        }
        assetValues.putAll(ownership);
        var changes = new ChangeSet(null, null, projectId);
        ObjectNode asset = changes.create(ASSET, assetValues);
        itemValues.put("assetId", asset.path("id").longValue());
        ObjectNode item = changes.create(ASSET_ITEM, itemValues);
        item.put("assetVersion", 0);
        asset.set("items", json.object(Map.of("items", List.of(item))).get("items"));
        return changes.result(asset);
    }

    public void lockCreation(Long projectId, CanvasActor actor) {
        if (projectId != null) {
            resources.require(PROJECT, projectId, true);
        } else {
            Long locked = actor.teamId() != null ? teams.lockEnabled(actor.teamId()) : users.lockEnabled(
                    actor.userId());
            if (locked == null) {
                throw new BusinessException(403, "当前账号或团队不可用，不能创建资产");
            }
        }
    }

    public List<Long> namedAssets(Long projectId, CanvasActor actor, String type, String name) {
        return assets.lockNamed(projectId, actor.teamId() == null ? 1 : 2,
                actor.teamId() == null ? actor.userId() : actor.teamId(), type, name);
    }

    public List<Long> namedItems(long assetId, String name) {
        return items.lockNamed(assetId, name);
    }

    public final class ChangeSet {
        public final ResourceHeader asset;
        public final ResourceHeader target;
        private final Long projectId;
        private final List<ResourceChange> changes = new ArrayList<>();
        private final List<ObjectNode> before = new ArrayList<>();

        private ChangeSet(ResourceHeader asset, ResourceHeader target, Long projectId) {
            this.asset = asset;
            this.target = target;
            this.projectId = projectId;
        }

        public ObjectNode data(ResourceHeader header) {
            return resources.data(header);
        }

        public void requireParentVersion(Integer expected) {
            if (expected == null) {
                throw AssetFields.invalid("子资产写入必须提供 expectedParentVersion，保护主资产封面和变体集合");
            }
            ResourceMutations.requireVersion(asset, expected);
        }

        public List<ResourceHeader> items() {
            return items.lockChildren(asset.resourceId()).stream()
                    .map(id -> resources.require(ASSET_ITEM, id, true)).toList();
        }

        public ObjectNode patch(ResourceHeader header, Map<String, Object> values) {
            ResourceMutations.requireVersion(header, header.version());
            before.add(json.object(Map.of("header", header, "data", data(header))));
            Map<String, Object> columns = new LinkedHashMap<>();
            values.forEach((key, value) -> columns.put(AssetFields.column(key), value));
            int rows = mutations.patch(header.resourceType(), header.resourceId(), header.version(), columns, false);
            if (rows != 1) {
                throw new IllegalStateException("已锁定资产的版本发生变化");
            }
            boolean deleted = Boolean.TRUE.equals(values.get("deleted"));
            changes.add(ResourceMutations.change(header.resourceType(), header.resourceId(), projectId,
                    header.version() + 1,
                    deleted, values.isEmpty() ? List.of("items") : List.copyOf(values.keySet())));
            if (deleted) {
                return json.object(Map.of("id", header.resourceId(), "version", header.version() + 1, "deleted", true));
            }
            return data(resources.require(header.resourceType(), header.resourceId(), false));
        }

        public ObjectNode create(CanvasValueType type, Map<String, Object> values) {
            Map<String, Object> columns = new LinkedHashMap<>();
            values.forEach((key, value) -> columns.put(AssetFields.column(key), value));
            columns.put("version", 0);
            var row = new NewResource(type, columns);
            mutations.insert(row);
            long id = row.getId();
            changes.add(ResourceMutations.change(type, id, projectId, 0, false, List.copyOf(values.keySet())));
            return data(resources.require(type, id, false));
        }

        public void syncParent(ObjectNode item, boolean mediaChanged) {
            Map<String, Object> values = new LinkedHashMap<>();
            ObjectNode root = data(asset);
            if (mediaChanged && ("initial".equals(item.path("itemType").asText()) || root.path("coverUrl").asText("")
                    .isBlank())) {
                values.put("coverUrl", item.hasNonNull("imageUrl") ? item.path("imageUrl").textValue() : null);
            }
            patch(asset, values);
        }

        public void delete(ResourceHeader header) {
            patch(header, Map.of("deleted", true));
        }

        public ObjectNode detachItem() {
            ObjectNode item = data(target);
            ObjectNode root = data(asset);
            ObjectNode deleted = patch(target, Map.of("deleted", true));
            Map<String, Object> values = new LinkedHashMap<>();
            if (item.hasNonNull("imageUrl") && Objects.equals(item.get("imageUrl"), root.get("coverUrl"))) {
                values.put("coverUrl", null);
            }
            patch(asset, values);
            return deleted;
        }

        public ResourceMutation result(ObjectNode value) {
            return new ResourceMutation(value, List.copyOf(changes), json.object(Map.of("resources", before)));
        }
    }


}
