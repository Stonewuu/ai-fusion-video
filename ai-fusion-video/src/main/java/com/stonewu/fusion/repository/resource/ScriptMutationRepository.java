package com.stonewu.fusion.repository.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceMutations;
import com.stonewu.fusion.service.resource.ScriptFields;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import com.stonewu.fusion.service.resource.model.ResourceMutation;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper.NewResource;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import org.springframework.stereotype.Repository;

import java.util.*;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Repository
@RequiredArgsConstructor
public class ScriptMutationRepository {
    private final ResourceMutationMapper mutations;
    private final ScriptEpisodeMapper episodes;
    private final ResourceReadRepository resources;
    private final CanvasJson json;

    /**
     * 与项目删除、项目改名统一锁顺序：项目 → 剧本 → 分集 → 场次。
     */
    public ChangeSet lock(CanvasValueType type, long id) {
        ResourceHeader initial = resources.require(type, id, false);
        resources.require(PROJECT, initial.projectId(), true);
        ObjectNode data = resources.data(resources.require(type, id, false));
        long scriptId = type == SCRIPT ? id : data.path("scriptId").longValue();
        ResourceHeader script = resources.require(SCRIPT, scriptId, true);
        ResourceHeader episode = null;
        if (type == SCRIPT_EPISODE || type == SCRIPT_SCENE) {
            long episodeId = type == SCRIPT_EPISODE ? id : data.path("episodeId").longValue();
            episode = resources.require(SCRIPT_EPISODE, episodeId, true);
            if (resources.data(episode).path("scriptId").longValue() != scriptId) {
                throw ScriptFields.invalid("场次与分集所属剧本不一致");
            }
        }
        ResourceHeader target = resources.require(type, id, true);
        return new ChangeSet(script, episode, target);
    }

    public final class ChangeSet {
        public final ResourceHeader script;
        public final ResourceHeader episode;
        public final ResourceHeader target;
        private final List<ResourceChange> changes = new ArrayList<>();
        private final List<ObjectNode> before = new ArrayList<>();

        private ChangeSet(ResourceHeader script, ResourceHeader episode, ResourceHeader target) {
            this.script = script;
            this.episode = episode;
            this.target = target;
        }

        public ObjectNode data(ResourceHeader header) {
            return resources.data(header);
        }

        public ResourceHeader header(CanvasValueType type, long id) {
            var header = resources.require(type, id, true);
            if (!Objects.equals(header.projectId(), script.projectId())) {
                throw ScriptFields.invalid("对象不属于当前项目");
            }
            return header;
        }

        public List<ResourceHeader> children(CanvasValueType type, long parentId) {
            String column = type == SCRIPT_EPISODE ? "script_id" : "episode_id";
            List<Long> ids = mutations.lockChildren(type, column, parentId);
            return ids.stream().map(id -> header(type, id)).toList();
        }

        public ObjectNode patch(ResourceHeader header, Map<String, Object> fields) {
            ResourceMutations.requireVersion(header, header.version());
            before.add(json.object(Map.of("header", header, "data", data(header))));
            Map<String, Object> columns = new LinkedHashMap<>();
            fields.forEach((key, value) -> columns.put(ScriptFields.column(key), value));
            int rows = mutations.patch(header.resourceType(), header.resourceId(), header.version(), columns, false);
            if (rows != 1) {
                throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, "剧本内容版本已变化", Map.of());
            }
            boolean deleted = Boolean.TRUE.equals(fields.get("deleted"));
            changes.add(ResourceMutations.change(header.resourceType(), header.resourceId(), script.projectId(),
                    header.version() + 1,
                    deleted, fields.isEmpty() ? List.of("structure") : List.copyOf(fields.keySet())));
            if (deleted) {
                return json.object(Map.of("id", header.resourceId(), "version", header.version() + 1, "deleted", true));
            }
            return data(resources.require(header.resourceType(), header.resourceId(), false));
        }

        public ObjectNode create(CanvasValueType type, Map<String, Object> fields) {
            Map<String, Object> columns = new LinkedHashMap<>();
            fields.forEach((key, value) -> columns.put(ScriptFields.column(key), value));
            columns.put("version", 0);
            var row = new NewResource(type, columns);
            mutations.insert(row);
            long id = row.getId();
            changes.add(ResourceMutations.change(type, id, script.projectId(), 0, false, List.copyOf(fields.keySet())));
            return data(resources.require(type, id, false));
        }

        public void delete(ResourceHeader header) {
            patch(header, Map.of("deleted", true));
        }

        public void refreshScript() {
            patch(script, Map.of("totalEpisodes", children(SCRIPT_EPISODE, script.resourceId()).size()));
        }

        public void refreshEpisode() {
            patch(episode, Map.of("totalScenes", children(SCRIPT_SCENE, episode.resourceId()).size()));
        }

        public void requireEpisodeNumber(int number, Long excludingId) {
            long matches = episodes.selectCount(new LambdaQueryWrapper<ScriptEpisode>()
                    .eq(ScriptEpisode::getScriptId, script.resourceId()).eq(ScriptEpisode::getEpisodeNumber, number)
                    .ne(excludingId != null, ScriptEpisode::getId, excludingId));
            if (matches > 0) {
                throw ScriptFields.invalid("集号已存在，请按实际分集 ID 修改");
            }
        }

        public void validateAssets(ObjectNode fields) {
            if (fields.hasNonNull("sceneAssetId")) {
                requireAsset(fields.get("sceneAssetId").longValue(), "scene");
            }
            for (String key : List.of("characterAssetIds", "propAssetIds")) {
                if (fields.hasNonNull(key)) {
                    for (JsonNode id : fields.get(key)) {
                        requireAsset(id.longValue(), key.equals("characterAssetIds") ? "character" : "prop");
                    }
                }
            }
            if (fields.hasNonNull("dialogues")) {
                for (JsonNode item : fields.get("dialogues")) {
                    if (item.hasNonNull("character_asset_id")) {
                        requireAsset(item.get("character_asset_id").longValue(), "character");
                    }
                }
            }
            if (fields.hasNonNull("charactersJson")) {
                for (JsonNode item : fields.get("charactersJson")) {
                    if (item.hasNonNull("assetId")) {
                        requireAsset(item.get("assetId").longValue(), "character");
                    }
                }
            }
        }

        private void requireAsset(long id, String type) {
            var asset = resources.require(ASSET, id, true);
            ObjectNode data = resources.data(asset);
            if (!Objects.equals(asset.projectId(), script.projectId()) || !type.equals(data.path("type").textValue())) {
                throw ScriptFields.invalid("关联资产必须属于当前项目且类型正确");
            }
        }

        public ResourceMutation result(ObjectNode result) {
            return new ResourceMutation(result, List.copyOf(changes), json.object(Map.of("resources", before)));
        }
    }


}
