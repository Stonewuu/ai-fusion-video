package com.stonewu.fusion.repository.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceMutations;
import com.stonewu.fusion.service.resource.StoryboardFields;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import com.stonewu.fusion.service.resource.model.ResourceMutation;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper.NewResource;
import com.stonewu.fusion.mapper.storyboard.StoryboardItemMapper;
import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.util.*;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Repository
@RequiredArgsConstructor
public class StoryboardMutationRepository {
    private final ResourceMutationMapper mutations;
    private final StoryboardItemMapper shots;
    private final ResourceReadRepository resources;
    private final CanvasJson json;

    /**
     * 项目 → 分镜 → 分集 → 场次 → 镜头；所有旧入口迁入后共享此串行化边界。
     */
    public ChangeSet lock(CanvasValueType type, long id) {
        var initial = resources.require(type, id, false);
        resources.require(PROJECT, initial.projectId(), true);
        var data = resources.data(resources.require(type, id, false));
        long boardId = type == STORYBOARD ? id : data.path("storyboardId").longValue();
        var board = resources.require(STORYBOARD, boardId, true);
        Long episodeId = nullableId(data, type == SHOT ? "storyboardEpisodeId" : "episodeId");
        if (type == STORYBOARD_EPISODE) {
            episodeId = id;
        }
        Long sceneId = null;
        if (type == STORYBOARD_SCENE) {
            sceneId = id;
        } else if (type == SHOT) {
            sceneId = nullableId(data, "storyboardSceneId");
        }
        if (sceneId != null) {
            var scene = resources.data(resources.require(STORYBOARD_SCENE, sceneId, false));
            if (scene.path("storyboardId").longValue() != boardId) {
                throw StoryboardFields.invalid("场次不属于当前分镜");
            }
            long actualEpisode = scene.path("episodeId").longValue();
            if (episodeId != null && episodeId != actualEpisode) {
                throw StoryboardFields.invalid("镜头分集与场次不一致");
            }
            episodeId = actualEpisode;
        }
        if (episodeId != null && resources.data(resources.require(STORYBOARD_EPISODE, episodeId, true))
                .path("storyboardId").longValue() != boardId) {
            throw StoryboardFields.invalid("分集不属于当前分镜");
        }
        if (sceneId != null) {
            resources.require(STORYBOARD_SCENE, sceneId, true);
        }
        return new ChangeSet(board, resources.require(type, id, true));
    }

    private static Long nullableId(ObjectNode data, String key) {
        return data.hasNonNull(key) ? data.get(key).longValue() : null;
    }

    public final class ChangeSet {
        public final ResourceHeader board;
        public final ResourceHeader target;

        private record Key(CanvasValueType type, long id) {
        }

        private final List<ResourceChange> changes = new ArrayList<>();
        private final List<ObjectNode> before = new ArrayList<>();
        private final Set<Key> touched = new HashSet<>();
        private final Set<Key> parents = new HashSet<>();
        private int creations;

        private ChangeSet(ResourceHeader board, ResourceHeader target) {
            this.board = board;
            this.target = target;
        }

        public ObjectNode data(ResourceHeader header) {
            return resources.data(header);
        }

        public ResourceHeader header(CanvasValueType type, long id) {
            var result = resources.require(type, id, true);
            ObjectNode value = data(result);
            long boardId = type == STORYBOARD ? id : value.path("storyboardId").longValue();
            if (!Objects.equals(result.projectId(), board.projectId()) || boardId != board.resourceId()) {
                throw StoryboardFields.invalid("对象不属于当前分镜");
            }
            return result;
        }

        public List<ResourceHeader> children(ResourceHeader parent) {
            var child = StoryboardFields.child(parent.resourceType());
            String column = StoryboardFields.column(StoryboardFields.parentKey(child));
            return findChildren(child, column, parent.resourceId());
        }

        private List<ResourceHeader> findChildren(CanvasValueType type, String column, long parentId) {
            return mutations.lockChildren(type, column, parentId).stream().map(id -> header(type, id)).toList();
        }

        public ObjectNode patch(ResourceHeader current, Map<String, Object> fields) {
            ResourceMutations.requireVersion(current, current.version());
            ObjectNode previous = data(current);
            before.add(json.object(Map.of("header", current, "data", previous)));
            Map<String, Object> columns = new LinkedHashMap<>();
            fields.forEach((key, value) -> columns.put(StoryboardFields.column(key), value));
            int rows = mutations.patch(current.resourceType(), current.resourceId(), current.version(), columns, false);
            if (rows != 1) {
                throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, "分镜版本已变化", Map.of());
            }
            boolean deleted = Boolean.TRUE.equals(fields.get("deleted"));
            changes.add(ResourceMutations.change(current.resourceType(), current.resourceId(), board.projectId(),
                    current.version() + 1,
                    deleted, fields.isEmpty() ? List.of("structure") : List.copyOf(fields.keySet())));
            touched.add(new Key(current.resourceType(), current.resourceId()));
            markParents(current.resourceType(), previous);
            return deleted ? json.object(
                    Map.of("id", current.resourceId(), "version", current.version() + 1, "deleted", true))
                    : data(header(current.resourceType(), current.resourceId()));
        }

        public ObjectNode create(CanvasValueType type, Map<String, Object> fields) {
            if (++creations > 5000) {
                throw StoryboardFields.invalid("单次操作最多创建 5000 个分镜对象");
            }
            Map<String, Object> values = new LinkedHashMap<>();
            fields.forEach((key, value) -> values.put(StoryboardFields.column(key), value));
            values.put("version", 0);
            Instant now = Instant.now();
            values.put("create_time", now);
            values.put("update_time", now);
            var row = new NewResource(type, values);
            mutations.insert(row);
            long id = row.getId();
            changes.add(ResourceMutations.change(type, id, board.projectId(), 0, false, List.copyOf(fields.keySet())));
            ObjectNode created = data(header(type, id));
            touched.add(new Key(type, id));
            markParents(type, created);
            return created;
        }

        private void markParents(CanvasValueType type, ObjectNode value) {
            if (type != STORYBOARD) {
                parents.add(new Key(STORYBOARD, board.resourceId()));
            }
            if (type == STORYBOARD_SCENE) {
                parents.add(new Key(STORYBOARD_EPISODE, value.path("episodeId").longValue()));
            }
            if (type == SHOT) {
                if (value.hasNonNull("storyboardEpisodeId")) {
                    parents.add(new Key(STORYBOARD_EPISODE, value.path("storyboardEpisodeId").longValue()));
                }
                if (value.hasNonNull("storyboardSceneId")) {
                    long scene = value.path("storyboardSceneId").longValue();
                    parents.add(new Key(STORYBOARD_SCENE, scene));
                    parents.add(new Key(STORYBOARD_EPISODE,
                            data(header(STORYBOARD_SCENE, scene)).path("episodeId").longValue()));
                }
            }
        }

        public void clear(ResourceHeader parent) {
            if (parent.resourceType() == STORYBOARD) {
                findChildren(SHOT, "storyboard_id", parent.resourceId()).forEach(this::delete);
                findChildren(STORYBOARD_SCENE, "storyboard_id", parent.resourceId()).forEach(this::delete);
                children(parent).forEach(this::delete);
            } else if (parent.resourceType() == STORYBOARD_EPISODE) {
                mutations.lockEpisodeShots(parent.resourceId()).stream().map(id -> header(SHOT, id))
                        .forEach(this::delete);
                children(parent).forEach(this::delete);
            } else if (parent.resourceType() == STORYBOARD_SCENE) {
                children(parent).forEach(this::delete);
            } else {
                throw StoryboardFields.invalid("镜头没有可清空的子结构");
            }
            parents.add(new Key(parent.resourceType(), parent.resourceId()));
            markParents(parent.resourceType(), data(parent));
        }

        public ObjectNode deleteTree(ResourceHeader target) {
            if (target.resourceType() != SHOT) {
                clear(target);
            }
            return delete(target);
        }

        private ObjectNode delete(ResourceHeader target) {
            return patch(target, target.resourceType() == STORYBOARD_EPISODE
                    ? Map.of("deleted", true, "deletedId", target.resourceId()) : Map.of("deleted", true));
        }

        public void reorder(ResourceHeader parent, List<Long> ids) {
            var current = children(parent);
            if (new HashSet<>(ids).size() != ids.size() || !new HashSet<>(
                    current.stream().map(ResourceHeader::resourceId).toList()).equals(new HashSet<>(ids))) {
                throw StoryboardFields.invalid("排序必须包含父集合中的全部对象且不能重复");
            }
            CanvasValueType type = StoryboardFields.child(parent.resourceType());
            for (int index = 0; index < ids.size(); index++) {
                var item = header(type, ids.get(index));
                if (data(item).path("sortOrder").intValue() != index) {
                    patch(item, Map.of("sortOrder", index));
                }
            }
            parents.add(new Key(parent.resourceType(), parent.resourceId()));
            markParents(parent.resourceType(), data(parent));
        }

        public void finish() {
            var ordered = parents.stream().sorted(Comparator.comparingInt(key -> switch (key.type()) {
                case STORYBOARD_SCENE -> 0;
                case STORYBOARD_EPISODE -> 1;
                default -> 2;
            })).toList();
            for (var key : ordered) {
                var maybe = resources.find(key.type(), key.id(), true, false);
                if (maybe.isEmpty()) {
                    if (touched.contains(key)) {
                        continue; // 本事务已经明确软删除的祖先不再递增。
                    }
                    throw StoryboardFields.invalid("分镜祖先结构不完整");
                }
                var parent = maybe.get();
                if (key.type() == STORYBOARD) {
                    Long duration = shots.totalDurationSeconds(key.id());
                    if (duration == null || duration > Integer.MAX_VALUE) {
                        throw StoryboardFields.invalid("分镜总时长超过上限");
                    }
                    if (!touched.contains(key) || data(parent).path("totalDuration").longValue() != duration) {
                        patch(parent, Map.of("totalDuration", duration.intValue()));
                    }
                } else if (!touched.contains(key)) {
                    patch(parent, Map.of());
                }
            }
        }

        public ObjectNode current(CanvasValueType type, long id) {
            return data(header(type, id));
        }

        public ResourceMutation result(ObjectNode data) {
            return new ResourceMutation(data, List.copyOf(changes), json.object(Map.of("resources", before)));
        }
    }
}
