package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.repository.resource.ProjectWorkspaceRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.List;
import java.util.Map;
import java.util.Objects;

@Service
public class ResourceQueryService {
    private final ResourceReadRepository resources;
    private final ResourceSnapshotCache snapshots;
    private final CanvasAccessPolicy access;
    private final CanvasJson json;
    private final TransactionTemplate reads;
    private final ProjectWorkspaceRepository workspaces;
    private final ResourceAccessPolicy resourceAccess;
    private final StoryboardVersionView storyboardVersions;

    public ResourceQueryService(ResourceReadRepository resources, ResourceSnapshotCache snapshots,
                                CanvasAccessPolicy access,
                                CanvasJson json, ProjectWorkspaceRepository workspaces,
                                ResourceAccessPolicy resourceAccess,
                                StoryboardVersionView storyboardVersions, PlatformTransactionManager manager) {
        this.resources = resources;
        this.snapshots = snapshots;
        this.access = access;
        this.json = json;
        this.workspaces = workspaces;
        this.resourceAccess = resourceAccess;
        this.storyboardVersions = storyboardVersions;
        reads = new TransactionTemplate(manager);
        reads.setReadOnly(true);
        reads.setIsolationLevel(TransactionDefinition.ISOLATION_REPEATABLE_READ);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ, noRollbackFor = BusinessException.class)
    public ResourceSnapshot get(CanvasValueType type, long id, CanvasActor actor) {
        if (id <= 0) {
            throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "业务对象编号无效");
        }
        var header = resources.require(type, id, false);
        resourceAccess.require(header, actor);
        ObjectNode data = json.read(snapshots.data(header), ObjectNode.class);
        if (type == CanvasValueType.ASSET_ITEM) {
            data.put("assetVersion",
                    resources.require(CanvasValueType.ASSET, data.path("assetId").longValue(), false).version());
        }
        if (List.of(CanvasValueType.STORYBOARD_EPISODE, CanvasValueType.STORYBOARD_SCENE, CanvasValueType.SHOT)
                .contains(type)) {
            storyboardVersions.attach(type, data);
        }
        return new ResourceSnapshot(header, data);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ, noRollbackFor = BusinessException.class)
    public void authorizeHistorical(CanvasValueType type, long id, CanvasActor actor) {
        resourceAccess.requireHistorical(resources.historical(type, id), actor);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public boolean canReadHistorical(CanvasValueType type, long id, CanvasActor actor) {
        try {
            authorizeHistorical(type, id, actor);
            return true;
        } catch (BusinessException denied) {
            if (List.of(403, 404, 410).contains(denied.getCode())) {
                return false;
            }
            throw denied;
        }
    }

    public List<ObjectNode> projects(CanvasActor actor) {
        return Objects.requireNonNull(reads.execute(ignored -> resources.projects(access.readableProjects(actor))));
    }

    public ObjectNode script(long projectId, CanvasActor actor) {
        return reads.execute(ignored -> {
            access.requireProject(projectId, actor);
            return resources.scriptForProject(projectId).map(id -> get(CanvasValueType.SCRIPT, id, actor).data())
                    .orElse(null);
        });
    }

    public List<ObjectNode> scriptChildren(CanvasValueType type, long parentId, CanvasActor actor) {
        CanvasValueType parentType = switch (type) {
            case SCRIPT_EPISODE -> CanvasValueType.SCRIPT;
            case SCRIPT_SCENE -> CanvasValueType.SCRIPT_EPISODE;
            default -> throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "不是剧本子对象");
        };
        return Objects.requireNonNull(reads.execute(ignored -> {
            get(parentType, parentId, actor);
            return resources.scriptChildren(type, parentId).stream().map(id -> get(type, id, actor).data()).toList();
        }));
    }

    public ObjectNode scriptTree(CanvasValueType type, long id, CanvasActor actor) {
        return Objects.requireNonNull(reads.execute(ignored -> {
            ObjectNode root = get(type, id, actor).data();
            String key;
            CanvasValueType child;
            if (type == CanvasValueType.SCRIPT) {
                key = "episodes";
                child = CanvasValueType.SCRIPT_EPISODE;
            } else if (type == CanvasValueType.SCRIPT_EPISODE) {
                key = "scenes";
                child = CanvasValueType.SCRIPT_SCENE;
            } else {
                throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "此对象没有剧本子结构");
            }
            root.set(key, json.object(Map.of(key, scriptChildren(child, id, actor))).get(key));
            return root;
        }));
    }

    public ObjectNode projectScriptTree(long projectId, boolean scenes, CanvasActor actor) {
        return reads.execute(ignored -> {
            ObjectNode script = script(projectId, actor);
            if (script == null) {
                return null;
            }
            ObjectNode root = scriptTree(CanvasValueType.SCRIPT, script.path("id").longValue(), actor);
            if (scenes) {
                for (var episode : root.path("episodes")) {
                    ((ObjectNode) episode).set("scenes",
                            scriptTree(CanvasValueType.SCRIPT_EPISODE, episode.path("id").longValue(), actor).get(
                                    "scenes"));
                }
            }
            return root;
        });
    }

    public PageResult<ObjectNode> projectPage(CanvasActor actor, int pageNo, int pageSize) {
        if (pageNo < 1 || pageSize < 1 || pageSize > 100) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "项目分页参数无效");
        }
        return Objects.requireNonNull(reads.execute(ignored -> {
            List<Long> ids = access.readableProjects(actor);
            return new PageResult<>(resources.projectPage(ids, pageNo, pageSize), (long) ids.size());
        }));
    }

    public ObjectNode workspace(long projectId, CanvasActor actor) {
        return Objects.requireNonNull(reads.execute(ignored -> {
            access.requireProject(projectId, actor);
            return workspaces.overview(projectId);
        }));
    }
}
