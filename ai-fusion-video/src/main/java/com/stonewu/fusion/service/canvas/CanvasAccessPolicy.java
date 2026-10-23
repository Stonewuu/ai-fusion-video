package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.repository.canvas.CanvasAccessRepository;
import com.stonewu.fusion.repository.canvas.CanvasAccessRepository.ProjectScope;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasAccessPolicy {
    private final CanvasAccessRepository access;
    private final CanvasDocumentRepository documents;

    public ProjectScope requireProject(long projectId, CanvasActor actor) {
        requireActorScope(actor);
        return access.project(projectId, actor).orElseThrow(this::denied);
    }

    public void requireProjectEditor(long projectId, CanvasActor actor) {
        requireProject(projectId, actor);
        if (!access.canEditProject(projectId, actor)) {
            throw denied();
        }
    }

    public List<Long> readableProjects(CanvasActor actor) {
        requireActorScope(actor);
        return access.readableProjects(actor);
    }

    public void requireActorScope(CanvasActor actor) {
        requireActive(actor);
        if (actor.teamId() != null && !access.activeTeamMember(actor.userId(), actor.teamId())) {
            throw denied();
        }
    }

    public void requireProjectManager(long projectId, CanvasActor actor) {
        requireProject(projectId, actor);
        if (!documents.isProjectManager(projectId, actor.userId())) {
            throw denied();
        }
    }

    public CanvasPermissions requireView(CanvasDocument document, CanvasActor actor) {
        if (document.projectId() != null) {
            requireProject(document.projectId(), actor);
        } else {
            requireActorScope(actor);
        }
        boolean projectEditor = document.projectId() != null && access.canEditProject(document.projectId(), actor);
        if ((document.createdBy() == actor.userId() && (document.projectId() == null || projectEditor))
                || (document.ownerType() == 1 && document.ownerId() == actor.userId())
                || (document.projectId() != null && documents.isProjectManager(document.projectId(), actor.userId()))) {
            return CanvasPermissions.owner();
        }
        return documents.memberPermissions(document.id(), actor.userId()).orElseGet(() -> {
            if (projectEditor) {
                return new CanvasPermissions(true, true, true, false);
            }
            if (document.projectId() != null) {
                return CanvasPermissions.viewer();
            }
            throw denied();
        });
    }

    public void requireEdit(CanvasDocument document, CanvasActor actor) {
        if (!requireView(document, actor).canEdit()) {
            throw denied();
        }
        if (document.archived()) {
            throw new CanvasException(409, CanvasErrorCode.CANVAS_ARCHIVED, "请先恢复已归档的画布", Map.of());
        }
    }

    public void requireExecute(CanvasDocument document, CanvasActor actor) {
        if (!requireView(document, actor).canExecute()) {
            throw denied();
        }
        if (document.archived()) {
            throw new CanvasException(409, CanvasErrorCode.CANVAS_ARCHIVED, "已归档的画布不能运行", Map.of());
        }
    }

    public void requireManage(CanvasDocument document, CanvasActor actor) {
        if (!requireView(document, actor).canManage()) {
            throw denied();
        }
    }

    public void validateBindings(CanvasDocument document, List<CanvasNode> nodes) {
        for (CanvasNode node : nodes) {
            if (node.resource() == null) {
                continue;
            }
            if (document.projectId() == null || !access.referenceBelongsToProject(node.resource(),
                    document.projectId())) {
                throw new CanvasException(409, CanvasErrorCode.INVALID_RESOURCE_REFERENCE,
                        "引用对象已删除或不属于画布项目",
                        Map.of("nodeId", node.id(), "resourceType", node.resource().type().code()));
            }
        }
    }

    private CanvasException denied() {
        return new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "无权访问或操作此资源", Map.of());
    }

    private void requireActive(CanvasActor actor) {
        if (!access.userEnabled(actor.userId())) {
            throw denied();
        }
    }
}
