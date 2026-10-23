package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.preview.MediaPreviewOwnership;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Objects;

@Component
@RequiredArgsConstructor
public class MediaJobFence {
    private final MediaJobMapper mediaJobs;
    private final UserMapper users;
    private final ProjectMapper projects;
    private final CanvasJson json;
    private final CanvasAccessPolicy access;
    private final ResourceAgentFence agents;
    private final CanvasDocumentRepository canvases;
    private final MediaPreviewOwnership previews;

    @Transactional(propagation = Propagation.MANDATORY)
    public void require(MediaJob.Claim claim, ResourceCommandContext expected, boolean allowCancel) {
        var row = mediaJobs.selectById(claim.id());
        if (row == null || !json.read(row.getContextJson(), ResourceCommandContext.class).equals(expected)) {
            throw lost();
        }
        var scope = new Scope(row.getKind(), row.getProjectId(), expected);
        agents.requireActive(expected, scope.projectId());
        users.lockEnabled(expected.actor().userId());
        if (scope.kind().equals("COMFY_WORKFLOW_TEST")) {
            requireWorkflowTester(expected.actor());
        } else if (scope.kind().equals("CANVAS_MEDIA")) {
            if (expected.canvasId() == null) {
                throw lost();
            }
            var canvas = canvases.require(expected.canvasId(), true);
            if (allowCancel) {
                access.requireView(canvas, expected.actor());
            } else {
                access.requireExecute(canvas, expected.actor());
            }
            var nodes = mediaJobs.activeCanvasNodes(claim.id(), expected.canvasId(), expected.actor().userId(),
                    expected.actor().teamId(), allowCancel);
            if (nodes.size() != 1) {
                throw lost();
            }
        } else if (scope.kind().equals("MEDIA_PREVIEW")) {
            previews.source(claim.id(), expected, true);
        } else if (scope.kind().equals("SOURCE_IMPORT")) {
            if (expected.canvasId() == null) {
                throw lost();
            }
            var canvas = canvases.require(expected.canvasId(), true);
            if (allowCancel) {
                access.requireView(canvas, expected.actor());
            } else {
                access.requireEdit(canvas, expected.actor());
            }
            if (!Objects.equals(canvas.projectId(), scope.projectId())) {
                throw lost();
            }
            if (expected.agentRunId() != null && mediaJobs.importAgentScope(expected.agentRunId(),
                            expected.actor().userId(), expected.actor().teamId())
                    .size() != 1) {
                throw lost();
            }
            var imports = mediaJobs.importNodeRuns(claim.id(), expected.canvasId());
            if (imports.size() != 1) {
                throw lost();
            }
            if (imports.getFirst() != null) {
                var parents = mediaJobs.activeImportParents(imports.getFirst(), claim.id(), expected.canvasId(),
                        expected.actor().userId(), expected.actor().teamId(), allowCancel);
                if (parents.size() != 1) {
                    throw lost();
                }
            }
        } else if (scope.kind().equals("EPISODE_COMPOSE") && scope.projectId() != null) {
            projects.lock(scope.projectId());
            access.requireProjectEditor(scope.projectId(), expected.actor());
        } else {
            throw MediaException.invalid("媒体任务类型或归属无效");
        }
        var valid = mediaJobs.lockOwner(claim.id(), claim.owner(), claim.epoch(), allowCancel);
        if (valid.size() != 1) {
            throw lost();
        }
    }

    public static MediaException lost() {
        return new MediaException(409, "MEDIA_JOB_FENCED", "媒体任务已取消、超时或由其他工作者接管");
    }

    public void requireWorkflowTester(CanvasActor actor) {
        access.requireActorScope(actor);
        int count = mediaJobs.countWorkflowTesterRoles(actor.userId());
        if (count == 0) {
            throw new MediaException(403, "COMFY_TEST_ACCESS_DENIED", "工作流试运行需要有效的管理员权限");
        }
    }

    private record Scope(String kind, Long projectId, ResourceCommandContext context) {
    }
}
