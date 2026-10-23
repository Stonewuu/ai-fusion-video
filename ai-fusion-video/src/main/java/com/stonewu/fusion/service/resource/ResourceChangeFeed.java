package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.team.TeamMember;
import com.stonewu.fusion.entity.project.ProjectMember;
import com.stonewu.fusion.mapper.resource.ResourceCommandMapper;
import com.stonewu.fusion.mapper.team.TeamMemberMapper;
import com.stonewu.fusion.mapper.project.ProjectMemberMapper;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 原页面也读取已提交资源事件，既不依赖工具名称，也不把全局自增 ID 当作提交顺序。
 */
@Component
@RequiredArgsConstructor
public class ResourceChangeFeed {
    private final ResourceCommandMapper commands;
    private final TeamMemberMapper teamMembers;
    private final ProjectMemberMapper projectMembers;
    private final CanvasJson json;
    private final CanvasAccessPolicy scopes;
    private final ResourceAccessPolicy access;
    private final ResourceReadRepository resources;

    public record Entry(long sequence, String commandId, ResourceChange change) {
    }

    public record Page(long cursor, long head, String scope, boolean reset, boolean hasMore, List<Entry> events) {
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Page read(CanvasActor actor, Long afterSequence, String observedScope, int limit) {
        scopes.requireActorScope(actor);
        if (limit < 1 || limit > 500 || (afterSequence != null && afterSequence < 0)) {
            throw new CanvasException(CanvasErrorCode.INVALID_CURSOR, "资源事件游标或分页大小无效");
        }
        String scope = scope(actor);
        long head = commands.head(false);
        if (afterSequence == null || !scope.equals(observedScope)) {
            return new Page(head, head, scope, true, false, List.of());
        }
        if (afterSequence > head) {
            throw new CanvasException(CanvasErrorCode.INVALID_CURSOR, "资源事件游标超出已提交范围");
        }
        var batch = commands.feed(afterSequence, head, limit).stream()
                .map(row -> new Entry(row.getSequenceNo(), row.getCommandId(),
                        new ResourceChange(row.getEventId(), CanvasValueType.fromCode(row.getResourceType()),
                                row.getResourceId(),
                                row.getProjectId(), row.getResourceVersion(), row.getDeleted(),
                                List.of(json.read(row.getChangedFields(), String[].class))))).toList();
        long cursor = afterSequence;
        List<Entry> visible = new ArrayList<>();
        for (Entry entry : batch) {
            if (entry.sequence() != cursor + 1) {
                return new Page(head, head, scope, true, false, List.of());
            }
            cursor = entry.sequence();
            if (canRead(entry.change(), actor)) {
                visible.add(entry);
            }
        }
        if (batch.isEmpty() && cursor < head) {
            return new Page(head, head, scope, true, false, List.of());
        }
        return new Page(cursor, head, scope, false, cursor < head, List.copyOf(visible));
    }

    private boolean canRead(ResourceChange change, CanvasActor actor) {
        try {
            var header = resources.find(change.resourceType(), change.resourceId(), false, true);
            if (header.isEmpty()) {
                return false;
            }
            access.require(header.get(), actor);
            return true;
        } catch (CanvasException failure) {
            if (failure.getCode() == 403 || failure.getCode() == 404) {
                return false;
            }
            throw failure;
        }
    }

    private String scope(CanvasActor actor) {
        // 成员新增/移除与项目删除会改变可见集合：客户端重新读取，而不是保留撤权后的旧投影。
        var members = actor.teamId() == null ? List.of() : teamMembers.selectMaps(
                new LambdaQueryWrapper<TeamMember>().select(TeamMember::getUserId, TeamMember::getRole)
                        .eq(TeamMember::getTeamId, actor.teamId()).eq(TeamMember::getStatus, 1)
                        .orderByAsc(TeamMember::getUserId, TeamMember::getId));
        var projectRoles = projectMembers.selectMaps(
                new LambdaQueryWrapper<ProjectMember>().select(ProjectMember::getProjectId, ProjectMember::getRole)
                        .eq(ProjectMember::getUserId, actor.userId())
                        .orderByAsc(ProjectMember::getProjectId, ProjectMember::getId));
        return json.hash(
                Map.of("actor", actor, "projects", scopes.readableProjects(actor), "members", members, "projectRoles",
                        projectRoles));
    }
}
