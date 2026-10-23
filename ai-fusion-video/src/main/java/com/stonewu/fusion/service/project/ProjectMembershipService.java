package com.stonewu.fusion.service.project;

import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.project.ProjectMemberMapper;
import com.stonewu.fusion.mapper.project.ProjectMembershipCommandMapper;
import com.stonewu.fusion.mapper.project.ProjectMembershipStateMapper;
import com.stonewu.fusion.mapper.project.ProjectMemberMapper.PersonRow;
import com.stonewu.fusion.mapper.canvas.CanvasMemberMapper;
import com.stonewu.fusion.entity.project.ProjectMember;
import com.stonewu.fusion.entity.project.ProjectMembershipCommandEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import java.util.Optional;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.repository.canvas.CanvasAccessRepository;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasRequestValidator;
import com.stonewu.fusion.service.canvas.collaboration.CanvasCollaborationData.Person;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

import static com.stonewu.fusion.service.project.ProjectMembershipData.*;

/**
 * 权限读取实时鉴权，不与可缓存的项目内容混合；所有成员写入在项目行锁内审计。
 */
@Service
@RequiredArgsConstructor
public class ProjectMembershipService {
    private final ProjectMapper projects;
    private final ProjectMemberMapper members;
    private final ProjectMembershipCommandMapper commands;
    private final ProjectMembershipStateMapper states;
    private final CanvasMemberMapper canvasMembers;
    private final CanvasAccessPolicy access;
    private final CanvasAccessRepository identities;
    private final CanvasDocumentRepository canvases;
    private final CanvasRequestValidator validator;
    private final CanvasJson json;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Members members(long projectId, CanvasActor actor, int page, int size) {
        var project = access.requireProject(projectId, actor);
        page(page, size);
        var result = members.members(new Page<>(page, size), projectId, project.ownerId(), project.ownerType());
        return new Members(revision(projectId), canvases.isProjectManager(projectId, actor.userId()),
                project.ownerType() == 2, result.getRecords().stream().map(this::member).toList(), result.getTotal());
    }

    public Member find(long projectId, String username, CanvasActor actor) {
        var project = access.requireProject(projectId, actor);
        access.requireProjectManager(projectId, actor);
        if (username == null || username.isBlank() || username.length() > 64) {
            throw invalid("请输入完整用户名");
        }
        return Optional.ofNullable(
                        members.findPerson(project.ownerId(), project.ownerType(), projectId, username.trim()))
                .map(this::member)
                .orElseThrow(() -> invalid("未找到可添加的账号"));
    }

    public PageResult<Audit> history(long projectId, CanvasActor actor, int page, int size) {
        access.requireProjectManager(projectId, actor);
        page(page, size);
        var result = commands.history(new Page<>(page, size), projectId);
        return new PageResult<>(result.getRecords().stream().map(row -> new Audit(receipt(row),
                new Person(row.getActorUserId(), row.getActorUsername(), row.getActorName()),
                new Person(row.getUserId(), row.getMemberUsername(), row.getMemberName()))).toList(),
                result.getTotal());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "projectMember", allEntries = true)
    public Receipt change(long projectId, Change command, CanvasActor actor) {
        validator.validate(command);
        if (command.role() != null && !List.of(2, 3, 4).contains(command.role())) {
            throw invalid("可分配的项目权限为查看、编辑或管理");
        }
        if (projects.lockActive(projectId) == null) {
            throw new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED, "项目已删除或不存在", Map.of());
        }
        var project = access.requireProject(projectId, actor);
        access.requireProjectManager(projectId, actor);
        String hash = json.hash(command);
        var replay = commands.selectOne(Wrappers.<ProjectMembershipCommandEntity>lambdaQuery()
                .eq(ProjectMembershipCommandEntity::getProjectId, projectId)
                .eq(ProjectMembershipCommandEntity::getActorUserId, actor.userId())
                .eq(ProjectMembershipCommandEntity::getRequestId, command.requestId()));
        if (replay != null) {
            if (!replay.getRequestHash().equals(hash)) {
                throw new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT, "请求标识已用于另一项成员调整",
                        Map.of());
            }
            return receipt(replay);
        }
        long current = revision(projectId);
        if (current != command.expectedRevision()) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, "项目成员权限已变化，请刷新后重新确认",
                    Map.of());
        }
        if (current == Long.MAX_VALUE) {
            throw invalid("成员权限版本已达到上限");
        }
        Integer previous = Optional.ofNullable(members.selectOne(Wrappers.<ProjectMember>lambdaQuery()
                        .eq(ProjectMember::getProjectId, projectId).eq(ProjectMember::getUserId, command.userId())))
                .map(ProjectMember::getRole).orElse(null);
        if ((project.ownerType() == 1 && project.ownerId() == command.userId()) || Objects.equals(previous, 1)) {
            throw invalid("项目拥有者不能通过成员设置移除或降级");
        }
        if (Objects.equals(previous, command.role())) {
            throw invalid("成员权限没有变化");
        }
        List<Map<String, Object>> removed = List.of();
        if (command.role() == null) {
            removed = canvasMembers.projectGrants(projectId, command.userId());
            canvasMembers.removeProjectGrants(projectId, command.userId());
            members.removeMember(projectId, command.userId());
        } else {
            if (!identities.userEnabled(command.userId())) {
                throw invalid("此账号已停用或不存在");
            }
            members.grantMember(projectId, command.userId(), command.role());
        }
        states.advance(projectId, current + 1);
        String id = UUID.randomUUID().toString();
        commands.create(ProjectMembershipCommandEntity.builder()
                .id(id)
                .projectId(projectId)
                .revision(current + 1)
                .actorUserId(actor.userId())
                .actorTeamId(actor.teamId())
                .requestId(command.requestId())
                .requestHash(hash)
                .userId(command.userId())
                .previousRole(previous)
                .currentRole(command.role())
                .removedCanvasGrants(json.write(removed)).build());
        return receipt(commands.selectById(id));
    }

    private long revision(long projectId) {
        var state = states.selectById(projectId);
        return state == null ? 0 : state.getRevision();
    }

    private Member member(PersonRow row) {
        return new Member(new Person(row.id(), row.username(), row.name()), row.role(),
                Objects.equals(row.role(), 1), row.status() == 1 && !row.deleted());
    }

    private Receipt receipt(ProjectMembershipCommandEntity row) {
        return new Receipt(row.getId(), row.getProjectId(), row.getRevision(),
                row.getUserId(),
                row.getPreviousRole(), row.getCurrentRole(),
                json.read(row.getRemovedCanvasGrants(), JsonNode.class).size(),
                row.getCreateTime().toInstant(ZoneOffset.UTC));
    }

    private void page(int page, int size) {
        if (page < 1 || size < 1 || size > 100) {
            throw invalid("每页数量应为 1–100");
        }
    }

    private CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }
}
