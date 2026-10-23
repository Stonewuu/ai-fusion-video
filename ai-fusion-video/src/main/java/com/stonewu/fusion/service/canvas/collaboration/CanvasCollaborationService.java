package com.stonewu.fusion.service.canvas.collaboration;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.canvas.CanvasMemberEntity;

import com.stonewu.fusion.mapper.canvas.CanvasMemberMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.repository.canvas.CanvasAccessRepository;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasRequestValidator;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasGraphChangeSet;
import com.stonewu.fusion.service.canvas.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

import static com.stonewu.fusion.service.canvas.collaboration.CanvasCollaborationData.*;

@Service
@RequiredArgsConstructor
public class CanvasCollaborationService {
    private static final CanvasGraphChangeSet EMPTY = new CanvasGraphChangeSet(List.of(), List.of(), List.of(),
            List.of());
    private final CanvasDocumentRepository documents;
    private final CanvasAccessRepository identities;
    private final CanvasJournalRepository journal;
    private final CanvasAccessPolicy access;
    private final CanvasRequestValidator validation;
    private final CanvasJson json;
    private final CanvasMemberMapper memberships;
    private final ProjectMapper projects;

    public PageResult<Member> members(String canvasId, CanvasActor actor, int page, int size) {
        var document = readable(canvasId, actor);
        if (page < 1 || size < 1 || size > 100) {
            throw invalid("每页成员数量应为 1–100");
        }
        var people = memberships.people(new Page<>(page, size), canvasId, document.projectId());
        return new PageResult<>(people.getRecords().stream().map(person -> member(document, person, actor)).toList(),
                people.getTotal());
    }

    /**
     * 精确查找已知用户名，不向普通账号暴露完整站点用户目录。
     */
    public Member findMember(String canvasId, String username, CanvasActor actor) {
        var document = readable(canvasId, actor);
        access.requireManage(document, actor);
        if (username == null || username.isBlank() || username.length() > 64) {
            throw invalid("请输入完整用户名");
        }
        Person person = Optional.ofNullable(memberships.findPerson(username.trim()))
                .orElseThrow(() -> invalid("未找到可添加的账号"));
        requireEligible(document, person.userId(), actor);
        return member(document, person, actor);
    }

    @Transactional
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.revision() - 1)")
    public CanvasCommandReceipt grant(String canvasId, Grant command, CanvasActor actor) {
        validation.id(canvasId);
        validation.validate(command);
        var document = documents.require(canvasId, true);
        if (document.projectId() != null) {
            projects.lock(document.projectId());
        }
        access.requireManage(document, actor);
        String hash = json.hash(command);
        var replay = journal.find(canvasId, actor.userId(), command.requestId());
        if (replay.isPresent()) {
            if (!replay.get().requestHash().equals(hash)) {
                throw new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT,
                        "请求标识已用于另一项成员调整", Map.of());
            }
            return replay.get().receipt();
        }
        if (document.revision() != command.expectedRevision()) {
            throw conflict("画布已变化，请刷新成员信息后再次确认");
        }
        if (protectedOwner(document, command.userId(), actor)) {
            throw invalid("拥有者及继承的管理权限不能在画布中移除");
        }
        if (command.canManage() && !command.canEdit()) {
            throw invalid("画布管理者必须具备编辑权限");
        }
        var previous = documents.memberPermissions(canvasId, command.userId()).orElse(null);
        if (command.remove()) {
            if (previous == null) {
                throw invalid("此成员没有单独的画布授权；项目权限请在项目成员设置中调整");
            }
            memberships.delete(Wrappers.<CanvasMemberEntity>lambdaQuery()
                    .eq(CanvasMemberEntity::getCanvasId, canvasId).eq(CanvasMemberEntity::getUserId, command.userId()));
        } else {
            requireEligible(document, command.userId(), actor);
            memberships.grant(canvasId, command.userId(), command.canEdit(), command.canExecute(), command.canManage(),
                    actor.userId());
        }
        Map<String, Object> audit = new LinkedHashMap<>();
        audit.put("command", command);
        audit.put("previous", previous);
        audit.put("current",
                command.remove() ? null : new CanvasPermissions(true, command.canEdit(), command.canExecute(),
                        command.canManage()));
        var receipt = new CanvasCommandReceipt(UUID.randomUUID().toString(), command.requestId(), canvasId,
                document.revision() + 1, document.nextEventSequence(), EMPTY);
        journal.append(actor.userId(), "user", null, null, hash, audit, receipt, EMPTY, "canvas.members.changed");
        documents.advance(document);
        return receipt;
    }

    public Person person(long userId) {
        return Optional.ofNullable(memberships.person(userId))
                .orElseThrow(() -> invalid("成员账号不存在"));
    }

    private Member member(CanvasDocument document, Person person, CanvasActor requester) {
        var granted = documents.memberPermissions(document.id(), person.userId()).orElse(null);
        CanvasPermissions effective;
        try {
            effective = access.requireView(document, actorFor(document, person.userId(), requester));
        } catch (CanvasException denied) {
            if (denied.getCode() != 403) {
                throw denied;
            }
            effective = new CanvasPermissions(false, false, false, false);
        }
        boolean owner = protectedOwner(document, person.userId(), requester);
        return new Member(person, effective, granted, owner ? "OWNER" : granted != null ? "CANVAS" : "PROJECT", owner);
    }

    private boolean protectedOwner(CanvasDocument document, long userId, CanvasActor requester) {
        if (document.ownerType() == 1 && document.ownerId() == userId) {
            return true;
        }
        if (document.projectId() == null) {
            return document.createdBy() == userId;
        }
        return documents.isProjectManager(document.projectId(), userId)
                || (document.createdBy() == userId && identities.canEditProject(document.projectId(),
                actorFor(document, userId, requester)));
    }

    private CanvasActor actorFor(CanvasDocument document, long userId, CanvasActor requester) {
        if (document.ownerType() == 2 && identities.activeTeamMember(userId, document.ownerId())) {
            return new CanvasActor(userId, document.ownerId());
        }
        if (requester.teamId() != null && identities.activeTeamMember(userId, requester.teamId())) {
            return new CanvasActor(userId, requester.teamId());
        }
        return new CanvasActor(userId, null);
    }

    private void requireEligible(CanvasDocument document, long userId, CanvasActor requester) {
        var target = actorFor(document, userId, requester);
        access.requireActorScope(target);
        if (document.projectId() != null) {
            try {
                access.requireProject(document.projectId(), target);
            } catch (CanvasException denied) {
                if (denied.getCode() != 403) {
                    throw denied;
                }
                throw invalid("此账号尚无项目访问权限，请先在项目设置中添加成员");
            }
        }
    }

    private CanvasDocument readable(String id, CanvasActor actor) {
        validation.id(id);
        var document = documents.require(id, false);
        access.requireView(document, actor);
        return document;
    }

    private CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }

    private CanvasException conflict(String message) {
        return new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, message, Map.of());
    }
}
