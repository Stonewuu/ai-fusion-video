package com.stonewu.fusion.repository.canvas;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.canvas.CanvasAccessMapper;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * 权限实时读取，不缓存；异步读写使用请求内固定的身份。
 */
@Repository
@RequiredArgsConstructor
public class CanvasAccessRepository {
    private final UserMapper users;
    private final CanvasAccessMapper access;

    public record ProjectScope(long id, int ownerType, long ownerId) {
    }

    public boolean userEnabled(long userId) {
        return users.exists(Wrappers.<User>lambdaQuery().eq(User::getId, userId).eq(User::getStatus, 1));
    }

    public boolean activeTeamMember(long userId, long teamId) {
        return access.activeTeamMember(userId, teamId) > 0;
    }

    public Optional<ProjectScope> project(long projectId, CanvasActor actor) {
        return Optional.ofNullable(access.project(projectId, actor.userId(), actor.teamId()));
    }

    public List<Long> readableProjects(CanvasActor actor) {
        return access.readableProjects(actor.userId(), actor.teamId());
    }

    public boolean canEditProject(long projectId, CanvasActor actor) {
        return access.canEditProject(projectId, actor.userId(), actor.teamId()) == 1;
    }

    public boolean referenceBelongsToProject(CanvasResourceReference reference, long projectId) {
        switch (reference.type()) {
            case PROJECT, SCRIPT, SCRIPT_EPISODE, SCRIPT_SCENE, STORYBOARD, STORYBOARD_EPISODE,
                 STORYBOARD_SCENE, SHOT, ASSET, ASSET_ITEM -> {
                return access.referenceBelongs(reference.type(), reference.id(), projectId) == 1;
            }
            default -> throw new IllegalArgumentException("不是业务对象引用");
        }
    }
}
