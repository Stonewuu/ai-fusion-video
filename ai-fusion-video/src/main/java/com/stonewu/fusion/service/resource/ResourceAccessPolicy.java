package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.team.TeamMember;
import com.stonewu.fusion.mapper.team.TeamMemberMapper;
import com.stonewu.fusion.mapper.asset.AssetMapper;
import com.stonewu.fusion.mapper.asset.AssetItemMapper;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;

/**
 * 项目资源与独立素材共用的实时授权；所有者 ID 必须连同所有者类型判断。
 */
@Component
@RequiredArgsConstructor
public class ResourceAccessPolicy {
    private final CanvasAccessPolicy projects;
    private final TeamMemberMapper teamMembers;
    private final AssetMapper assets;
    private final AssetItemMapper items;

    public void requireEdit(ResourceHeader header, CanvasActor actor) {
        require(header, actor);
        if (header.projectId() != null) {
            projects.requireProjectEditor(header.projectId(), actor);
            return;
        }
        Owner owner = owner(header);
        if (Objects.equals(owner.type(), 1) && Objects.equals(owner.id(), actor.userId())) {
            return;
        }
        if (actor.teamId() != null && Objects.equals(owner.type(), 2) && Objects.equals(owner.id(), actor.teamId())) {
            return;
        }
        throw denied();
    }

    public boolean canEdit(ResourceHeader header, CanvasActor actor) {
        try {
            requireEdit(header, actor);
            return true;
        } catch (CanvasException denied) {
            if (denied.getCode() == 403) {
                return false;
            }
            throw denied;
        }
    }

    public void require(ResourceHeader header, CanvasActor actor) {
        require(header, actor, false);
    }

    public void requireHistorical(ResourceHeader header, CanvasActor actor) {
        require(header, actor, true);
    }

    private void require(ResourceHeader header, CanvasActor actor, boolean historical) {
        if (header.projectId() != null) {
            projects.requireProject(header.projectId(), actor);
            return;
        }
        projects.requireActorScope(actor);
        if (header.resourceType() != CanvasValueType.ASSET && header.resourceType() != CanvasValueType.ASSET_ITEM) {
            throw denied();
        }
        Owner owner = owner(header, historical);
        if (Objects.equals(owner.type(), 1) && Objects.equals(owner.id(), actor.userId())) {
            return;
        }
        if (actor.teamId() != null && Objects.equals(owner.type(), 2) && Objects.equals(owner.id(), actor.teamId())) {
            return;
        }
        if (actor.teamId() != null && Objects.equals(owner.type(), 1) && owner.id() != null) {
            long members = teamMembers.selectCount(new LambdaQueryWrapper<TeamMember>()
                    .eq(TeamMember::getTeamId, actor.teamId()).eq(TeamMember::getUserId, owner.id())
                    .eq(TeamMember::getStatus, 1));
            if (members > 0) {
                return;
            }
        }
        throw denied();
    }

    private Owner owner(ResourceHeader header) {
        return owner(header, false);
    }

    private Owner owner(ResourceHeader header, boolean historical) {
        long assetId = header.resourceType() == CanvasValueType.ASSET ? header.resourceId()
                : items.parentAssetId(header.resourceId());
        var asset = assets.selectOwnerRecord(assetId, historical || header.deleted());
        if (asset == null) {
            throw new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED, "主资产已删除", Map.of());
        }
        return new Owner(asset.getOwnerType(), asset.getOwnerId());
    }

    private CanvasException denied() {
        return new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "无权访问此业务资源", Map.of());
    }

    private record Owner(Integer type, Long id) {
    }
}
