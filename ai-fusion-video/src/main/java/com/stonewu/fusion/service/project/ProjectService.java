package com.stonewu.fusion.service.project;

import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.project.ProjectMember;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.mapper.asset.AssetItemMapper;
import com.stonewu.fusion.mapper.asset.AssetMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.project.ProjectMemberMapper;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import com.stonewu.fusion.mapper.script.ScriptSceneItemMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardItemMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardSceneMapper;
import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.service.project.dto.ProjectWorkspaceOverview;
import com.stonewu.fusion.service.storyboard.dto.StoryboardStatistics;
import com.stonewu.fusion.service.team.TeamService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * 项目服务
 */
@Service
@RequiredArgsConstructor
public class ProjectService {

    private static final int OWNER_TYPE_PERSONAL = 1;
    private static final int OWNER_TYPE_TEAM = 2;

    private final ProjectMapper projectMapper;
    private final ProjectMemberMapper memberMapper;
    private final AssetMapper assetMapper;
    private final AssetItemMapper assetItemMapper;
    private final ScriptMapper scriptMapper;
    private final ScriptEpisodeMapper scriptEpisodeMapper;
    private final ScriptSceneItemMapper scriptSceneItemMapper;
    private final StoryboardMapper storyboardMapper;
    private final StoryboardEpisodeMapper storyboardEpisodeMapper;
    private final StoryboardSceneMapper storyboardSceneMapper;
    private final StoryboardItemMapper storyboardItemMapper;
    private final TeamService teamService;

    @Cacheable(value = "project", key = "#id")
    public Project getById(Long id) {
        Project project = projectMapper.selectById(id);
        if (project == null) throw new BusinessException("项目不存在: " + id);
        return project;
    }

    public PageResult<Project> page(int pageNo, int pageSize) {
        return PageResult.of(projectMapper.selectPage(new Page<>(pageNo, pageSize), null));
    }

    @Cacheable(value = "project", key = "'owner:' + #ownerType + ':' + #ownerId")
    public List<Project> listByOwner(Integer ownerType, Long ownerId) {
        return projectMapper.selectList(new LambdaQueryWrapper<Project>()
                .eq(Project::getOwnerType, ownerType)
                .eq(Project::getOwnerId, ownerId)
                .orderByDesc(Project::getCreateTime));
    }

    public List<Project> listAccessibleByUser(Long userId) {
        Long currentTeamId = teamService.getCurrentTeamIdByUser(userId);
        if (currentTeamId == null) {
            return listByOwner(OWNER_TYPE_PERSONAL, userId);
        }
        List<Long> memberUserIds = teamService.listMemberUserIds(currentTeamId);
        return projectMapper.selectList(new LambdaQueryWrapper<Project>()
                .and(wrapper -> wrapper
                        .and(teamOwned -> teamOwned
                                .eq(Project::getOwnerType, OWNER_TYPE_TEAM)
                    .eq(Project::getOwnerId, currentTeamId))
                        .or(memberOwned -> memberOwned
                                .eq(Project::getOwnerType, OWNER_TYPE_PERSONAL)
                                .in(Project::getOwnerId, memberUserIds)))
                .orderByDesc(Project::getCreateTime));
    }

    public boolean canAccessProject(Long projectId, Long userId) {
        return canAccessProject(getById(projectId), userId);
    }

    public boolean canAccessProject(Project project, Long userId) {
        if (project == null) {
            return false;
        }
        if (OWNER_TYPE_PERSONAL == project.getOwnerType() && userId.equals(project.getOwnerId())) {
            return true;
        }
        if (isMember(project.getId(), userId)) {
            return true;
        }
        Long currentTeamId = teamService.getCurrentTeamIdByUser(userId);
        if (currentTeamId == null) {
            return false;
        }
        if (OWNER_TYPE_TEAM == project.getOwnerType() && currentTeamId.equals(project.getOwnerId())) {
            return true;
        }
        return OWNER_TYPE_PERSONAL == project.getOwnerType()
                && teamService.listMemberUserIds(currentTeamId).contains(project.getOwnerId());
    }

    public ProjectWorkspaceOverview getWorkspaceOverview(Long projectId) {
        getById(projectId);
        Script script = findEffectiveScript(projectId);
        Storyboard storyboard = findEffectiveStoryboard(projectId, script);
        return buildWorkspaceOverview(script, storyboard);
    }

    @CacheEvict(value = { "project", "script", "storyboard" }, allEntries = true)
    @Transactional
    public ProjectWorkspaceOverview initializeWorkspace(Long projectId) {
        Project project = lockProject(projectId);
        Script script = findEffectiveScript(projectId);
        if (script == null) {
            script = createProjectScript(project);
        }

        Storyboard storyboard = findEffectiveStoryboard(projectId, script);
        if (storyboard == null) {
            storyboard = createProjectStoryboard(project, script);
        }
        return buildWorkspaceOverview(script, storyboard);
    }

    private ProjectWorkspaceOverview buildWorkspaceOverview(Script script, Storyboard storyboard) {

        long scriptEpisodeCount = script == null ? 0 : scriptEpisodeMapper.selectCount(
                new LambdaQueryWrapper<ScriptEpisode>().eq(ScriptEpisode::getScriptId, script.getId()));
        long scriptSceneCount = script == null ? 0 : scriptSceneItemMapper.selectCount(
                new LambdaQueryWrapper<ScriptSceneItem>().eq(ScriptSceneItem::getScriptId, script.getId()));
        StoryboardStatistics storyboardStatistics = storyboard == null
                ? new StoryboardStatistics(0, 0, 0)
                : new StoryboardStatistics(
                        storyboardEpisodeMapper.selectCount(new LambdaQueryWrapper<StoryboardEpisode>()
                                .eq(StoryboardEpisode::getStoryboardId, storyboard.getId())),
                        storyboardSceneMapper.selectCount(new LambdaQueryWrapper<StoryboardScene>()
                                .eq(StoryboardScene::getStoryboardId, storyboard.getId())),
                        storyboardItemMapper.selectCount(new LambdaQueryWrapper<StoryboardItem>()
                                .eq(StoryboardItem::getStoryboardId, storyboard.getId())));
        return new ProjectWorkspaceOverview(
                script,
                scriptEpisodeCount,
                scriptSceneCount,
                storyboard,
                storyboardStatistics);
    }

    @CacheEvict(value = { "project", "script", "storyboard" }, allEntries = true)
    @Transactional
    public Project create(Project project) {
        validateProjectMediaUrls(project);
        applyCurrentTeamOwnership(project);
        projectMapper.insert(project);
        Script script = createProjectScript(project);
        createProjectStoryboard(project, script);
        return project;
    }

    @CacheEvict(value = { "project", "script", "storyboard" }, allEntries = true)
    @Transactional
    public Project update(Project project) {
        getById(project.getId());
        validateProjectMediaUrls(project);
        projectMapper.updateById(project);
        if (StrUtil.isNotBlank(project.getName())) {
            scriptMapper.update(null, new LambdaUpdateWrapper<Script>()
                    .eq(Script::getProjectId, project.getId())
                    .set(Script::getTitle, project.getName()));
            storyboardMapper.update(null, new LambdaUpdateWrapper<Storyboard>()
                    .eq(Storyboard::getProjectId, project.getId())
                    .set(Storyboard::getTitle, project.getName()));
        }
        return projectMapper.selectById(project.getId());
    }

    @CacheEvict(
            value = {
                    "project", "projectMember", "asset", "assetItem", "script", "episode", "scene",
                    "storyboard", "storyboardEpisode", "storyboardScene", "storyboardItem", "storyboardStatistics"
            },
            allEntries = true)
    @Transactional
    public void delete(Long id) {
        List<Storyboard> storyboards = storyboardMapper.selectList(
                new LambdaQueryWrapper<Storyboard>().eq(Storyboard::getProjectId, id));
        for (Storyboard storyboard : storyboards) {
            storyboardItemMapper.delete(new LambdaQueryWrapper<StoryboardItem>()
                    .eq(StoryboardItem::getStoryboardId, storyboard.getId()));
            storyboardSceneMapper.delete(new LambdaQueryWrapper<StoryboardScene>()
                    .eq(StoryboardScene::getStoryboardId, storyboard.getId()));
            storyboardEpisodeMapper.delete(new LambdaQueryWrapper<StoryboardEpisode>()
                    .eq(StoryboardEpisode::getStoryboardId, storyboard.getId()));
        }
        storyboardMapper.delete(new LambdaQueryWrapper<Storyboard>()
                .eq(Storyboard::getProjectId, id));

        List<Script> scripts = scriptMapper.selectList(
                new LambdaQueryWrapper<Script>().eq(Script::getProjectId, id));
        for (Script script : scripts) {
            scriptSceneItemMapper.delete(new LambdaQueryWrapper<ScriptSceneItem>()
                    .eq(ScriptSceneItem::getScriptId, script.getId()));
            scriptEpisodeMapper.delete(new LambdaQueryWrapper<ScriptEpisode>()
                    .eq(ScriptEpisode::getScriptId, script.getId()));
        }
        scriptMapper.delete(new LambdaQueryWrapper<Script>()
                .eq(Script::getProjectId, id));

        // 级联删除项目下的所有资产及其子项
        List<Asset> assets = assetMapper.selectList(
                new LambdaQueryWrapper<Asset>().eq(Asset::getProjectId, id));
        for (Asset asset : assets) {
            assetItemMapper.delete(
                    new LambdaQueryWrapper<AssetItem>().eq(AssetItem::getAssetId, asset.getId()));
        }
        if (!assets.isEmpty()) {
            assetMapper.delete(new LambdaQueryWrapper<Asset>().eq(Asset::getProjectId, id));
        }
        // 删除项目成员
        memberMapper.delete(new LambdaQueryWrapper<ProjectMember>().eq(ProjectMember::getProjectId, id));
        // 删除项目本身
        projectMapper.deleteById(id);
    }

    // ========== 成员管理 ==========

    public boolean isMember(Long projectId, Long userId) {
        return memberMapper.exists(new LambdaQueryWrapper<ProjectMember>()
                .eq(ProjectMember::getProjectId, projectId)
                .eq(ProjectMember::getUserId, userId));
    }

    @Cacheable(value = "projectMember", key = "#projectId")
    public List<ProjectMember> listMembers(Long projectId) {
        return memberMapper.selectList(
                new LambdaQueryWrapper<ProjectMember>().eq(ProjectMember::getProjectId, projectId));
    }

    @CacheEvict(value = "projectMember", allEntries = true)
    @Transactional
    public ProjectMember addMember(Long projectId, Long userId, Integer role) {
        if (isMember(projectId, userId)) {
            throw new BusinessException("该用户已是项目成员");
        }
        ProjectMember member = ProjectMember.builder()
                .projectId(projectId)
                .userId(userId)
                .role(role)
                .build();
        memberMapper.insert(member);
        return member;
    }

    @CacheEvict(value = "projectMember", allEntries = true)
    @Transactional
    public void removeMember(Long projectId, Long userId) {
        memberMapper.delete(new LambdaQueryWrapper<ProjectMember>()
                .eq(ProjectMember::getProjectId, projectId)
                .eq(ProjectMember::getUserId, userId));
    }

    private void validateProjectMediaUrls(Project project) {
        if (project == null) {
            return;
        }
        rejectDataUrl(project.getCoverUrl(), "coverUrl");
        rejectDataUrl(project.getArtStyleImageUrl(), "artStyleImageUrl");
    }

    private void applyCurrentTeamOwnership(Project project) {
        Long currentUserId = SecurityUtils.getCurrentUserId();
        if (currentUserId == null) {
            return;
        }
        TeamService.OwnerScope ownerScope = teamService.getRequiredCurrentOwnerScopeByUser(currentUserId);
        project.setOwnerType(ownerScope.getOwnerType());
        project.setOwnerId(ownerScope.getOwnerId());
    }

    private Project lockProject(Long projectId) {
        Project project = projectMapper.selectOne(new LambdaQueryWrapper<Project>()
                .eq(Project::getId, projectId)
                .last("FOR UPDATE"));
        if (project == null) {
            throw new BusinessException("项目不存在: " + projectId);
        }
        return project;
    }

    private Script findEffectiveScript(Long projectId) {
        return scriptMapper.selectOne(new LambdaQueryWrapper<Script>()
                .eq(Script::getProjectId, projectId)
                .orderByDesc(Script::getCreateTime)
                .orderByDesc(Script::getId)
                .last("LIMIT 1"));
    }

    private Storyboard findEffectiveStoryboard(Long projectId, Script script) {
        if (script == null) {
            return null;
        }
        return storyboardMapper.selectOne(new LambdaQueryWrapper<Storyboard>()
                .eq(Storyboard::getProjectId, projectId)
                .eq(Storyboard::getScriptId, script.getId())
                .orderByDesc(Storyboard::getCreateTime)
                .orderByDesc(Storyboard::getId)
                .last("LIMIT 1"));
    }

    private Script createProjectScript(Project project) {
        Script script = Script.builder()
                .projectId(project.getId())
                .title(project.getName())
                .scope(project.getScope())
                .ownerType(project.getOwnerType())
                .ownerId(project.getOwnerId())
                .build();
        scriptMapper.insert(script);
        return script;
    }

    private Storyboard createProjectStoryboard(Project project, Script script) {
        Storyboard storyboard = Storyboard.builder()
                .projectId(project.getId())
                .scriptId(script.getId())
                .title(project.getName())
                .scope(project.getScope())
                .ownerType(project.getOwnerType())
                .ownerId(project.getOwnerId())
                .build();
        storyboardMapper.insert(storyboard);
        return storyboard;
    }

    private void rejectDataUrl(String rawUrl, String fieldName) {
        if (StrUtil.isNotBlank(rawUrl) && StrUtil.startWithIgnoreCase(rawUrl.trim(), "data:")) {
            throw new BusinessException(fieldName + " 不支持 base64，请先调用 /api/storage/upload 上传二进制文件");
        }
    }
}
