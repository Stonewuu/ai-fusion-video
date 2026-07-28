package com.stonewu.fusion.service.storyboard;

import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardItemMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardSceneMapper;
import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.service.storyboard.dto.StoryboardItemAssetsPatch;
import com.stonewu.fusion.service.storyboard.dto.StoryboardStatistics;
import com.stonewu.fusion.service.team.TeamService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.util.List;

/**
 * 分镜脚本服务（含分镜集、分镜场次、分镜条目管理）
 */
@Service
@RequiredArgsConstructor
public class StoryboardService {

    private final StoryboardMapper storyboardMapper;
    private final StoryboardEpisodeMapper episodeMapper;
    private final StoryboardSceneMapper sceneMapper;
    private final StoryboardItemMapper itemMapper;
    private final ScriptEpisodeMapper scriptEpisodeMapper;
    private final TeamService teamService;

    // ========== 分镜脚本 ==========

    @Cacheable(value = "storyboard", key = "#id")
    public Storyboard getById(Long id) {
        Storyboard sb = storyboardMapper.selectById(id);
        if (sb == null) throw new BusinessException("分镜脚本不存在: " + id);
        return sb;
    }

    @Cacheable(value = "storyboard", key = "'project:' + #projectId")
    public List<Storyboard> listByProject(Long projectId) {
        return storyboardMapper.selectList(new LambdaQueryWrapper<Storyboard>()
                .eq(Storyboard::getProjectId, projectId)
                .orderByDesc(Storyboard::getCreateTime));
    }

    @CacheEvict(value = "storyboard", allEntries = true)
    @Transactional
    public Storyboard create(Storyboard storyboard) {
        applyCurrentTeamOwnership(storyboard);
        storyboardMapper.insert(storyboard);
        return storyboard;
    }

    @CacheEvict(value = "storyboard", allEntries = true)
    @Transactional
    public Storyboard update(Storyboard storyboard) {
        getById(storyboard.getId());
        storyboardMapper.updateById(storyboard);
        return storyboardMapper.selectById(storyboard.getId());
    }

    @CacheEvict(value = { "storyboard", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public void delete(Long id) {
        storyboardMapper.deleteById(id);
    }

    @Cacheable(value = "storyboardStatistics", key = "#storyboardId")
    public StoryboardStatistics getStatistics(Long storyboardId) {
        long episodeCount = episodeMapper.selectCount(new LambdaQueryWrapper<StoryboardEpisode>()
                .eq(StoryboardEpisode::getStoryboardId, storyboardId));
        long sceneCount = sceneMapper.selectCount(new LambdaQueryWrapper<StoryboardScene>()
                .eq(StoryboardScene::getStoryboardId, storyboardId));
        long itemCount = itemMapper.selectCount(new LambdaQueryWrapper<StoryboardItem>()
                .eq(StoryboardItem::getStoryboardId, storyboardId));
        return new StoryboardStatistics(episodeCount, sceneCount, itemCount);
    }

    // ========== 分镜集 ==========

    @Cacheable(value = "storyboardEpisode", key = "#id")
    public StoryboardEpisode getEpisodeById(Long id) {
        StoryboardEpisode ep = episodeMapper.selectById(id);
        if (ep == null) throw new BusinessException("分镜集不存在: " + id);
        return ep;
    }

    @Cacheable(value = "storyboardEpisode", key = "'storyboard:' + #storyboardId")
    public List<StoryboardEpisode> listEpisodes(Long storyboardId) {
        return episodeMapper.selectList(new LambdaQueryWrapper<StoryboardEpisode>()
                .eq(StoryboardEpisode::getStoryboardId, storyboardId)
                .orderByAsc(StoryboardEpisode::getSortOrder)
                .orderByAsc(StoryboardEpisode::getEpisodeNumber));
    }

    @CacheEvict(value = { "storyboardEpisode", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public StoryboardEpisode createEpisode(StoryboardEpisode episode) {
        if (episode.getScriptEpisodeId() != null) {
            validateScriptEpisodeBinding(episode.getStoryboardId(), episode.getScriptEpisodeId(), null);
        }
        if (episode.getDeletedId() == null) {
            episode.setDeletedId(0L);
        }
        episodeMapper.insert(episode);
        return episode;
    }

    @CacheEvict(value = { "storyboardEpisode", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public StoryboardEpisode updateEpisode(StoryboardEpisode episode) {
        StoryboardEpisode existing = getEpisodeById(episode.getId());
        if (episode.getScriptEpisodeId() != null
                && !episode.getScriptEpisodeId().equals(existing.getScriptEpisodeId())) {
            validateScriptEpisodeBinding(existing.getStoryboardId(), episode.getScriptEpisodeId(), episode.getId());
        }
        episodeMapper.updateById(episode);
        return episodeMapper.selectById(episode.getId());
    }

    @CacheEvict(
            value = { "storyboardEpisode", "storyboardScene", "storyboardItem", "storyboardStatistics" },
            allEntries = true)
    @Transactional
    public void deleteEpisode(Long id) {
        StoryboardEpisode episode = getEpisodeById(id);
        if (episode.getDeletedId() == null || episode.getDeletedId() == 0L) {
            StoryboardEpisode update = new StoryboardEpisode();
            update.setId(id);
            update.setDeletedId(id);
            episodeMapper.updateById(update);
        }
        // 级联删除分镜集下的所有场次 and 条目
        clearEpisodeContent(id);

        episodeMapper.deleteById(id);
    }

    /**
     * 根据剧本分集查找已绑定的分镜集。
     *
     * @param storyboardId    分镜脚本ID
     * @param scriptEpisodeId 剧本分集ID
     * @return 已绑定的分镜集，不存在则返回 null
     */
    public StoryboardEpisode getEpisodeByScriptEpisode(Long storyboardId, Long scriptEpisodeId) {
        if (storyboardId == null || scriptEpisodeId == null) {
            return null;
        }
        return episodeMapper.selectOne(new LambdaQueryWrapper<StoryboardEpisode>()
                .eq(StoryboardEpisode::getStoryboardId, storyboardId)
                .eq(StoryboardEpisode::getScriptEpisodeId, scriptEpisodeId));
    }

    /**
     * 绑定分镜集和剧本分集。
     *
     * @param episodeId       分镜集ID
     * @param scriptEpisodeId 剧本分集ID
     * @return 绑定后的分镜集
     */
    @CacheEvict(value = "storyboardEpisode", allEntries = true)
    @Transactional
    public StoryboardEpisode bindScriptEpisode(Long episodeId, Long scriptEpisodeId) {
        StoryboardEpisode episode = getEpisodeById(episodeId);
        validateScriptEpisodeBinding(episode.getStoryboardId(), scriptEpisodeId, episodeId);

        StoryboardEpisode update = new StoryboardEpisode();
        update.setId(episodeId);
        update.setScriptEpisodeId(scriptEpisodeId);
        episodeMapper.updateById(update);
        return episodeMapper.selectById(episodeId);
    }

    /**
     * 保存或复用剧本分集对应的分镜集。
     *
     * @param storyboardId     分镜脚本ID
     * @param scriptEpisodeId  剧本分集ID
     * @param episodeNumber    集号
     * @param title            集标题
     * @param synopsis         集梗概
     * @return 已存在或新创建的分镜集
     */
    @CacheEvict(value = { "storyboardEpisode", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public StoryboardEpisode saveEpisodeForScript(Long storyboardId,
                                                  Long scriptEpisodeId,
                                                  Integer episodeNumber,
                                                  String title,
                                                  String synopsis) {
        ScriptEpisode scriptEpisode = validateScriptEpisodeBelongsToStoryboard(storyboardId, scriptEpisodeId);
        StoryboardEpisode existing = getEpisodeByScriptEpisode(storyboardId, scriptEpisodeId);
        Integer resolvedEpisodeNumber = episodeNumber != null ? episodeNumber : scriptEpisode.getEpisodeNumber();

        if (existing != null) {
            StoryboardEpisode update = new StoryboardEpisode();
            update.setId(existing.getId());
            update.setEpisodeNumber(resolvedEpisodeNumber);
            update.setTitle(title != null ? title : scriptEpisode.getTitle());
            update.setSynopsis(synopsis != null ? synopsis : scriptEpisode.getSynopsis());
            update.setSortOrder(resolvedEpisodeNumber != null ? resolvedEpisodeNumber - 1 : existing.getSortOrder());
            update.setStatus(1);
            episodeMapper.updateById(update);
            return episodeMapper.selectById(existing.getId());
        }

        StoryboardEpisode episode = StoryboardEpisode.builder()
                .storyboardId(storyboardId)
                .scriptEpisodeId(scriptEpisodeId)
                .episodeNumber(resolvedEpisodeNumber)
                .title(title != null ? title : scriptEpisode.getTitle())
                .synopsis(synopsis != null ? synopsis : scriptEpisode.getSynopsis())
                .sortOrder(resolvedEpisodeNumber != null ? resolvedEpisodeNumber - 1 : 0)
                .status(1)
                .deletedId(0L)
                .build();
        episodeMapper.insert(episode);
        return episode;
    }

    /**
     * 清空指定分镜集下的场次和镜头，保留分镜集及其剧本分集绑定关系。
     *
     * @param episodeId 分镜集ID
     */
    @CacheEvict(value = { "storyboardScene", "storyboardItem", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public void clearEpisodeContent(Long episodeId) {
        getEpisodeById(episodeId);
        List<Long> sceneIds = listScenesByEpisode(episodeId).stream()
                .map(StoryboardScene::getId)
                .toList();

        LambdaQueryWrapper<StoryboardItem> itemWrapper = new LambdaQueryWrapper<StoryboardItem>()
                .and(wrapper -> {
                    wrapper.eq(StoryboardItem::getStoryboardEpisodeId, episodeId);
                    if (!sceneIds.isEmpty()) {
                        wrapper.or().in(StoryboardItem::getStoryboardSceneId, sceneIds);
                    }
                });
        itemMapper.delete(itemWrapper);
        sceneMapper.delete(new LambdaQueryWrapper<StoryboardScene>()
                .eq(StoryboardScene::getEpisodeId, episodeId));
    }

    /**
     * 校验剧本分集是否可绑定到指定分镜脚本。
     *
     * @param storyboardId       分镜脚本ID
     * @param scriptEpisodeId    剧本分集ID
     * @param excludedEpisodeId  更新当前分镜集时需要排除的分镜集ID，新建时为 null
     * @return 校验通过后的剧本分集
     */
    private ScriptEpisode validateScriptEpisodeBinding(Long storyboardId,
                                                       Long scriptEpisodeId,
                                                       Long excludedEpisodeId) {
        ScriptEpisode scriptEpisode = validateScriptEpisodeBelongsToStoryboard(storyboardId, scriptEpisodeId);

        LambdaQueryWrapper<StoryboardEpisode> wrapper = new LambdaQueryWrapper<StoryboardEpisode>()
                .eq(StoryboardEpisode::getStoryboardId, storyboardId)
                .eq(StoryboardEpisode::getScriptEpisodeId, scriptEpisodeId);
        if (excludedEpisodeId != null) {
            wrapper.ne(StoryboardEpisode::getId, excludedEpisodeId);
        }
        StoryboardEpisode duplicated = episodeMapper.selectOne(wrapper);
        if (duplicated != null) {
            throw new BusinessException("该剧本分集已绑定到分镜集: " + duplicated.getId());
        }
        return scriptEpisode;
    }

    /**
     * 校验剧本分集是否属于分镜脚本关联的剧本。
     *
     * @param storyboardId    分镜脚本ID
     * @param scriptEpisodeId 剧本分集ID
     * @return 校验通过后的剧本分集
     */
    private ScriptEpisode validateScriptEpisodeBelongsToStoryboard(Long storyboardId, Long scriptEpisodeId) {
        if (storyboardId == null) {
            throw new BusinessException("分镜ID不能为空，无法绑定剧本分集");
        }
        if (scriptEpisodeId == null) {
            throw new BusinessException("剧本分集ID不能为空");
        }

        Storyboard storyboard = getById(storyboardId);
        if (storyboard.getScriptId() == null) {
            throw new BusinessException("分镜未关联剧本，无法绑定剧本分集");
        }

        ScriptEpisode scriptEpisode = scriptEpisodeMapper.selectById(scriptEpisodeId);
        if (scriptEpisode == null) {
            throw new BusinessException("剧本分集不存在: " + scriptEpisodeId);
        }
        if (!storyboard.getScriptId().equals(scriptEpisode.getScriptId())) {
            throw new BusinessException("剧本分集不属于当前分镜关联的剧本");
        }
        return scriptEpisode;
    }

    // ========== 分镜场次 ==========

    @Cacheable(value = "storyboardScene", key = "#id")
    public StoryboardScene getSceneById(Long id) {
        StoryboardScene scene = sceneMapper.selectById(id);
        if (scene == null) throw new BusinessException("分镜场次不存在: " + id);
        return scene;
    }

    @Cacheable(value = "storyboardScene", key = "'episode:' + #episodeId")
    public List<StoryboardScene> listScenesByEpisode(Long episodeId) {
        return sceneMapper.selectList(new LambdaQueryWrapper<StoryboardScene>()
                .eq(StoryboardScene::getEpisodeId, episodeId)
                .orderByAsc(StoryboardScene::getSortOrder));
    }

    public List<StoryboardScene> listScenesByStoryboard(Long storyboardId) {
        return sceneMapper.selectList(new LambdaQueryWrapper<StoryboardScene>()
                .eq(StoryboardScene::getStoryboardId, storyboardId)
                .orderByAsc(StoryboardScene::getSortOrder));
    }

    @CacheEvict(value = { "storyboardScene", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public StoryboardScene createScene(StoryboardScene scene) {
        sceneMapper.insert(scene);
        return scene;
    }

    @CacheEvict(value = { "storyboardScene", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public StoryboardScene updateScene(StoryboardScene scene) {
        getSceneById(scene.getId());
        sceneMapper.updateById(scene);
        return sceneMapper.selectById(scene.getId());
    }

    @CacheEvict(value = { "storyboardScene", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public void deleteScene(Long id) {
        sceneMapper.deleteById(id);
    }

    // ========== 分镜条目 ==========

    @Cacheable(value = "storyboardItem", key = "#id")
    public StoryboardItem getItemById(Long id) {
        StoryboardItem item = itemMapper.selectById(id);
        if (item == null) throw new BusinessException("分镜条目不存在: " + id);
        return item;
    }

    @Cacheable(value = "storyboardItem", key = "'storyboard:' + #storyboardId")
    public List<StoryboardItem> listItems(Long storyboardId) {
        return itemMapper.selectList(new LambdaQueryWrapper<StoryboardItem>()
                .eq(StoryboardItem::getStoryboardId, storyboardId)
                .orderByAsc(StoryboardItem::getSortOrder));
    }

    public List<StoryboardItem> listItemsByScene(Long sceneId) {
        return itemMapper.selectList(new LambdaQueryWrapper<StoryboardItem>()
                .eq(StoryboardItem::getStoryboardSceneId, sceneId)
                .orderByAsc(StoryboardItem::getSortOrder));
    }

    @CacheEvict(value = { "storyboardItem", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public StoryboardItem createItem(StoryboardItem item) {
        itemMapper.insert(item);
        return item;
    }

    private void applyCurrentTeamOwnership(Storyboard storyboard) {
        Long currentUserId = SecurityUtils.getCurrentUserId();
        if (currentUserId == null) {
            return;
        }
        TeamService.OwnerScope ownerScope = teamService.getRequiredCurrentOwnerScopeByUser(currentUserId);
        storyboard.setOwnerType(ownerScope.getOwnerType());
        storyboard.setOwnerId(ownerScope.getOwnerId());
    }

    @CacheEvict(value = { "storyboardItem", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public StoryboardItem updateItem(StoryboardItem item) {
        getItemById(item.getId());
        itemMapper.updateById(item);
        return itemMapper.selectById(item.getId());
    }

    /**
     * 局部更新分镜条目的角色、场景和道具关联。
     *
     * <p>字段缺省时不修改；角色、道具空数组写入 JSON 空数组；场景显式 null 写入数据库 null。</p>
     */
    @CacheEvict(value = "storyboardItem", allEntries = true)
    @Transactional
    public StoryboardItem updateItemAssets(Long itemId, StoryboardItemAssetsPatch patch) {
        StoryboardItem existing = getItemById(itemId);
        if (!patch.hasUpdates()) {
            return existing;
        }
        if (patch.characterIdsPresent() && patch.characterIds() == null) {
            throw new BusinessException("角色关联清空时请传空数组，不能传 null");
        }
        if (patch.propIdsPresent() && patch.propIds() == null) {
            throw new BusinessException("道具关联清空时请传空数组，不能传 null");
        }

        UpdateWrapper<StoryboardItem> wrapper = new UpdateWrapper<StoryboardItem>()
                .eq("id", itemId);
        if (patch.characterIdsPresent()) {
            wrapper.set("character_ids", JSONUtil.toJsonStr(patch.characterIds()));
        }
        if (patch.sceneAssetItemIdPresent()) {
            wrapper.set("scene_asset_item_id", patch.sceneAssetItemId());
        }
        if (patch.propIdsPresent()) {
            wrapper.set("prop_ids", JSONUtil.toJsonStr(patch.propIds()));
        }
        itemMapper.update(null, wrapper);
        return itemMapper.selectById(itemId);
    }

    /**
     * 更新分镜条目的首帧或尾帧字段。
     *
     * @param itemId    分镜条目ID
     * @param frameType 帧类型：first-首帧，last-尾帧
     * @param imageUrl  图片URL，允许为空，空值表示清空对应帧
     * @param prompt    AI生成提示词，允许为空，空值表示清空对应提示词
     * @return 更新后的分镜条目
     */
    @CacheEvict(value = "storyboardItem", allEntries = true)
    @Transactional
    public StoryboardItem updateItemFrame(Long itemId, String frameType, String imageUrl, String prompt) {
        getItemById(itemId);
        String normalizedFrameType = normalizeFrameType(frameType);
        String normalizedImageUrl = StringUtils.hasText(imageUrl) ? imageUrl.trim() : null;
        String normalizedPrompt = StringUtils.hasText(prompt) ? prompt.trim() : null;
        UpdateWrapper<StoryboardItem> wrapper = new UpdateWrapper<StoryboardItem>()
                .eq("id", itemId);
        if ("first".equals(normalizedFrameType)) {
            wrapper.set("first_frame_image_url", normalizedImageUrl)
                    .set("first_frame_prompt", normalizedPrompt);
        } else {
            wrapper.set("last_frame_image_url", normalizedImageUrl)
                    .set("last_frame_prompt", normalizedPrompt);
        }
        itemMapper.update(null, wrapper);
        return itemMapper.selectById(itemId);
    }

    /**
     * 规范化帧类型。
     *
     * @param frameType 原始帧类型
     * @return 规范化后的帧类型
     */
    public String normalizeFrameType(String frameType) {
        String normalizedFrameType = StringUtils.hasText(frameType) ? frameType.trim() : "";
        if ("first".equals(normalizedFrameType) || "last".equals(normalizedFrameType)) {
            return normalizedFrameType;
        }
        throw new BusinessException("帧类型仅支持 first 或 last");
    }

    @CacheEvict(value = { "storyboardItem", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public void deleteItem(Long id) {
        itemMapper.deleteById(id);
    }

    @CacheEvict(value = { "storyboardItem", "storyboardStatistics" }, allEntries = true)
    @Transactional
    public void batchCreateItems(List<StoryboardItem> items) {
        for (StoryboardItem item : items) {
            itemMapper.insert(item);
        }
    }

    @CacheEvict(value = "storyboardItem", allEntries = true)
    @Transactional
    public void batchUpdateItemSort(List<Long> ids) {
        if (ids == null || ids.isEmpty()) {
            return;
        }
        for (int i = 0; i < ids.size(); i++) {
            StoryboardItem item = new StoryboardItem();
            item.setId(ids.get(i));
            item.setSortOrder(i);
            itemMapper.updateById(item);
        }
    }
}
