package com.stonewu.fusion.service.storyboard;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardItemMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardSceneMapper;
import com.stonewu.fusion.service.storyboard.dto.StoryboardStatistics;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 供尚在迁移的合成服务与资源守卫使用的只读查询；业务修改统一进入资源命令。
 */
@Service
@RequiredArgsConstructor
public class StoryboardService {
    private final StoryboardMapper storyboardMapper;
    private final StoryboardEpisodeMapper episodeMapper;
    private final StoryboardSceneMapper sceneMapper;
    private final StoryboardItemMapper itemMapper;
    private final ScriptMapper scriptMapper;

    @Cacheable(value = "storyboard", key = "#id", unless = "#result == null")
    public Storyboard getById(Long id) {
        Storyboard sb = storyboardMapper.selectById(id);
        if (sb == null) {
            throw new BusinessException("分镜脚本不存在: " + id);
        }
        return sb;
    }

    @Cacheable(value = "storyboard", key = "'project:' + #projectId", unless = "#result == null")
    public Storyboard getByProjectId(Long projectId) {
        Script script = scriptMapper.selectPage(new Page<>(1, 1, false), new LambdaQueryWrapper<Script>()
                .eq(Script::getProjectId, projectId)
                .orderByDesc(Script::getCreateTime)
                .orderByDesc(Script::getId)).getRecords().stream().findFirst().orElse(null);
        if (script == null) {
            return null;
        }
        return storyboardMapper.selectPage(new Page<>(1, 1, false), new LambdaQueryWrapper<Storyboard>()
                .eq(Storyboard::getProjectId, projectId)
                .eq(Storyboard::getScriptId, script.getId())
                .orderByDesc(Storyboard::getCreateTime)
                .orderByDesc(Storyboard::getId)).getRecords().stream().findFirst().orElse(null);
    }

    @Cacheable(value = "storyboardStatistics", key = "#storyboardId", unless = "#result == null")
    public StoryboardStatistics getStatistics(Long storyboardId) {
        long episodeCount = episodeMapper.selectCount(new LambdaQueryWrapper<StoryboardEpisode>()
                .eq(StoryboardEpisode::getStoryboardId, storyboardId));
        long sceneCount = sceneMapper.selectCount(new LambdaQueryWrapper<StoryboardScene>()
                .eq(StoryboardScene::getStoryboardId, storyboardId));
        long itemCount = itemMapper.selectCount(new LambdaQueryWrapper<StoryboardItem>()
                .eq(StoryboardItem::getStoryboardId, storyboardId));
        return new StoryboardStatistics(episodeCount, sceneCount, itemCount);
    }

    @Cacheable(value = "storyboardEpisode", key = "#id", unless = "#result == null")
    public StoryboardEpisode getEpisodeById(Long id) {
        StoryboardEpisode ep = episodeMapper.selectById(id);
        if (ep == null) {
            throw new BusinessException("分镜集不存在: " + id);
        }
        return ep;
    }

    @Cacheable(value = "storyboardEpisode", key = "'storyboard:' + #storyboardId", unless = "#result == null")
    public List<StoryboardEpisode> listEpisodes(Long storyboardId) {
        return episodeMapper.selectList(new LambdaQueryWrapper<StoryboardEpisode>()
                .eq(StoryboardEpisode::getStoryboardId, storyboardId)
                .orderByAsc(StoryboardEpisode::getSortOrder)
                .orderByAsc(StoryboardEpisode::getEpisodeNumber));
    }

    @Cacheable(value = "storyboardEpisode", key = "'binding:' + #storyboardId + ':' + #scriptEpisodeId", unless = "#result == null")
    public StoryboardEpisode getEpisodeByScriptEpisode(Long storyboardId, Long scriptEpisodeId) {
        if (storyboardId == null || scriptEpisodeId == null) {
            return null;
        }
        return episodeMapper.selectOne(new LambdaQueryWrapper<StoryboardEpisode>()
                .eq(StoryboardEpisode::getStoryboardId, storyboardId)
                .eq(StoryboardEpisode::getScriptEpisodeId, scriptEpisodeId));
    }

    @Cacheable(value = "storyboardScene", key = "#id", unless = "#result == null")
    public StoryboardScene getSceneById(Long id) {
        StoryboardScene scene = sceneMapper.selectById(id);
        if (scene == null) {
            throw new BusinessException("分镜场次不存在: " + id);
        }
        return scene;
    }

    @Cacheable(value = "storyboardScene", key = "'episode:' + #episodeId", unless = "#result == null")
    public List<StoryboardScene> listScenesByEpisode(Long episodeId) {
        return sceneMapper.selectList(new LambdaQueryWrapper<StoryboardScene>()
                .eq(StoryboardScene::getEpisodeId, episodeId)
                .orderByAsc(StoryboardScene::getSortOrder));
    }

    @Cacheable(value = "storyboardScene", key = "'storyboard:' + #storyboardId", unless = "#result == null")
    public List<StoryboardScene> listScenesByStoryboard(Long storyboardId) {
        return sceneMapper.selectList(new LambdaQueryWrapper<StoryboardScene>()
                .eq(StoryboardScene::getStoryboardId, storyboardId)
                .orderByAsc(StoryboardScene::getSortOrder));
    }

    @Cacheable(value = "storyboardItem", key = "#id", unless = "#result == null")
    public StoryboardItem getItemById(Long id) {
        StoryboardItem item = itemMapper.selectById(id);
        if (item == null) {
            throw new BusinessException("分镜条目不存在: " + id);
        }
        return item;
    }

    @Cacheable(value = "storyboardItem", key = "'storyboard:' + #storyboardId", unless = "#result == null")
    public List<StoryboardItem> listItems(Long storyboardId) {
        return itemMapper.selectList(new LambdaQueryWrapper<StoryboardItem>()
                .eq(StoryboardItem::getStoryboardId, storyboardId)
                .orderByAsc(StoryboardItem::getSortOrder));
    }

    @Cacheable(value = "storyboardItem", key = "'scene:' + #sceneId", unless = "#result == null")
    public List<StoryboardItem> listItemsByScene(Long sceneId) {
        return itemMapper.selectList(new LambdaQueryWrapper<StoryboardItem>()
                .eq(StoryboardItem::getStoryboardSceneId, sceneId)
                .orderByAsc(StoryboardItem::getSortOrder));
    }
}
