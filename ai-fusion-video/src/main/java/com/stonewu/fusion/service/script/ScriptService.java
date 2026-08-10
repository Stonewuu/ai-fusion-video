package com.stonewu.fusion.service.script;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.bean.copier.CopyOptions;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.mapper.script.ScriptSceneItemMapper;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * 剧本服务（含分集、分场次管理）
 */
@Service
@RequiredArgsConstructor
public class ScriptService {

    /** BeanUtil 更新时需要排除的基础字段（不应由前端覆盖） */
    private static final String[] IGNORE_FIELDS = {
            "id", "projectId", "title", "scope", "ownerType", "ownerId",
            "createTime", "updateTime", "deleted"
    };

    private final ScriptMapper scriptMapper;
    private final ScriptEpisodeMapper episodeMapper;
    private final ScriptSceneItemMapper sceneItemMapper;

    // ========== 剧本 ==========

    @Cacheable(value = "script", key = "#id")
    public Script getById(Long id) {
        Script script = scriptMapper.selectById(id);
        if (script == null)
            throw new BusinessException("剧本不存在: " + id);
        return script;
    }

    @Cacheable(value = "script", key = "'project:' + #projectId", unless = "#result == null")
    public Script getByProjectId(Long projectId) {
        return scriptMapper.selectOne(new LambdaQueryWrapper<Script>()
                .eq(Script::getProjectId, projectId)
                .orderByDesc(Script::getCreateTime)
                .orderByDesc(Script::getId)
                .last("LIMIT 1"));
    }

    @CacheEvict(value = "script", allEntries = true)
    @Transactional
    public Script update(Script script) {
        Script existing = getById(script.getId());
        BeanUtil.copyProperties(script, existing,
                CopyOptions.create().ignoreNullValue().setIgnoreProperties(IGNORE_FIELDS));
        int rows = scriptMapper.updateById(existing);
        if (rows == 0) {
            throw new BusinessException("更新失败，数据已被其他操作修改，请刷新后重试");
        }
        return existing;
    }

    @CacheEvict(value = "script", allEntries = true)
    @Transactional
    public void updateParsingStatus(Long scriptId, Integer status, String progress) {
        Script script = getById(scriptId);
        script.setParsingStatus(status);
        script.setParsingProgress(progress);
        scriptMapper.updateById(script);
    }

    @CacheEvict(value = { "script", "episode", "scene" }, allEntries = true)
    @Transactional
    public Script replaceSourceAndReset(Long scriptId, String rawContent) {
        getById(scriptId);
        sceneItemMapper.delete(new LambdaQueryWrapper<ScriptSceneItem>()
                .eq(ScriptSceneItem::getScriptId, scriptId));
        episodeMapper.delete(new LambdaQueryWrapper<ScriptEpisode>()
                .eq(ScriptEpisode::getScriptId, scriptId));

        UpdateWrapper<Script> update = new UpdateWrapper<Script>()
                .eq("id", scriptId)
                .set("raw_content", rawContent)
                .set("content", null)
                .set("total_episodes", 0)
                .set("story_synopsis", null)
                .set("characters_json", null)
                .set("parsing_status", 0)
                .set("parsing_progress", null)
                .set("summary", null)
                .set("genre", null)
                .set("target_audience", null)
                .set("duration_estimate", null)
                .setSql("version = version + 1");
        scriptMapper.update(null, update);
        return getById(scriptId);
    }

    // ========== 分集 ==========

    @Cacheable(value = "episode", key = "#id")
    public ScriptEpisode getEpisodeById(Long id) {
        ScriptEpisode ep = episodeMapper.selectById(id);
        if (ep == null)
            throw new BusinessException("分集不存在: " + id);
        return ep;
    }

    @Cacheable(value = "episode", key = "'script:' + #scriptId")
    public List<ScriptEpisode> listEpisodes(Long scriptId) {
        return episodeMapper.selectList(new LambdaQueryWrapper<ScriptEpisode>()
                .eq(ScriptEpisode::getScriptId, scriptId)
                .orderByAsc(ScriptEpisode::getSortOrder)
                .orderByAsc(ScriptEpisode::getEpisodeNumber));
    }

    @CacheEvict(value = "episode", allEntries = true)
    @Transactional
    public ScriptEpisode createEpisode(ScriptEpisode episode) {
        episodeMapper.insert(episode);
        return episode;
    }

    @CacheEvict(value = "episode", allEntries = true)
    @Transactional
    public ScriptEpisode updateEpisode(ScriptEpisode episode) {
        ScriptEpisode existing = getEpisodeById(episode.getId());
        BeanUtil.copyProperties(episode, existing,
                CopyOptions.create().ignoreNullValue().setIgnoreProperties(IGNORE_FIELDS));
        int rows = episodeMapper.updateById(existing);
        if (rows == 0) {
            throw new BusinessException("更新失败，数据已被其他操作修改，请刷新后重试");
        }
        return existing;
    }

    @CacheEvict(value = { "episode", "scene" }, allEntries = true)
    @Transactional
    public void deleteEpisode(Long id) {
        getEpisodeById(id);
        sceneItemMapper.delete(new LambdaQueryWrapper<ScriptSceneItem>()
                .eq(ScriptSceneItem::getEpisodeId, id));
        episodeMapper.deleteById(id);
    }

    // ========== 分场次 ==========

    @Cacheable(value = "scene", key = "#id")
    public ScriptSceneItem getSceneById(Long id) {
        ScriptSceneItem scene = sceneItemMapper.selectById(id);
        if (scene == null)
            throw new BusinessException("场次不存在: " + id);
        return scene;
    }

    @Cacheable(value = "scene", key = "'episode:' + #episodeId")
    public List<ScriptSceneItem> listScenesByEpisode(Long episodeId) {
        return sceneItemMapper.selectList(new LambdaQueryWrapper<ScriptSceneItem>()
                .eq(ScriptSceneItem::getEpisodeId, episodeId)
                .orderByAsc(ScriptSceneItem::getSortOrder));
    }

    public List<ScriptSceneItem> listScenesByScript(Long scriptId) {
        return sceneItemMapper.selectList(new LambdaQueryWrapper<ScriptSceneItem>()
                .eq(ScriptSceneItem::getScriptId, scriptId)
                .orderByAsc(ScriptSceneItem::getSortOrder));
    }

    @CacheEvict(value = "scene", allEntries = true)
    @Transactional
    public ScriptSceneItem createScene(ScriptSceneItem scene) {
        sceneItemMapper.insert(scene);
        return scene;
    }

    @CacheEvict(value = "scene", allEntries = true)
    @Transactional
    public ScriptSceneItem updateScene(ScriptSceneItem scene) {
        // 读取数据库中的完整记录（含正确的 version，乐观锁需要）
        ScriptSceneItem existing = getSceneById(scene.getId());
        BeanUtil.copyProperties(scene, existing,
                CopyOptions.create().ignoreNullValue().setIgnoreProperties(IGNORE_FIELDS));
        int rows = sceneItemMapper.updateById(existing);
        if (rows == 0) {
            throw new BusinessException("更新失败，数据已被其他操作修改，请刷新后重试");
        }
        return existing;
    }

    @CacheEvict(value = "scene", allEntries = true)
    @Transactional
    public void deleteScene(Long id) {
        sceneItemMapper.deleteById(id);
    }

    // ========== AI 工具支撑方法 ==========

    @CacheEvict(value = "episode", allEntries = true)
    @Transactional
    public ScriptEpisode saveEpisode(Long scriptId, Integer episodeNumber, String title,
            String synopsis, String rawContent, Integer sourceType) {
        return saveEpisode(scriptId, episodeNumber, title, synopsis, rawContent, sourceType, null);
    }

    @CacheEvict(value = "episode", allEntries = true)
    @Transactional
    public ScriptEpisode saveEpisode(Long scriptId, Integer episodeNumber, String title,
            String synopsis, String rawContent, Integer sourceType, Integer sortOrder) {
        ScriptEpisode episode = episodeMapper.selectOne(new LambdaQueryWrapper<ScriptEpisode>()
                .eq(ScriptEpisode::getScriptId, scriptId)
                .eq(ScriptEpisode::getEpisodeNumber, episodeNumber));

        if (episode != null) {
            if (title != null)
                episode.setTitle(title);
            if (synopsis != null)
                episode.setSynopsis(synopsis);
            if (rawContent != null)
                episode.setRawContent(rawContent);
            if (sourceType != null)
                episode.setSourceType(sourceType);
            if (sortOrder != null)
                episode.setSortOrder(sortOrder);
            episodeMapper.updateById(episode);
        } else {
            episode = ScriptEpisode.builder()
                    .scriptId(scriptId)
                    .episodeNumber(episodeNumber)
                    .title(title)
                    .synopsis(synopsis)
                    .rawContent(rawContent)
                    .sourceType(sourceType != null ? sourceType : 0)
                    .sortOrder(sortOrder != null ? sortOrder : episodeNumber)
                    .build();
            episodeMapper.insert(episode);
        }
        return episode;
    }

    @CacheEvict(value = { "scene", "episode" }, allEntries = true)
    @Transactional
    public void batchSaveSceneItems(Long episodeId, Integer episodeVersion, List<ScriptSceneItem> sceneItems) {
        batchSaveSceneItems(episodeId, episodeVersion, sceneItems, false);
    }

    @CacheEvict(value = { "scene", "episode" }, allEntries = true)
    @Transactional
    public void batchSaveSceneItems(Long episodeId, Integer episodeVersion, List<ScriptSceneItem> sceneItems,
            boolean overwriteMode) {
        ScriptEpisode episode = getEpisodeById(episodeId);

        // 乐观锁校验仅在覆盖模式下执行，避免多次追加调用之间发生版本冲突。
        if (overwriteMode && episodeVersion != null && !episodeVersion.equals(episode.getVersion())) {
            throw new BusinessException(String.format(
                    "版本冲突：期望版本 %d，实际版本 %d。请重新获取最新版本后再试。",
                    episodeVersion, episode.getVersion()));
        }

        int startIndex = 0;
        if (!overwriteMode) {
            // 追加模式：查询已有场次数量，从末尾继续编号
            Long existingCount = sceneItemMapper.selectCount(
                    new LambdaQueryWrapper<ScriptSceneItem>().eq(ScriptSceneItem::getEpisodeId, episodeId));
            startIndex = existingCount.intValue();
        } else {
            // 覆盖模式：删除旧场次
            sceneItemMapper
                    .delete(new LambdaQueryWrapper<ScriptSceneItem>().eq(ScriptSceneItem::getEpisodeId, episodeId));
        }

        // 写入新场次
        for (int i = 0; i < sceneItems.size(); i++) {
            ScriptSceneItem item = sceneItems.get(i);
            item.setId(null);
            item.setEpisodeId(episodeId);
            item.setScriptId(episode.getScriptId());
            item.setSortOrder(startIndex + i);
            item.setSceneNumber(String.format("%d-%d", episode.getEpisodeNumber(), startIndex + i + 1));
            sceneItemMapper.insert(item);
        }

        // 更新集的场次计数
        if (!overwriteMode) {
            // 追加模式：重新查询实际场次总数
            Long totalCount = sceneItemMapper.selectCount(
                    new LambdaQueryWrapper<ScriptSceneItem>().eq(ScriptSceneItem::getEpisodeId, episodeId));
            episode.setTotalScenes(totalCount.intValue());
        } else {
            episode.setTotalScenes(sceneItems.size());
        }
        episodeMapper.updateById(episode);
    }
}
