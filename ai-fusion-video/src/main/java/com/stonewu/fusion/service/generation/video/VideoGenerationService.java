package com.stonewu.fusion.service.generation.video;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.mapper.generation.VideoItemMapper;
import com.stonewu.fusion.mapper.generation.VideoTaskMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * 生视频任务服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class VideoGenerationService {

    private final VideoTaskMapper taskMapper;
    private final VideoItemMapper itemMapper;

    @Cacheable(value = "videoTask", key = "#id")
    public VideoTask getById(Long id) {
        VideoTask task = taskMapper.selectById(id);
        if (task == null) throw new BusinessException("生视频任务不存在: " + id);
        return task;
    }

    @Cacheable(value = "videoTask", key = "'taskId:' + #taskId")
    public VideoTask getByTaskId(String taskId) {
        VideoTask task = taskMapper.selectOne(new LambdaQueryWrapper<VideoTask>().eq(VideoTask::getTaskId, taskId));
        if (task == null) throw new BusinessException("生视频任务不存在: " + taskId);
        return task;
    }

    public PageResult<VideoTask> pageByUser(Long userId, int pageNo, int pageSize) {
        return PageResult.of(taskMapper.selectPage(new Page<>(pageNo, pageSize),
                new LambdaQueryWrapper<VideoTask>()
                        .eq(VideoTask::getUserId, userId)
                        .orderByDesc(VideoTask::getCreateTime)));
    }

    @CacheEvict(value = "videoTask", allEntries = true)
    @Transactional
    public VideoTask create(VideoTask task) {
        taskMapper.insert(task);
        return task;
    }

    @CacheEvict(value = "videoTask", allEntries = true)
    @Transactional
    public VideoTask update(VideoTask task) {
        taskMapper.updateById(task);
        return task;
    }

    @CacheEvict(value = "videoTask", allEntries = true)
    @Transactional
    public void updateStatus(Long id, Integer status, String errorMsg) {
        VideoTask task = getById(id);
        task.setStatus(status);
        task.setErrorMsg(errorMsg);
        taskMapper.updateById(task);
    }

    @CacheEvict(value = "videoTask", allEntries = true)
    @Transactional
    public void delete(Long id) {
        taskMapper.deleteById(id);
    }

    // ========== Video Items ==========

    @Cacheable(value = "videoItems", key = "#taskId")
    public List<VideoItem> listItems(Long taskId) {
        return itemMapper.selectList(new LambdaQueryWrapper<VideoItem>().eq(VideoItem::getTaskId, taskId));
    }

    @CacheEvict(value = "videoItems", allEntries = true)
    @Transactional
    public VideoItem createItem(VideoItem item) {
        itemMapper.insert(item);
        return item;
    }

    @CacheEvict(value = "videoItems", allEntries = true)
    @Transactional
    public VideoItem updateItem(VideoItem item) {
        itemMapper.updateById(item);
        return item;
    }

    public List<VideoTask> findPendingTasks() {
        return taskMapper.selectList(new LambdaQueryWrapper<VideoTask>().in(VideoTask::getStatus, 0, 1));
    }
}
