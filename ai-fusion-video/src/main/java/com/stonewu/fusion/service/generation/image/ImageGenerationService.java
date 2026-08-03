package com.stonewu.fusion.service.generation.image;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.mapper.generation.ImageItemMapper;
import com.stonewu.fusion.mapper.generation.ImageTaskMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * 生图任务服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ImageGenerationService {

    private final ImageTaskMapper taskMapper;
    private final ImageItemMapper itemMapper;

    @Cacheable(value = "imageTask", key = "#id")
    public ImageTask getById(Long id) {
        ImageTask task = taskMapper.selectById(id);
        if (task == null) throw new BusinessException("生图任务不存在: " + id);
        return task;
    }

    @Cacheable(value = "imageTask", key = "'taskId:' + #taskId")
    public ImageTask getByTaskId(String taskId) {
        ImageTask task = taskMapper.selectOne(new LambdaQueryWrapper<ImageTask>().eq(ImageTask::getTaskId, taskId));
        if (task == null) throw new BusinessException("生图任务不存在: " + taskId);
        return task;
    }

    public PageResult<ImageTask> pageByUser(Long userId, int pageNo, int pageSize) {
        return PageResult.of(taskMapper.selectPage(new Page<>(pageNo, pageSize),
                new LambdaQueryWrapper<ImageTask>()
                        .eq(ImageTask::getUserId, userId)
                        .orderByDesc(ImageTask::getCreateTime)));
    }

    @CacheEvict(value = "imageTask", allEntries = true)
    @Transactional
    public ImageTask create(ImageTask task) {
        taskMapper.insert(task);
        return task;
    }

    @CacheEvict(value = "imageTask", allEntries = true)
    @Transactional
    public ImageTask update(ImageTask task) {
        taskMapper.updateById(task);
        return task;
    }

    @CacheEvict(value = "imageTask", allEntries = true)
    @Transactional
    public void updateStatus(Long id, Integer status, String errorMsg) {
        ImageTask task = getById(id);
        task.setStatus(status);
        task.setErrorMsg(errorMsg);
        taskMapper.updateById(task);
    }

    @CacheEvict(value = "imageTask", allEntries = true)
    @Transactional
    public void delete(Long id) {
        taskMapper.deleteById(id);
    }

    // ========== Image Items ==========

    @Cacheable(value = "imageItems", key = "#taskId")
    public List<ImageItem> listItems(Long taskId) {
        return itemMapper.selectList(new LambdaQueryWrapper<ImageItem>().eq(ImageItem::getTaskId, taskId));
    }

    @CacheEvict(value = "imageItems", allEntries = true)
    @Transactional
    public ImageItem createItem(ImageItem item) {
        itemMapper.insert(item);
        return item;
    }

    @CacheEvict(value = "imageItems", allEntries = true)
    @Transactional
    public ImageItem updateItem(ImageItem item) {
        itemMapper.updateById(item);
        return item;
    }

    public List<ImageTask> findPendingTasks() {
        return taskMapper.selectList(new LambdaQueryWrapper<ImageTask>().in(ImageTask::getStatus, 0, 1));
    }
}
