package com.stonewu.fusion.service.generation.image;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.mapper.generation.ImageItemMapper;
import com.stonewu.fusion.mapper.generation.ImageTaskMapper;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.job.GenerationDeletionGuard;
import com.stonewu.fusion.service.media.GenerationMediaLinks;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.Caching;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.annotation.Isolation;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 生图任务服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ImageGenerationService {

    private final ImageTaskMapper taskMapper;
    private final ImageItemMapper itemMapper;
    private final GenerationDeletionGuard deletionGuard;
    private final GenerationMediaLinks mediaLinks;

    public ImageTask getById(Long id) {
        ImageTask task = taskMapper.selectById(id);
        if (task == null) {
            throw new BusinessException(404, "生图任务不存在: " + id);
        }
        return task;
    }

    public ImageTask getByTaskId(String taskId) {
        ImageTask task = taskMapper.selectOne(new LambdaQueryWrapper<ImageTask>().eq(ImageTask::getTaskId, taskId));
        if (task == null) {
            throw new BusinessException(404, "生图任务不存在: " + taskId);
        }
        return task;
    }

    public PageResult<ImageTask> pageByUser(CanvasActor actor, List<Long> projects, int pageNo, int pageSize) {
        var query = new LambdaQueryWrapper<ImageTask>().eq(ImageTask::getUserId, actor.userId())
                .and(scope -> {
                    scope.and(personal -> personal.isNull(ImageTask::getProjectId).and(owner -> {
                        owner.and(
                                user -> user.eq(ImageTask::getOwnerType, 1).eq(ImageTask::getOwnerId, actor.userId()));
                        if (actor.teamId() != null) {
                            owner.or(
                                    team -> team.eq(ImageTask::getOwnerType, 2)
                                            .eq(ImageTask::getOwnerId, actor.teamId()));
                        }
                    }));
                    if (!projects.isEmpty()) {
                        scope.or().in(ImageTask::getProjectId, projects);
                    }
                }).orderByDesc(ImageTask::getCreateTime);
        return PageResult.of(taskMapper.selectPage(new Page<>(pageNo, pageSize), query));
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
        int changed = taskMapper.update(null, new UpdateWrapper<ImageTask>().eq("id", task.getId())
                .in("status", 0, 1).set("success_count", task.getSuccessCount()));
        if (changed != 1) {
            throw new BusinessException(409, "生成任务已停止，不能更新结果计数");
        }
        return task;
    }

    @CacheEvict(value = "imageTask", allEntries = true)
    @Transactional
    public void updateStatus(Long id, Integer status, String errorMsg) {
        if (status == null || status < 1 || status > 3) {
            throw new BusinessException(400, "生成任务状态无效");
        }
        var update = new UpdateWrapper<ImageTask>().eq("id", id).set("status", status).set("error_msg", errorMsg);
        if (status == 1) {
            update.eq("status", 0);
        } else if (status == 2) {
            update.eq("status", 1);
        } else {
            update.in("status", 0, 1);
        }
        int changed = taskMapper.update(null, update);
        if (changed != 1 && status != 3) {
            throw new BusinessException(409, "生成任务已删除、停止或已被其他工作者领取");
        }
    }

    @Caching(evict = {
            @CacheEvict(value = "imageTask", allEntries = true),
            @CacheEvict(value = "imageItems", allEntries = true)
    })
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void delete(Long id, CanvasActor actor) {
        deletionGuard.lockScope(GenerationKind.IMAGE, id, actor);
        ImageTask task = taskMapper.selectForUpdate(id);
        if (task == null) {
            throw new BusinessException(404, "图片任务不存在");
        }
        if (task.getStatus() == null || !List.of(2, 3, 4, 6).contains(task.getStatus())) {
            throw new BusinessException(409, "请等待图片生成结束后再删除记录");
        }
        deletionGuard.requireTerminal(GenerationKind.IMAGE, id);
        // 删除关联的图片条目
        itemMapper.delete(new LambdaQueryWrapper<ImageItem>().eq(ImageItem::getTaskId, id));
        taskMapper.deleteById(id);
    }

    // ========== Image Items ==========

    @Cacheable(value = "imageItems", key = "#taskId", condition = "@imageGenerationService.cacheableResults(#taskId)")
    public List<ImageItem> listItems(Long taskId) {
        return itemMapper.selectList(
                new LambdaQueryWrapper<ImageItem>().eq(ImageItem::getTaskId, taskId).orderByAsc(ImageItem::getId));
    }

    @CacheEvict(value = "imageItems", allEntries = true)
    @Transactional
    public ImageItem createItem(ImageItem item) {
        item.setVersion(0);
        itemMapper.insert(item);
        return item;
    }

    @CacheEvict(value = "imageItems", allEntries = true)
    @Transactional
    public ImageItem updateItem(ImageItem item) {
        if (item.getVersion() == null || item.getVersion() < 0) {
            throw new BusinessException(400, "生成条目缺少读取版本");
        }
        Map<String, String> sources = new HashMap<>();
        sources.put("imageUrl", item.getImageUrl());
        sources.put("thumbnailUrl", item.getThumbnailUrl());
        if (!Integer.valueOf(2).equals(item.getStatus())) {
            mediaLinks.beforeWrite(GenerationKind.IMAGE, item.getTaskId(), sources);
        }
        var parent = taskMapper.selectForUpdate(item.getTaskId());
        if (parent == null) {
            throw new BusinessException(404, "生成任务不存在");
        }
        if (Integer.valueOf(3).equals(parent.getStatus()) && Integer.valueOf(2).equals(item.getStatus())) {
            int cancelled = itemMapper.update(null,
                    new LambdaUpdateWrapper<ImageItem>().eq(ImageItem::getId, item.getId())
                            .eq(ImageItem::getVersion, item.getVersion())
                            .eq(ImageItem::getStatus, 0).set(ImageItem::getStatus, 2)
                            .set(ImageItem::getErrorMsg, item.getErrorMsg())
                            .setIncrBy(ImageItem::getVersion, 1));
            if (cancelled != 1) {
                throw new BusinessException(409, "生成条目已完成或发生变化");
            }
            item.setVersion(item.getVersion() + 1);
            return item;
        }
        if (!Integer.valueOf(1).equals(parent.getStatus())) {
            throw new BusinessException(409, "生成任务已停止，不能修改输出");
        }
        int changed = itemMapper.update(item, new LambdaUpdateWrapper<ImageItem>().eq(ImageItem::getId, item.getId())
                .eq(ImageItem::getVersion, item.getVersion()).setIncrBy(ImageItem::getVersion, 1));
        if (changed != 1) {
            throw new BusinessException(409, "生成条目已变化或被删除");
        }
        item.setVersion(item.getVersion() + 1);
        mediaLinks.capture(GenerationKind.IMAGE, item.getTaskId(), item.getId(), item.getVersion(), sources);
        return item;
    }

    public boolean cacheableResults(Long taskId) {
        var task = taskMapper.selectById(taskId);
        return task != null && Integer.valueOf(2).equals(task.getStatus());
    }

    public List<ImageTask> findPendingTasks() {
        return taskMapper.selectList(new LambdaQueryWrapper<ImageTask>().in(ImageTask::getStatus, 0, 1));
    }
}
