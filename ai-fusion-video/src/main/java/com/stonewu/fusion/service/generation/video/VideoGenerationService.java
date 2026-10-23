package com.stonewu.fusion.service.generation.video;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.mapper.generation.VideoItemMapper;
import com.stonewu.fusion.mapper.generation.VideoTaskMapper;
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
 * 生视频任务服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class VideoGenerationService {

    private final VideoTaskMapper taskMapper;
    private final VideoItemMapper itemMapper;
    private final GenerationDeletionGuard deletionGuard;
    private final GenerationMediaLinks mediaLinks;

    public VideoTask getById(Long id) {
        VideoTask task = taskMapper.selectById(id);
        if (task == null) {
            throw new BusinessException(404, "生视频任务不存在: " + id);
        }
        return task;
    }

    public VideoTask getByTaskId(String taskId) {
        VideoTask task = taskMapper.selectOne(new LambdaQueryWrapper<VideoTask>().eq(VideoTask::getTaskId, taskId));
        if (task == null) {
            throw new BusinessException(404, "生视频任务不存在: " + taskId);
        }
        return task;
    }

    public PageResult<VideoTask> pageByUser(CanvasActor actor, List<Long> projects, int pageNo, int pageSize) {
        var query = new LambdaQueryWrapper<VideoTask>().eq(VideoTask::getUserId, actor.userId())
                .and(scope -> {
                    scope.and(personal -> personal.isNull(VideoTask::getProjectId).and(owner -> {
                        owner.and(
                                user -> user.eq(VideoTask::getOwnerType, 1).eq(VideoTask::getOwnerId, actor.userId()));
                        if (actor.teamId() != null) {
                            owner.or(
                                    team -> team.eq(VideoTask::getOwnerType, 2)
                                            .eq(VideoTask::getOwnerId, actor.teamId()));
                        }
                    }));
                    if (!projects.isEmpty()) {
                        scope.or().in(VideoTask::getProjectId, projects);
                    }
                }).orderByDesc(VideoTask::getCreateTime);
        return PageResult.of(taskMapper.selectPage(new Page<>(pageNo, pageSize), query));
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
        int changed = taskMapper.update(null, new UpdateWrapper<VideoTask>().eq("id", task.getId())
                .in("status", 0, 1).set("success_count", task.getSuccessCount()));
        if (changed != 1) {
            throw new BusinessException(409, "生成任务已停止，不能更新结果计数");
        }
        return task;
    }

    @CacheEvict(value = "videoTask", allEntries = true)
    @Transactional
    public void updateStatus(Long id, Integer status, String errorMsg) {
        if (status == null || status < 1 || status > 3) {
            throw new BusinessException(400, "生成任务状态无效");
        }
        var update = new UpdateWrapper<VideoTask>().eq("id", id).set("status", status).set("error_msg", errorMsg);
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
            @CacheEvict(value = "videoTask", allEntries = true),
            @CacheEvict(value = "videoItems", allEntries = true)
    })
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void delete(Long id, CanvasActor actor) {
        deletionGuard.lockScope(GenerationKind.VIDEO, id, actor);
        VideoTask task = taskMapper.selectForUpdate(id);
        if (task == null) {
            throw new BusinessException(404, "视频任务不存在");
        }
        if (task.getStatus() == null || !List.of(2, 3, 4, 6).contains(task.getStatus())) {
            throw new BusinessException(409, "请等待视频生成结束后再删除记录");
        }
        deletionGuard.requireTerminal(GenerationKind.VIDEO, id);
        // 删除关联的视频条目
        itemMapper.delete(new LambdaQueryWrapper<VideoItem>().eq(VideoItem::getTaskId, id));
        taskMapper.deleteById(id);
    }

    // ========== Video Items ==========

    @Cacheable(value = "videoItems", key = "#taskId", condition = "@videoGenerationService.cacheableResults(#taskId)")
    public List<VideoItem> listItems(Long taskId) {
        return itemMapper.selectList(
                new LambdaQueryWrapper<VideoItem>().eq(VideoItem::getTaskId, taskId).orderByAsc(VideoItem::getId));
    }

    @CacheEvict(value = "videoItems", allEntries = true)
    @Transactional
    public VideoItem createItem(VideoItem item) {
        item.setVersion(0);
        itemMapper.insert(item);
        return item;
    }

    @CacheEvict(value = "videoItems", allEntries = true)
    @Transactional
    public VideoItem updateItem(VideoItem item) {
        if (item.getVersion() == null || item.getVersion() < 0) {
            throw new BusinessException(400, "生成条目缺少读取版本");
        }
        Map<String, String> sources = new HashMap<>();
        sources.put("videoUrl", item.getVideoUrl());
        sources.put("coverUrl", item.getCoverUrl());
        sources.put("firstFrameUrl", item.getFirstFrameUrl());
        sources.put("lastFrameUrl", item.getLastFrameUrl());
        if (!Integer.valueOf(2).equals(item.getStatus())) {
            mediaLinks.beforeWrite(GenerationKind.VIDEO, item.getTaskId(), sources);
        }
        var parent = taskMapper.selectForUpdate(item.getTaskId());
        if (parent == null) {
            throw new BusinessException(404, "生成任务不存在");
        }
        if (Integer.valueOf(3).equals(parent.getStatus()) && Integer.valueOf(2).equals(item.getStatus())) {
            int cancelled = itemMapper.update(null,
                    new LambdaUpdateWrapper<VideoItem>().eq(VideoItem::getId, item.getId())
                            .eq(VideoItem::getVersion, item.getVersion())
                            .eq(VideoItem::getStatus, 0).set(VideoItem::getStatus, 2)
                            .set(VideoItem::getErrorMsg, item.getErrorMsg())
                            .setIncrBy(VideoItem::getVersion, 1));
            if (cancelled != 1) {
                throw new BusinessException(409, "生成条目已完成或发生变化");
            }
            item.setVersion(item.getVersion() + 1);
            return item;
        }
        if (!Integer.valueOf(1).equals(parent.getStatus())) {
            throw new BusinessException(409, "生成任务已停止，不能修改输出");
        }
        int changed = itemMapper.update(item, new LambdaUpdateWrapper<VideoItem>().eq(VideoItem::getId, item.getId())
                .eq(VideoItem::getVersion, item.getVersion()).setIncrBy(VideoItem::getVersion, 1));
        if (changed != 1) {
            throw new BusinessException(409, "生成条目已变化或被删除");
        }
        item.setVersion(item.getVersion() + 1);
        mediaLinks.capture(GenerationKind.VIDEO, item.getTaskId(), item.getId(), item.getVersion(), sources);
        return item;
    }

    public boolean cacheableResults(Long taskId) {
        var task = taskMapper.selectById(taskId);
        return task != null && Integer.valueOf(2).equals(task.getStatus());
    }

    public List<VideoTask> findPendingTasks() {
        return taskMapper.selectList(new LambdaQueryWrapper<VideoTask>().in(VideoTask::getStatus, 0, 1));
    }
}
