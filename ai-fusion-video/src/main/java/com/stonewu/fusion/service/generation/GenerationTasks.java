package com.stonewu.fusion.service.generation;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.mapper.generation.ImageItemMapper;
import com.stonewu.fusion.mapper.generation.ImageTaskMapper;
import com.stonewu.fusion.mapper.generation.VideoItemMapper;
import com.stonewu.fusion.mapper.generation.VideoTaskMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * 图片与视频任务共用的执行状态访问；业务权限由执行上下文校验。
 */
@Component
@RequiredArgsConstructor
public class GenerationTasks {
    private final ImageTaskMapper images;
    private final VideoTaskMapper videos;
    private final ImageItemMapper imageItems;
    private final VideoItemMapper videoItems;

    public Context context(GenerationKind kind, long id) {
        if (kind == GenerationKind.IMAGE) {
            ImageTask row = images.selectOne(new LambdaQueryWrapper<ImageTask>()
                    .select(ImageTask::getUserId, ImageTask::getProjectId, ImageTask::getStatus,
                            ImageTask::getExecutionContextJson, ImageTask::getOwnerType, ImageTask::getOwnerId)
                    .eq(ImageTask::getId, id));
            return row == null ? null : new Context(row.getUserId(), row.getProjectId(), row.getStatus(),
                    row.getExecutionContextJson(), row.getOwnerType(), row.getOwnerId());
        }
        VideoTask row = videos.selectOne(new LambdaQueryWrapper<VideoTask>()
                .select(VideoTask::getUserId, VideoTask::getProjectId, VideoTask::getStatus,
                        VideoTask::getExecutionContextJson, VideoTask::getOwnerType, VideoTask::getOwnerId)
                .eq(VideoTask::getId, id));
        return row == null ? null : new Context(row.getUserId(), row.getProjectId(), row.getStatus(),
                row.getExecutionContextJson(), row.getOwnerType(), row.getOwnerId());
    }

    public void lock(GenerationKind kind, long id) {
        if (kind == GenerationKind.IMAGE) {
            images.lock(id);
        } else {
            videos.lock(id);
        }
    }

    public void start(GenerationKind kind, long id) {
        if (kind == GenerationKind.IMAGE) {
            images.update(new LambdaUpdateWrapper<ImageTask>().eq(ImageTask::getId, id)
                    .eq(ImageTask::getStatus, 0).set(ImageTask::getStatus, 1));
        } else {
            videos.update(new LambdaUpdateWrapper<VideoTask>().eq(VideoTask::getId, id)
                    .eq(VideoTask::getStatus, 0).set(VideoTask::getStatus, 1));
        }
    }

    public long succeeded(GenerationKind kind, long taskId) {
        return kind == GenerationKind.IMAGE
                ? imageItems.selectCount(new LambdaQueryWrapper<ImageItem>()
                .eq(ImageItem::getTaskId, taskId).eq(ImageItem::getStatus, 1))
                : videoItems.selectCount(new LambdaQueryWrapper<VideoItem>()
                .eq(VideoItem::getTaskId, taskId).eq(VideoItem::getStatus, 1));
    }

    public void aggregate(GenerationKind kind, long id, int status, long succeeded, String error) {
        if (kind == GenerationKind.IMAGE) {
            images.update(new LambdaUpdateWrapper<ImageTask>().eq(ImageTask::getId, id)
                    .set(ImageTask::getStatus, status).set(ImageTask::getSuccessCount, succeeded)
                    .set(ImageTask::getErrorMsg, error));
        } else {
            videos.update(new LambdaUpdateWrapper<VideoTask>().eq(VideoTask::getId, id)
                    .set(VideoTask::getStatus, status).set(VideoTask::getSuccessCount, succeeded)
                    .set(VideoTask::getErrorMsg, error));
        }
    }

    public record Context(long userId, Long projectId, int status, String executionContextJson,
                          Integer ownerType, Long ownerId) {
    }
}
