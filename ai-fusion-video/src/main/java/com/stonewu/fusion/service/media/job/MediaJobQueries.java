package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.entity.media.MediaJobEntity;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Component
@RequiredArgsConstructor
public class MediaJobQueries {
    private final MediaJobRepository jobs;
    private final MediaJobMapper mediaJobs;
    private final ProjectMapper projects;
    private final CanvasAccessPolicy access;

    public boolean exists(String id) {
        return mediaJobs.exists(Wrappers.<MediaJobEntity>lambdaQuery().eq(MediaJobEntity::getId, id));
    }

    public MediaJob.View get(String id, CanvasActor actor) {
        var job = jobs.get(id);
        requireEpisodeJob(job);
        access.requireProject(job.projectId(), actor);
        return job.view();
    }

    public List<MediaJob.View> list(long projectId, long episodeId, CanvasActor actor) {
        access.requireProject(projectId, actor);
        return mediaJobs.selectPage(new Page<MediaJobEntity>(1, 50, false), Wrappers.<MediaJobEntity>lambdaQuery()
                        .select(MediaJobEntity::getId).eq(MediaJobEntity::getProjectId, projectId)
                        .eq(MediaJobEntity::getTargetId, episodeId)
                        .orderByDesc(MediaJobEntity::getCreatedAt, MediaJobEntity::getId)).getRecords().stream()
                .map(row -> get(row.getId(), actor)).toList();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public boolean cancel(String id, CanvasActor actor) {
        var job = jobs.get(id);
        requireEpisodeJob(job);
        projects.lock(job.projectId());
        access.requireProjectEditor(job.projectId(), actor);
        return mediaJobs.requestComposeCancel(id) == 1;
    }

    private void requireEpisodeJob(MediaJob job) {
        if (!job.kind().equals("EPISODE_COMPOSE") || job.projectId() == null) {
            throw new MediaException(403, "MEDIA_JOB_ACCESS_DENIED", "请从任务所属的工作区读取和操作此任务");
        }
    }
}
