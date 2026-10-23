package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.entity.media.MediaJobEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import java.util.List;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;

/**
 * 编辑入口以执行记录判断是否结束；撤权后的收尾不需要冒用旧身份修改分镜状态。
 */
@Component
@RequiredArgsConstructor
public class EpisodeComposeGuard {
    private final MediaJobMapper mediaJobs;

    public void requireIdle(long episodeId, ObjectNode episode) {
        if (mediaJobs.exists(Wrappers.<MediaJobEntity>lambdaQuery()
                .eq(MediaJobEntity::getKind, "EPISODE_COMPOSE").eq(MediaJobEntity::getTargetId, episodeId)
                .in(MediaJobEntity::getStatus,
                        List.of("QUEUED", "RUNNING", "WAITING", "NEEDS_REVIEW", "UNKNOWN_SUBMISSION")))) {
            throw active();
        }
        if (episode.path("composeStatus").asInt() != 1) {
            return;
        }
        var ended = mediaJobs.exists(
                Wrappers.<MediaJobEntity>lambdaQuery().eq(MediaJobEntity::getId, episode.path("composeJobId").asText())
                        .eq(MediaJobEntity::getKind, "EPISODE_COMPOSE").eq(MediaJobEntity::getTargetId, episodeId)
                        .in(MediaJobEntity::getStatus, List.of("SUCCEEDED", "FAILED", "CANCELLED", "CONFLICT")));
        if (!ended) {
            throw active();
        }
    }

    private static MediaException active() {
        return new MediaException(409, "MEDIA_JOB_ALREADY_RUNNING", "分集仍在合成或等待核实，请先停止或等待原合成结束");
    }
}
