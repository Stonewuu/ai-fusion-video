package com.stonewu.fusion.service.storyboard;

import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.media.EpisodeAudioJobMapper;
import com.stonewu.fusion.entity.media.EpisodeAudioJobEntity;
import com.stonewu.fusion.mapper.storyboard.StoryboardItemMapper;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.config.EpisodeBgmProperties;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.job.ComposeSnapshot;
import com.stonewu.fusion.service.media.job.EpisodeComposeGuard;
import com.stonewu.fusion.service.media.job.MediaJobNotifications;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceMutations;
import com.stonewu.fusion.service.resource.model.ComposeWritebackCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.SHOT;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.STORYBOARD_EPISODE;

/**
 * 提交只保存任务和输入快照。领取、恢复、FFmpeg 处理由可恢复媒体任务执行器负责。
 */
@Service
@RequiredArgsConstructor
public class VideoComposeService {
    private final UserMapper users;
    private final MediaJobMapper mediaJobs;
    private final EpisodeAudioJobMapper audioJobs;
    private final StoryboardItemMapper items;
    private final ResourceReadRepository resources;
    private final StoryboardMutationRepository storyboards;
    private final CanvasAccessPolicy access;
    private final ResourceAgentFence agents;
    private final ResourceCommandService commands;
    private final MediaJobRepository jobs;
    private final MediaJobNotifications notifications;
    private final CanvasJson json;
    private final EpisodeBgmService bgm;
    private final EpisodeBgmProperties properties;
    private final EpisodeComposeGuard activeJobs;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"storyboard", "storyboardEpisode", "resourceDocument"}, allEntries = true)
    public String submitCompose(long episodeId, int expectedVersion, String requestId, ResourceCommandContext context,
                                boolean includeBgm) {
        if (requestId == null || !requestId.matches("[A-Za-z0-9._:-]{1,128}") || expectedVersion < 0) {
            throw MediaException.invalid("合成请求标识或版本无效");
        }
        var episode = resources.require(STORYBOARD_EPISODE, episodeId, false);
        agents.requireActive(context, episode.projectId());
        users.lockEnabled(context.actor().userId());
        var changes = storyboards.lock(STORYBOARD_EPISODE, episodeId);
        access.requireProjectEditor(episode.projectId(), context.actor());
        String hash = json.hash(
                Map.of("episodeId", episodeId, "version", expectedVersion, "context", context, "includeBgm",
                        includeBgm));
        var previous = mediaJobs.lockRequest(context.actor().userId(), requestId);
        if (previous != null) {
            if (!hash.equals(previous.getRequestHash())) {
                throw new MediaException(409, "MEDIA_JOB_REQUEST_CONFLICT", "同一请求 ID 不能用于不同的合成内容");
            }
            return previous.getId();
        }
        ResourceMutations.requireVersion(changes.target, expectedVersion);
        activeJobs.requireIdle(episodeId, changes.data(changes.target));
        List<MediaSourceReference> clips = clips(episodeId);
        if (clips.isEmpty()) {
            throw MediaException.invalid("本集没有镜头，请先添加分镜并生成视频");
        }
        ComposeSnapshot.Bgm settings = null;
        var config = includeBgm ? bgm.resolveApiConfig() : null;
        if (config != null) {
            if (!Double.isFinite(
                    properties.getMusicVolume()) || properties.getMusicVolume() < 0 || properties.getMusicVolume() > 1) {
                throw MediaException.invalid("配乐音量必须在 0 至 1 之间");
            }
            settings = new ComposeSnapshot.Bgm(config.getId(), json.hash(config), properties.getMusicPrompt(),
                    properties.getMusicVolume(),
                    properties.isSfxEnabled(), properties.getTimeoutSeconds(), properties.getPollIntervalSeconds());
        }
        String id = UUID.randomUUID().toString();
        mediaJobs.insertCompose(id, episode.projectId(), episodeId, expectedVersion + 1, context.actor().userId(),
                requestId, hash, json.write(context), json.write(new ComposeSnapshot(clips, settings)));
        if (settings != null) {
            audioJobs.insert(EpisodeAudioJobEntity.builder().jobId(id).apiConfigId(settings.apiConfigId())
                    .apiConfigHash(settings.configHash()).sfxEnabled(settings.sfx()).build());
        }
        commands.execute(new ComposeWritebackCommand("compose:" + id + ":start", episodeId, id,
                ComposeWritebackCommand.Stage.START, null), context);
        notifications.started(jobs.get(id),
                resources.data(resources.require(STORYBOARD_EPISODE, episodeId, false)).path("title").asText("分镜集"));
        return id;
    }

    private List<MediaSourceReference> clips(long episodeId) {
        List<Long> ids = items.composeClipIds(episodeId);
        if (ids.size() > 500) {
            throw MediaException.invalid("单次分集合成最多支持 500 个镜头");
        }
        List<MediaSourceReference> result = new ArrayList<>();
        for (long id : ids) {
            var item = resources.require(SHOT, id, true);
            ObjectNode data = resources.data(item);
            String field = data.path("videoUrl").asText("").isBlank() ? "generatedVideoUrl" : "videoUrl";
            if (data.path(field).asText("").isBlank()) {
                throw MediaException.invalid(
                        "镜头 " + data.path("shotNumber").asText(Long.toString(id)) + " 缺少视频，请补齐后合成");
            }
            result.add(new MediaSourceReference(SHOT, id, item.version(), field));
        }
        return List.copyOf(result);
    }
}
