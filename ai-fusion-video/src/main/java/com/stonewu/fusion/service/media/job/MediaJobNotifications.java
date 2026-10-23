package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

/**
 * 任务状态和通知历史在同一数据库事务提交；实时流直接读取任务版本，不依赖 Redis 存活。
 */
@Component
@RequiredArgsConstructor
public class MediaJobNotifications {
    private final AgentConversationService conversations;
    private final AgentMessageService messages;

    @Transactional(propagation = Propagation.MANDATORY)
    public void started(MediaJob job, String title) {
        conversations.createOrUpdate(job.id(), job.context().actor().userId(), job.projectId(), "storyboard_episode",
                job.targetId(), "video_compose", "合成本集视频：" + title, "task");
        messages.saveAssistantMessage(job.id(), "合成任务已提交，等待处理", null, null);
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void progress(MediaJob job) {
        messages.saveAssistantMessage(job.id(), job.phase(), null, null);
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void finished(MediaJob job) {
        messages.saveAssistantMessage(job.id(), summary(job.view()), null, null);
        conversations.finish(job.id(), switch (job.status()) {
            case "SUCCEEDED" -> "completed";
            case "CANCELLED" -> "cancelled";
            default -> "failed";
        });
    }

    public static String summary(MediaJob.View job) {
        if (job.status().equals("SUCCEEDED")) {
            return "合成完成 · 视频地址：" + job.results().path("video").asText();
        }
        return job.error() == null ? job.phase() : job.error();
    }
}
