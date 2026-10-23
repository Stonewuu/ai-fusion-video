package com.stonewu.fusion.service.task;

import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.AiStreamRedisService;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.job.MediaJobNotifications;
import com.stonewu.fusion.service.media.job.MediaJobQueries;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.time.Duration;
import java.util.List;
import java.util.Set;

/**
 * 通用任务流服务。
 * <p>
 * 复用现有 conversation/message/Redis Stream 基础设施，
 * 让非 AI 的长任务也能拥有稳定的后端 taskId、SSE 重连与历史记录能力。
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class TaskStreamService {

    public static final String CATEGORY_TASK = "task";

    private final AgentConversationService conversationService;
    private final AgentMessageService messageService;
    private final AiStreamRedisService aiStreamRedisService;
    private final MediaJobQueries mediaJobs;
    private final CanvasAccessPolicy access;

    public String createTask(Long userId,
                             Long projectId,
                             String taskType,
                             String title,
                             String contextType,
                             Long contextId,
                             String initialContent) {
        String taskId = IdUtil.fastSimpleUUID();
        conversationService.createOrUpdate(taskId, userId, projectId,
                contextType, contextId, taskType, title, CATEGORY_TASK);
        aiStreamRedisService.cleanup(taskId);
        aiStreamRedisService.markActive(taskId);
        if (StrUtil.isNotBlank(initialContent)) {
            publishContent(taskId, initialContent);
        }
        return taskId;
    }

    public void publishContent(String taskId, String content) {
        if (StrUtil.isBlank(content)) {
            return;
        }
        messageService.saveAssistantMessage(taskId, content, null, null);
        aiStreamRedisService.publish(taskId, new AiChatStreamRespVO()
                .setConversationId(taskId)
                .setOutputType("CONTENT")
                .setContent(content)
                .setFinished(false));
    }

    public void complete(String taskId, String content) {
        if (StrUtil.isNotBlank(content)) {
            messageService.saveAssistantMessage(taskId, content, null, null);
        }
        aiStreamRedisService.publish(taskId, new AiChatStreamRespVO()
                .setConversationId(taskId)
                .setOutputType("DONE")
                .setContent(content)
                .setFinished(true));
        aiStreamRedisService.markCompleted(taskId);
        aiStreamRedisService.scheduleCleanup(taskId);
        conversationService.finish(taskId, "completed");
        log.info("[TaskStream] 任务完成: taskId={}", taskId);
    }

    public void fail(String taskId, String errorMessage) {
        String safeMessage = StrUtil.blankToDefault(errorMessage, "任务失败");
        String content = safeMessage;
        messageService.saveAssistantMessage(taskId, content, null, null);
        aiStreamRedisService.publish(taskId, new AiChatStreamRespVO()
                .setConversationId(taskId)
                .setOutputType("CONTENT")
                .setContent(content)
                .setFinished(false));
        aiStreamRedisService.publish(taskId, new AiChatStreamRespVO()
                .setConversationId(taskId)
                .setOutputType("ERROR")
                .setError(safeMessage)
                .setFinished(true));
        aiStreamRedisService.markError(taskId);
        aiStreamRedisService.scheduleCleanup(taskId);
        conversationService.finish(taskId, "failed");
        log.warn("[TaskStream] 任务失败: taskId={}, message={}", taskId, safeMessage);
    }

    public Flux<AiChatStreamRespVO> reconnect(String taskId, CanvasActor actor) {
        if (mediaJobs.exists(taskId)) {
            mediaJobs.get(taskId, actor);
            return Flux.interval(Duration.ZERO, Duration.ofSeconds(1))
                    .concatMap(tick -> Mono.fromCallable(() -> mediaJobs.get(taskId, actor))
                            .subscribeOn(Schedulers.boundedElastic()))
                    .distinctUntilChanged(job -> job.status() + ":" + job.phase() + ":" + job.error())
                    .map(job -> {
                        boolean done = !Set.of("QUEUED", "RUNNING", "WAITING").contains(job.status());
                        var event = new AiChatStreamRespVO().setConversationId(taskId).setFinished(done);
                        if (!done) {
                            return event.setOutputType("CONTENT").setContent(job.phase());
                        }
                        if (job.status().equals("SUCCEEDED")) {
                            return event.setOutputType("DONE").setContent(MediaJobNotifications.summary(job));
                        }
                        if (job.status().equals("CANCELLED")) {
                            return event.setOutputType("CANCELLED").setContent(MediaJobNotifications.summary(job));
                        }
                        return event.setOutputType("ERROR").setError(MediaJobNotifications.summary(job));
                    }).takeUntil(event -> Boolean.TRUE.equals(event.getFinished()));
        }
        requireLegacyRead(taskId, actor);
        String status = aiStreamRedisService.getStatus(taskId);
        if ("NONE".equals(status)) {
            return Flux.empty();
        }
        return aiStreamRedisService.subscribe(taskId).doOnNext(event -> requireLegacyRead(taskId, actor));
    }

    public String getStatus(String taskId, CanvasActor actor) {
        if (mediaJobs.exists(taskId)) {
            String status = mediaJobs.get(taskId, actor).status();
            if (status.equals("SUCCEEDED")) {
                return "COMPLETED";
            }
            if (status.equals("CANCELLED")) {
                return "CANCELLED";
            }
            return Set.of("QUEUED", "RUNNING", "WAITING").contains(status) ? "ACTIVE" : "ERROR";
        }
        requireLegacyRead(taskId, actor);
        return aiStreamRedisService.getStatus(taskId);
    }

    public List<AgentConversation> listRunning(CanvasActor actor) {
        access.requireActorScope(actor);
        return conversationService.listRunning(actor.userId()).stream().filter(conversation -> {
            try {
                if (conversation.getProjectId() != null) {
                    access.requireProject(conversation.getProjectId(), actor);
                }
                return true;
            } catch (CanvasException denied) {
                if (denied.getCode() == 403 || denied.getCode() == 404) {
                    return false;
                }
                throw denied;
            }
        }).toList();
    }

    private void requireLegacyRead(String taskId, CanvasActor actor) {
        access.requireActorScope(actor);
        var conversation = conversationService.getOwnedByConversationId(taskId, actor.userId());
        if (conversation == null || !CATEGORY_TASK.equals(conversation.getCategory())) {
            throw new BusinessException(404, "任务流不存在");
        }
        if (conversation.getProjectId() != null) {
            access.requireProject(conversation.getProjectId(), actor);
        }
    }
}
