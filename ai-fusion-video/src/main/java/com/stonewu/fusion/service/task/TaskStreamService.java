package com.stonewu.fusion.service.task;

import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.AiStreamRedisService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.util.List;

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

    public Flux<AiChatStreamRespVO> reconnect(String taskId) {
        String status = aiStreamRedisService.getStatus(taskId);
        if ("NONE".equals(status)) {
            return Flux.empty();
        }
        return aiStreamRedisService.subscribe(taskId);
    }

    public String getStatus(String taskId) {
        return aiStreamRedisService.getStatus(taskId);
    }

    public List<AgentConversation> listRunning(Long userId) {
        return conversationService.listRunning(userId);
    }
}