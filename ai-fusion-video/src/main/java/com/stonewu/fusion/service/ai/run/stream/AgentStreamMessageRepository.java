package com.stonewu.fusion.service.ai.run.stream;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;

/** 按完整消息分配顺序；Redis 操作在调用此组件的事务之外执行。 */
@Component
@RequiredArgsConstructor
public class AgentStreamMessageRepository {
    private final AgentRunMapper runs;
    private final AgentConversationMapper conversations;
    private final AgentMessageMapper messages;

    @Transactional
    public void save(String runId, String owner, long epoch, List<AgentMessage> candidates) {
        if (candidates.isEmpty()) {
            return;
        }
        AgentRun run = runs.selectByRunIdForUpdate(runId);
        LocalDateTime now = runs.selectDatabaseNow();
        // 取消确认会结束租约；原执行者仍须在取消终态提交前保存已经展示的内容。
        if (run == null || !Objects.equals(run.getOwnerInstanceId(), owner) || run.getOwnerEpoch() != epoch
                || ("RUNNING".equals(run.getStatus())
                && (run.getLeaseUntil() == null || !run.getLeaseUntil().isAfter(now)))
                || !("RUNNING".equals(run.getStatus()) || "CANCEL_REQUESTED".equals(run.getStatus()))) {
            throw new IllegalStateException("当前执行者已不能保存此运行的消息");
        }
        AgentConversation conversation = conversations.selectProjectionConversationForUpdate(run.getConversationId());
        if (conversation == null || Boolean.TRUE.equals(conversation.getDeleted())) {
            throw new IllegalStateException("消息所属会话已不可用");
        }
        List<AgentMessage> added = new ArrayList<>();
        long next = conversation.getNextMessageOrder();
        for (AgentMessage message : candidates) {
            message.setRunId(runId);
            message.setConversationId(run.getConversationId());
            // SDK 把子任务取消作为工具错误返回；展示和历史采用子运行已提交的取消状态。
            if ("tool".equals(message.getRole()) && "error".equals(message.getToolStatus())) {
                AgentRun cancelledChild = runs.selectOne(new LambdaQueryWrapper<AgentRun>()
                        .select(AgentRun::getRunId).eq(AgentRun::getParentRunId, runId)
                        .eq(AgentRun::getParentToolCallId, message.getToolCallId())
                        .eq(AgentRun::getStatus, "CANCELLED"));
                if (cancelledChild != null) {
                    message.setToolStatus("cancelled");
                }
            }
            AgentMessage existing = messages.selectByProjectionKey(message.getProjectionKey());
            if (existing != null) {
                requireSame(existing, message);
                continue;
            }
            message.setMessageOrder(next++);
            message.setCreateTime(now);
            message.setUpdateTime(now);
            added.add(message);
        }
        if (added.isEmpty()) {
            return;
        }
        if (messages.insertMessages(added) != added.size()) {
            throw new IllegalStateException("完整消息未全部保存");
        }
        if (conversations.update(null, new LambdaUpdateWrapper<AgentConversation>()
                .eq(AgentConversation::getId, conversation.getId())
                .set(AgentConversation::getNextMessageOrder, next)
                .set(AgentConversation::getMessageCount, conversation.getMessageCount() + added.size())
                .set(AgentConversation::getLastMessageTime, now)
                .set(AgentConversation::getUpdateTime, now)) != 1) {
            throw new IllegalStateException("会话消息计数未更新");
        }
    }

    public boolean hasToolCall(String runId, String toolCallId) {
        return messages.selectByProjectionKey(key(runId, "tool-call", toolCallId)) != null;
    }

    public static String key(String runId, String kind, String identity) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest((runId + '\0' + kind + '\0' + identity).getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("消息标识计算不可用", error);
        }
    }

    private void requireSame(AgentMessage saved, AgentMessage candidate) {
        if (!Objects.equals(saved.getRunId(), candidate.getRunId())
                || !Objects.equals(saved.getRole(), candidate.getRole())
                || !Objects.equals(saved.getContent(), candidate.getContent())
                || !Objects.equals(saved.getReasoningContent(), candidate.getReasoningContent())
                || !Objects.equals(saved.getToolCallId(), candidate.getToolCallId())
                || !Objects.equals(saved.getParentToolCallId(), candidate.getParentToolCallId())
                || !Objects.equals(saved.getToolName(), candidate.getToolName())
                || !Objects.equals(saved.getToolStatus(), candidate.getToolStatus())) {
            throw new IllegalStateException("同一消息标识对应的内容不一致");
        }
    }
}
