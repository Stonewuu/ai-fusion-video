package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.service.ai.run.stream.AgentStreamHistoryRepository.History;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Objects;

/** 快照之后到达的同一完整消息也要去重，覆盖数据库提交早于 Redis 发布的时间窗口。 */
final class AgentStreamSnapshot {
    private final History history;
    private final long cursor;
    private final Set<String> messages = new HashSet<>();
    private final Set<String> calls = new HashSet<>();
    private final Set<String> results = new HashSet<>();
    private final Set<String> controls = new HashSet<>();

    AgentStreamSnapshot(History history, long cursor) {
        this.history = history;
        this.cursor = cursor;
        history.messages().forEach(message -> {
            messages.add(message.getProjectionKey());
            if ("tool".equals(message.getRole())) {
                String key = identity(message.getRunId(), message.getToolCallId());
                calls.add(key);
                if (!"running".equals(message.getToolStatus())) {
                    results.add(key);
                }
            }
        });
        history.controls().forEach(control -> controls.add(identity(control.run().getRunId(),
                control.event().envelope().rawEventId())));
    }

    boolean includes(AiChatStreamRespVO event) {
        if (controls.contains(identity(event.getSourceRunId(), event.getRawEventId()))) {
            return false;
        }
        return switch (event.getOutputType()) {
            case "CONTENT", "REASONING" -> !messages.contains(event.getMessageId());
            case "TOOL_CALL_STARTED", "TOOL_CALL" -> !calls.contains(identity(event.getSourceRunId(), event.getToolCallId()));
            case "TOOL_FINISHED", "SUB_AGENT_FINISHED" -> !results.contains(identity(event.getSourceRunId(), event.getToolCallId()));
            case "MODEL_RETRY" -> !messages.contains(AgentStreamMessageRepository.key(
                    event.getSourceRunId(), "assistant", event.getReplyId()));
            default -> true;
        };
    }

    AiChatStreamRespVO response(List<AiChatStreamRespVO> drafts, AgentStreamEventFormatter formatter) {
        AgentRun run = history.run();
        List<AiChatStreamRespVO> pending = new ArrayList<>(drafts);
        List<AiChatStreamRespVO> rootTerminal = new ArrayList<>();
        for (var control : history.controls()) {
            AgentRun source = control.run();
            var event = control.event();
            boolean terminal = Long.valueOf(event.sequence()).equals(source.getTerminalSequence());
            boolean waiting = ("WAITING_CONFIRMATION".equals(source.getStatus())
                    && "USER_CONFIRMATION_REQUIRED".equals(event.outputType())
                    && Objects.equals(event.envelope().replyId(), source.getWaitingReplyId()))
                    || ("WAITING_EXTERNAL".equals(source.getStatus())
                    && "EXTERNAL_EXECUTION_REQUIRED".equals(event.outputType())
                    && Objects.equals(event.envelope().toolCallId(), source.getWaitingToolCallId()));
            if (!event.publishRequired() || !(terminal || waiting)) {
                continue;
            }
            AiChatStreamRespVO response = AgentStreamReader.observed(run, formatter.format(source, event.envelope()));
            if (terminal) {
                response.setContent(null).setReasoningContent(null);
            }
            if (source.getRunId().equals(run.getRunId()) && terminal) {
                rootTerminal.add(response);
            } else {
                pending.add(response);
            }
        }
        pending.addAll(rootTerminal);
        return new AiChatStreamRespVO().setSchemaVersion(2).setRunId(run.getRunId())
                .setConversationId(run.getConversationId()).setSequence(cursor).setOutputType("HISTORY")
                .setHistoryMessages(history.messages()).setPendingEvents(pending)
                .setRunStatus(run.getStatus()).setFinished(terminal());
    }

    boolean terminal() {
        return history.run().getTerminalSequence() != null;
    }

    private static String identity(String runId, String value) {
        return runId + '\0' + value;
    }
}
