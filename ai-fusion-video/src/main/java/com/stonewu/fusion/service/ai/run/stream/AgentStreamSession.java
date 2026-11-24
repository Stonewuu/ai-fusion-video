package com.stonewu.fusion.service.ai.run.stream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.run.AgentScopeToolResultStatus;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** 同一运行的片段、完整消息和终态保存串行处理，取消时保存已经展示的内容。 */
final class AgentStreamSession {
    private final AgentRun run;
    private final String owner;
    private final long epoch;
    private final AgentRedisStreamStore streams;
    private final AgentStreamMessageRepository messages;
    private final AgentStreamEventFormatter formatter;
    private final AgentEventRepository controls;
    private final ObjectMapper json;
    private final Map<String, Tool> tools = new LinkedHashMap<>();
    private Reply reply;
    private boolean sealed;
    private boolean started;

    AgentStreamSession(AgentRun run, String owner, long epoch, AgentRedisStreamStore streams,
                       AgentStreamMessageRepository messages, AgentStreamEventFormatter formatter,
                       AgentEventRepository controls, ObjectMapper json) {
        this.run = run;
        this.owner = owner;
        this.epoch = epoch;
        this.streams = streams;
        this.messages = messages;
        this.formatter = formatter;
        this.controls = controls;
        this.json = json;
    }

    synchronized Optional<CommittedAgentEvent> accept(AgentEventEnvelope event) {
        if (sealed) {
            return Optional.empty();
        }
        String type = event.rawEventType();
        if ("AGENT_START".equals(type) && !started) {
            started = true;
            return appendControl(event);
        }
        if ("REQUIRE_USER_CONFIRM".equals(type) || "REQUIRE_EXTERNAL_EXECUTION".equals(type)) {
            saveReplyAndCalls(null);
            return appendControl(event);
        }
        if ("USER_CONFIRM_RESULT".equals(type)) {
            for (JsonNode decision : event.payload().get("confirmResults")) {
                if (!decision.get("confirmed").booleanValue()) {
                    String id = decision.get("toolCall").get("id").textValue();
                    tools.computeIfAbsent(id, Tool::new).userRejected = true;
                }
            }
            return Optional.empty();
        }
        if ("MODEL_RETRY".equals(event.outputType())) {
            if (reply != null && Objects.equals(reply.id, event.replyId())) {
                reply = null;
            }
            tools.values().removeIf(tool -> Objects.equals(tool.replyId, event.replyId()) && !tool.saved);
        } else if ("CONTENT".equals(event.outputType()) || "REASONING".equals(event.outputType())) {
            collectReply(event);
        } else if ("TOOL_CALL_START".equals(type) || "TOOL_CALL_DELTA".equals(type)
                || "TOOL_CALL_END".equals(type)) {
            collectCall(event);
        } else if ("TOOL_RESULT_START".equals(type) || "TOOL_RESULT_TEXT_DELTA".equals(type)
                || "TOOL_RESULT_DATA_DELTA".equals(type) || "TOOL_RESULT_END".equals(type)) {
            collectResult(event);
            if ("TOOL_RESULT_END".equals(type) && tool(event).userRejected) {
                // 拒绝已由审批记录保存和展示，不再把 SDK 的拒绝回显当成一次工具执行。
                return Optional.empty();
            }
        } else if ("MODEL_CALL_END".equals(type)) {
            saveReplyAndCalls(event.replyId());
        }
        if (event.outputType() != null) {
            AiChatStreamRespVO response = formatter.format(run, event);
            if ("CONTENT".equals(event.outputType()) || "REASONING".equals(event.outputType())) {
                response.setMessageId(reply.messageKey());
            }
            if ("TOOL_CALL_STARTED".equals(event.outputType()) || "TOOL_CALL".equals(event.outputType())) {
                Tool tool = tool(event);
                response.setToolName(tool.name).setToolCalls(List.of(new AiChatStreamRespVO.ToolCallVO()
                        .setId(tool.id).setName(tool.name)
                        .setArguments("TOOL_CALL_STARTED".equals(event.outputType()) ? "" : tool.arguments())));
            } else if ("TOOL_FINISHED".equals(event.outputType())) {
                Tool tool = tool(event);
                response.setToolName(tool.name).setToolResult(tool.result())
                        .setToolStatus(tool.status());
            }
            streams.append(run.getStreamRunId(), run.getRunId() + ':' + event.rawEventId(), response);
        }
        return Optional.empty();
    }

    synchronized void finish() {
        if (sealed) {
            return;
        }
        saveReplyAndCalls(null);
        sealed = true;
        for (Tool tool : tools.values()) {
            tool.callSaved.tryEmitEmpty();
            tool.resultSaved.tryEmitEmpty();
        }
    }

    synchronized Mono<Void> awaitCall(String id) {
        return tools.computeIfAbsent(id, Tool::new).callSaved.asMono();
    }

    synchronized Mono<Void> awaitResult(String id) {
        return tools.computeIfAbsent(id, Tool::new).resultSaved.asMono();
    }

    private Optional<CommittedAgentEvent> appendControl(AgentEventEnvelope event) {
        Optional<CommittedAgentEvent> committed = controls.appendOwnedTx(run.getRunId(), owner, epoch, event);
        if (committed.isEmpty()) {
            throw new IllegalStateException("当前执行者已不能保存运行控制记录");
        }
        return committed;
    }

    private void collectReply(AgentEventEnvelope event) {
        String id = require(event.replyId(), "回复标识");
        if (reply == null || !reply.id.equals(id)) {
            saveReplyAndCalls(null);
            reply = new Reply(id);
        }
        String value = AgentStreamEventFormatter.text(event.payload(), "delta", "content", "reasoningContent");
        if (value == null) {
            throw new IllegalArgumentException("流式消息缺少文本片段");
        }
        if ("CONTENT".equals(event.outputType())) {
            reply.content.append(value);
        } else {
            reply.reasoning.append(value);
        }
        JsonNode duration = event.payload().get("reasoningDurationMs");
        if (duration != null && duration.canConvertToLong()) {
            reply.reasoningDuration = duration.longValue();
        }
    }

    private void collectCall(AgentEventEnvelope event) {
        Tool tool = tool(event);
        tool.replyId = event.replyId();
        if ("TOOL_CALL_DELTA".equals(event.rawEventType())) {
            tool.arguments.append(delta(event.payload()));
        } else if ("TOOL_CALL_END".equals(event.rawEventType())) {
            tool.callComplete = true;
        }
    }

    private void collectResult(AgentEventEnvelope event) {
        Tool tool = tool(event);
        if ("TOOL_RESULT_TEXT_DELTA".equals(event.rawEventType())
                || "TOOL_RESULT_DATA_DELTA".equals(event.rawEventType())) {
            tool.output.append(delta(event.payload()));
        } else if ("TOOL_RESULT_END".equals(event.rawEventType())) {
            tool.state = AgentStreamEventFormatter.text(event.payload(), "state", "toolStatus", "status");
            if (event.outputType() != null && !tool.userRejected) {
                AgentMessage result = toolMessage(tool, "tool-result", tool.result(),
                        tool.status());
                messages.save(run.getRunId(), owner, epoch, List.of(result));
                if ("cancelled".equals(result.getToolStatus())) {
                    tool.state = "CANCELLED";
                }
            }
            tool.resultSaved.tryEmitEmpty();
        }
    }

    private void saveReplyAndCalls(String replyId) {
        List<AgentMessage> pending = new ArrayList<>();
        Reply completed = reply != null && (replyId == null || reply.id.equals(replyId)) ? reply : null;
        if (completed != null && (!completed.content.isEmpty() || !completed.reasoning.isEmpty())) {
            pending.add(AgentMessage.builder().role("assistant").projectionKey(completed.messageKey())
                    .content(completed.content.isEmpty() ? null : completed.content.toString())
                    .reasoningContent(completed.reasoning.isEmpty() ? null : completed.reasoning.toString())
                    .reasoningDurationMs(completed.reasoningDuration)
                    .parentToolCallId(run.getParentToolCallId()).build());
        }
        List<Tool> calls = tools.values().stream().filter(tool -> !tool.saved && tool.callComplete
                && (replyId == null || Objects.equals(replyId, tool.replyId))).toList();
        for (Tool tool : calls) {
            pending.add(toolMessage(tool, "tool-call", tool.arguments(), "running"));
        }
        messages.save(run.getRunId(), owner, epoch, pending);
        for (Tool tool : calls) {
            tool.saved = true;
            tool.callSaved.tryEmitEmpty();
        }
        if (completed != null) {
            if (!completed.content.isEmpty() || !completed.reasoning.isEmpty()) {
                boolean nested = !run.getRunId().equals(run.getStreamRunId());
                AiChatStreamRespVO saved = new AiChatStreamRespVO().setConversationId(run.getConversationId())
                        .setSourceRunId(run.getRunId()).setControlRunId(run.getRunId()).setCreatedAt(Instant.now())
                        .setMessageId(completed.messageKey()).setReplyId(completed.id)
                        .setOutputType("CONTENT").setContentMode("replace")
                        .setContent(completed.content.toString()).setReasoningContent(completed.reasoning.toString())
                        .setReasoningDurationMs(completed.reasoningDuration)
                        .setParentToolCallId(nested ? run.getParentToolCallId() : null)
                        .setAgentName(nested ? run.getAgentName() : null).setFinished(false);
                streams.append(run.getStreamRunId(), "saved:" + completed.messageKey(), saved);
            }
            reply = null;
        }
    }

    private AgentMessage toolMessage(Tool tool, String kind, String content, String status) {
        return AgentMessage.builder().role("tool")
                .projectionKey(AgentStreamMessageRepository.key(run.getRunId(), kind, tool.id))
                .content(content).toolName(require(tool.name, "工具名称")).toolStatus(status)
                .toolCallId(tool.id).parentToolCallId(run.getParentToolCallId()).build();
    }

    private Tool tool(AgentEventEnvelope event) {
        Tool tool = tools.computeIfAbsent(require(event.toolCallId(), "工具调用标识"), Tool::new);
        String name = AgentStreamEventFormatter.text(event.payload(), "toolCallName", "toolName", "name");
        if (name != null) {
            tool.name = name;
        }
        return tool;
    }

    private String delta(JsonNode payload) {
        JsonNode value = payload.get("delta");
        if (value == null || value.isNull()) {
            return "";
        }
        if (value.isTextual()) {
            return value.textValue();
        }
        try {
            return json.writeValueAsString(value);
        } catch (JsonProcessingException error) {
            throw new IllegalStateException("工具内容序列化失败", error);
        }
    }

    private static String require(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + "不能为空");
        }
        return value;
    }

    private final class Reply {
        private final String id;
        private final StringBuilder content = new StringBuilder();
        private final StringBuilder reasoning = new StringBuilder();
        private Long reasoningDuration;

        private Reply(String id) {
            this.id = id;
        }

        private String messageKey() {
            return AgentStreamMessageRepository.key(run.getRunId(), "assistant", id);
        }
    }

    private static final class Tool {
        private final String id;
        private String name;
        private String replyId;
        private String state;
        private boolean callComplete;
        private boolean saved;
        private boolean userRejected;
        private final StringBuilder arguments = new StringBuilder();
        private final StringBuilder output = new StringBuilder();
        private final Sinks.Empty<Void> callSaved = Sinks.empty();
        private final Sinks.Empty<Void> resultSaved = Sinks.empty();

        private Tool(String id) {
            this.id = id;
        }

        private String arguments() {
            return arguments.isEmpty() ? "{}" : arguments.toString();
        }

        private String result() {
            return output.isEmpty() ? null : output.toString();
        }

        private String status() {
            if ("CANCELLED".equals(state)) {
                return "cancelled";
            }
            return "DENIED".equalsIgnoreCase(state) ? "error" : AgentScopeToolResultStatus.project(state, result());
        }
    }
}
