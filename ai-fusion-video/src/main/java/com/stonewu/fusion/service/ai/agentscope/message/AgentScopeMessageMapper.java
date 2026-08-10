package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.controller.ai.vo.AiMultimodalInputVO;
import com.stonewu.fusion.entity.ai.AgentMessage;
import io.agentscope.core.message.AudioBlock;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.DataBlock;
import io.agentscope.core.message.ImageBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.Source;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.URLSource;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.message.VideoBlock;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
public final class AgentScopeMessageMapper {

    public UserMessage toUserMessage(String text) {
        return toUserMessage(text, List.of());
    }

    public List<Msg> toUserMessages(String text) {
        return List.of(toUserMessage(text));
    }

    public UserMessage toUserMessage(String text, List<AiMultimodalInputVO> inputs) {
        List<ContentBlock> blocks = new ArrayList<>();
        if (text != null && !text.isBlank()) {
            blocks.add(TextBlock.builder().text(text).build());
        }
        if (inputs != null) {
            inputs.stream().map(this::toContentBlock).forEach(blocks::add);
        }
        if (blocks.isEmpty()) {
            throw new IllegalArgumentException("user message must contain text or multimodal input");
        }
        return new UserMessage(blocks);
    }

    public List<Msg> toUserMessages(String text, List<AiMultimodalInputVO> inputs) {
        return List.of(toUserMessage(text, inputs));
    }

    /**
     * Rebuilds a provider-safe conversation prefix at a terminal-run
     * continuation boundary. Tool calls are summarized as assistant context so
     * an interrupted call never becomes an invalid pending provider tool call.
     */
    public List<Msg> toRecoveredContinuationMessages(
            List<AgentMessage> persistedMessages, String continuationText) {
        List<Msg> recovered = new ArrayList<>();
        Map<String, AgentMessage> pendingToolCalls = new LinkedHashMap<>();
        for (AgentMessage message : List.copyOf(persistedMessages)) {
            if (message.getParentToolCallId() != null) {
                continue;
            }
            if ("user".equals(message.getRole())) {
                flushPendingToolCalls(recovered, pendingToolCalls);
                addTextMessage(recovered, MsgRole.USER, message.getContent());
                continue;
            }
            if ("assistant".equals(message.getRole())) {
                flushPendingToolCalls(recovered, pendingToolCalls);
                addTextMessage(recovered, MsgRole.ASSISTANT, assistantText(message));
                continue;
            }
            if (!"tool".equals(message.getRole()) || message.getToolCallId() == null) {
                continue;
            }
            if ("running".equals(message.getToolStatus())) {
                pendingToolCalls.put(message.getToolCallId(), message);
                continue;
            }
            AgentMessage call = pendingToolCalls.remove(message.getToolCallId());
            addTextMessage(
                    recovered,
                    MsgRole.ASSISTANT,
                    toolSummary(call, message, false));
        }
        flushPendingToolCalls(recovered, pendingToolCalls);
        recovered.add(toUserMessage(continuationText));
        return List.copyOf(recovered);
    }

    private void flushPendingToolCalls(
            List<Msg> recovered, Map<String, AgentMessage> pendingToolCalls) {
        pendingToolCalls.values().forEach(call -> addTextMessage(
                recovered,
                MsgRole.ASSISTANT,
                toolSummary(call, null, true)));
        pendingToolCalls.clear();
    }

    private String assistantText(AgentMessage message) {
        String content = normalize(message.getContent());
        String reasoning = normalize(message.getReasoningContent());
        if (content == null) {
            return reasoning;
        }
        if (reasoning == null) {
            return content;
        }
        return "此前分析：\n" + reasoning + "\n\n此前输出：\n" + content;
    }

    private String toolSummary(
            AgentMessage call, AgentMessage result, boolean interrupted) {
        AgentMessage identity = call == null ? result : call;
        StringBuilder summary = new StringBuilder(interrupted
                ? "此前工具调用未完成："
                : "此前工具调用记录：");
        summary.append("\n名称：").append(identity.getToolName());
        if (call != null && normalize(call.getContent()) != null) {
            summary.append("\n参数：").append(call.getContent());
        }
        if (result != null) {
            summary.append("\n状态：").append(result.getToolStatus());
            if (normalize(result.getContent()) != null) {
                summary.append("\n结果：").append(result.getContent());
            }
        }
        return summary.toString();
    }

    private void addTextMessage(List<Msg> messages, MsgRole role, String text) {
        String normalized = normalize(text);
        if (normalized == null) {
            return;
        }
        messages.add(Msg.builder()
                .role(role)
                .content(TextBlock.builder().text(normalized).build())
                .build());
    }

    private String normalize(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }

    private ContentBlock toContentBlock(AiMultimodalInputVO input) {
        Source source = "url".equals(input.getTransport())
                ? URLSource.builder().url(input.getUrl()).mimeType(input.getMimeType()).build()
                : Base64Source.builder().mediaType(input.getMimeType()).data(input.getData()).build();
        return switch (input.getInputType()) {
            case "image" -> ImageBlock.builder().source(source).build();
            case "video" -> VideoBlock.builder().source(source).build();
            case "audio" -> AudioBlock.builder().source(source).build();
            case "file" -> DataBlock.builder()
                    .id(input.getId())
                    .name(input.getName())
                    .source(source)
                    .build();
            case null, default -> throw new IllegalArgumentException(
                    "Unsupported multimodal input type: " + input.getInputType());
        };
    }
}
