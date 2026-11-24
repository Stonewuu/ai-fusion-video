package com.stonewu.fusion.service.ai.run.stream;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import org.springframework.stereotype.Component;

import java.io.IOException;

/** 新流的工具事件已经包含完整参数和结果，不读取数据库增量。 */
@Component
public class AgentStreamEventFormatter {
    private final ObjectReader responseReader;

    public AgentStreamEventFormatter(ObjectMapper json) {
        responseReader = json.readerFor(AiChatStreamRespVO.class)
                .without(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
    }

    public AiChatStreamRespVO format(AgentRun run, AgentEventEnvelope event) {
        AiChatStreamRespVO response;
        try {
            response = responseReader.readValue(event.payload());
        } catch (IOException error) {
            throw new IllegalStateException("消息事件内容无效", error);
        }
        boolean nested = !run.getRunId().equals(run.getStreamRunId());
        response.setSchemaVersion(2).setRunId(run.getStreamRunId())
                .setSourceRunId(run.getRunId()).setControlRunId(run.getRunId())
                .setConversationId(run.getConversationId()).setOutputType(event.outputType())
                .setMessageId(event.rawEventId()).setSource(event.source()).setReplyId(event.replyId())
                .setBlockId(event.blockId()).setToolCallId(event.toolCallId())
                .setParentToolCallId(nested ? run.getParentToolCallId() : event.parentToolCallId())
                .setAgentName(nested ? run.getAgentName() : event.agentName())
                .setRawEventId(event.rawEventId()).setRawEventType(event.rawEventType())
                .setCreatedAt(event.createdAt());
        if ("CONTENT".equals(event.outputType())) {
            response.setContent(text(event.payload(), "content", "delta"));
        } else if ("REASONING".equals(event.outputType())) {
            response.setReasoningContent(text(event.payload(), "reasoningContent", "delta"));
        }
        if ("DONE".equals(event.outputType()) || "ERROR".equals(event.outputType())
                || "CANCELLED".equals(event.outputType())) {
            response.setFinished(true);
            if (!nested) {
                response.setParentToolCallId(null).setAgentName(null);
            }
        } else {
            response.setFinished(false);
        }
        return response;
    }

    public static String text(JsonNode payload, String... names) {
        for (String name : names) {
            JsonNode value = payload.get(name);
            if (value != null && value.isTextual()) {
                return value.textValue();
            }
        }
        return null;
    }
}
