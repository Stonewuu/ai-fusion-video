package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.controller.ai.vo.AiMultimodalInputVO;
import io.agentscope.core.message.AudioBlock;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.DataBlock;
import io.agentscope.core.message.ImageBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.Source;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.URLSource;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.message.VideoBlock;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

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
