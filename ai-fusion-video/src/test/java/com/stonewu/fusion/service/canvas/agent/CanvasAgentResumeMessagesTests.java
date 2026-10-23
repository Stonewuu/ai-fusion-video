package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutorRegistry;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.extensions.model.openai.formatter.OpenAIMessageConverter;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

class CanvasAgentResumeMessagesTests {
    private final ObjectMapper mapper = new ObjectMapper();
    private final ToolExecutorRegistry tools = mock(ToolExecutorRegistry.class);

    @Test
    void everyParallelChildResultSurvivesTheActualOpenAiProtocolConverter() {
        var saved = mapper.createObjectNode();
        var results = saved.putArray("results");
        for (int i = 0; i < 3; i++) results.addObject().put("toolCallId", "child-" + i)
                .put("toolName", "generate_storyboard_frame").putObject("result").put("status", i == 2 ? "error" : "success").put("summary", "result-" + i);
        var messages = CanvasAgentResumeMessages.external(saved, tools);
        var converter = new OpenAIMessageConverter(message -> "", blocks -> blocks.toString());

        assertThat(messages).hasSize(3).allSatisfy(message -> assertThat(message.getContent()).hasSize(1));
        assertThat(messages.stream().map(message -> converter.convertToMessage(message, false).getToolCallId()))
                .containsExactly("child-0", "child-1", "child-2");
        assertThat(messages.getLast().getFirstContentBlock(ToolResultBlock.class).getState()).isEqualTo(ToolResultState.ERROR);
    }

    @Test
    void duplicateResultsCannotResumeTheSameToolTwice() {
        var saved = mapper.createObjectNode();
        var results = saved.putArray("results");
        for (int i = 0; i < 2; i++) results.addObject().put("toolCallId", "same")
                .put("toolName", "generate_storyboard_frame").putObject("result").put("status", "success");
        assertThatThrownBy(() -> CanvasAgentResumeMessages.external(saved, tools)).isInstanceOf(IllegalStateException.class);
    }
}
