package com.stonewu.fusion.service.ai.text;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.ImageBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.URLSource;
import io.agentscope.core.middleware.ReasoningInput;
import io.agentscope.core.state.AgentState;
import io.agentscope.harness.agent.filesystem.remote.store.StoreItem;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class LongToolResultTests {
    private final ObjectMapper json = new ObjectMapper();
    private final AgentWorkspaceBaseStore workspace = mock(AgentWorkspaceBaseStore.class);
    private final Map<List<String>, StoreItem> entries = new ConcurrentHashMap<>();
    private final LongToolResultStore results = new LongToolResultStore(workspace);

    LongToolResultTests() {
        when(workspace.get(any(), anyString())).thenAnswer(call -> entries.get(key(call.getArgument(0), call.getArgument(1))));
        when(workspace.putArtifactIfAbsent(any(), anyString(), any())).thenAnswer(call -> entries.putIfAbsent(
                key(call.getArgument(0), call.getArgument(1)),
                new StoreItem(call.getArgument(1), call.getArgument(2), 1)) == null);
    }

    @Test
    @DisplayName("超过原截断阈值的工具输出保存全文，并同时替换会话状态与模型输入")
    void preservesAll95910CharactersAndResultStateAndAttachments() throws Exception {
        String content = "首".repeat(2000) + "中".repeat(91910) + "尾".repeat(2000);
        var image = ImageBlock.builder().source(URLSource.builder().url("https://example.test/image.png").build()).build();
        var result = new ToolResultBlock("call", "external_tool", List.of(
                TextBlock.builder().text(content).build(), image), Map.of("trace", "kept"), ToolResultState.ERROR);
        Msg message = Msg.builder().role(MsgRole.TOOL).content(result).build();
        AgentState state = AgentState.builder().sessionId("session").userId("7").build();
        state.contextMutable().add(message);
        Agent agent = mock(Agent.class);
        when(agent.getAgentState()).thenReturn(state);
        var observed = new AtomicReference<ReasoningInput>();
        new LongToolResultMiddleware(results).onReasoning(agent,
                RuntimeContext.builder().userId("7").sessionId("session").build(),
                new ReasoningInput(List.of(message), List.of(), null), input -> {
                    observed.set(input);
                    return Flux.empty();
                }).blockLast();
        ToolResultBlock projected = (ToolResultBlock) observed.get().messages().getFirst().getContent().getFirst();
        assertThat(projected.getState()).isEqualTo(ToolResultState.ERROR);
        assertThat(projected.getMetadata()).containsEntry("trace", "kept");
        assertThat(projected.getOutput().get(1)).isSameAs(image);
        String text = ((TextBlock) projected.getOutput().getFirst()).getText();
        assertThat(text).doesNotContain("read_file").hasSizeLessThan(1000);
        var reference = json.readTree(text);
        assertThat(reference.path("readTool").asText()).isEqualTo("read_tool_result");
        assertThat(reference.path("totalChars").asInt()).isEqualTo(95910);
        String id = reference.path("resultId").asText();
        StringBuilder full = new StringBuilder();
        int offset = 0;
        do {
            var page = results.read("7", "session", id, new TextPages.Request(offset, 8000, id));
            full.append(page.content());
            offset = page.nextOffset() == null ? -1 : page.nextOffset();
        } while (offset >= 0);
        assertThat(full.toString()).isEqualTo(content);
        assertThat(((TextBlock) ((ToolResultBlock) state.contextMutable().getFirst().getContent().getFirst())
                .getOutput().getFirst()).getText()).isEqualTo(text);
        verify(workspace, times(1)).putArtifactIfAbsent(any(), anyString(), any());
    }

    @Test
    @DisplayName("结果编号不能跨用户或会话读取，也不能作为文件路径解释")
    void isolatesUsersAndSessionsAndRejectsPaths() {
        String id = results.save("7", "session", "私有结果");
        var first = new TextPages.Request(0, 100, null);
        assertThatThrownBy(() -> results.read("8", "session", id, first)).hasMessageContaining("没有此工具结果");
        assertThatThrownBy(() -> results.read("7", "other-session", id, first)).hasMessageContaining("没有此工具结果");
        assertThatThrownBy(() -> results.read("7", "session", "../../config", first)).hasMessageContaining("resultId");
        assertThat(results.read("7", "session", id, first).content()).isEqualTo("私有结果");
    }

    @Test
    @DisplayName("保存失败必须停止处理并保留原状态，不能返回无法读取的占位内容")
    void storageFailureDoesNotDiscardCanonicalContentOrCallTheModel() {
        String content = "文".repeat(95910);
        Msg message = Msg.builder().role(MsgRole.TOOL).content(ToolResultBlock.of("call", "large_query",
                TextBlock.builder().text(content).build())).build();
        AgentState state = AgentState.builder().build();
        state.contextMutable().add(message);
        Agent agent = mock(Agent.class);
        when(agent.getAgentState()).thenReturn(state);
        doThrow(new IllegalStateException("存储不可用")).when(workspace).putArtifactIfAbsent(any(), anyString(), any());
        assertThatThrownBy(() -> new LongToolResultMiddleware(results).onReasoning(agent,
                RuntimeContext.builder().userId("7").sessionId("session").build(),
                new ReasoningInput(List.of(message), List.of(), null), input -> {
                    throw new AssertionError("保存失败时不能调用模型");
                }).blockLast()).hasMessageContaining("存储不可用");
        assertThat(state.contextMutable()).containsExactly(message);
    }

    private List<String> key(List<String> namespace, String id) {
        var values = new ArrayList<>(namespace);
        values.add(id);
        return List.copyOf(values);
    }
}
