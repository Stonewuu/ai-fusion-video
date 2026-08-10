package com.stonewu.fusion.service.ai.agentscope.kernel;

import io.agentscope.core.model.ChatModelBase;
import io.agentscope.harness.agent.HarnessAgent;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class AgentKernelResourceTests {

    @Test
    void closesHarnessToolsAndModelInOrderExactlyOnce() {
        HarnessAgent agent = mock(HarnessAgent.class);
        AgentKernelToolkitResources tools = mock(AgentKernelToolkitResources.class);
        OwnedChatModel model = mock(OwnedChatModel.class);
        AgentKernelResource resource = new AgentKernelResource(agent, model, tools);

        resource.close();
        resource.close();

        InOrder order = inOrder(agent, tools, model);
        order.verify(agent).close();
        order.verify(tools).close();
        order.verify(model).close();
        verify(agent, times(1)).close();
        verify(tools, times(1)).close();
        verify(model, times(1)).close();
    }

    @Test
    void preservesFirstCloseFailureAndSuppressesLaterFailures() {
        HarnessAgent agent = mock(HarnessAgent.class);
        AgentKernelToolkitResources tools = mock(AgentKernelToolkitResources.class);
        OwnedChatModel model = mock(OwnedChatModel.class);
        IllegalStateException harnessFailure = new IllegalStateException("harness");
        IllegalArgumentException toolsFailure = new IllegalArgumentException("tools");
        IllegalMonitorStateException modelFailure = new IllegalMonitorStateException("model");
        doThrow(harnessFailure).when(agent).close();
        doThrow(toolsFailure).when(tools).close();
        doThrow(modelFailure).when(model).close();

        assertThatThrownBy(() -> new AgentKernelResource(agent, model, tools).close())
                .isSameAs(harnessFailure)
                .hasSuppressedException(toolsFailure)
                .hasSuppressedException(modelFailure);
    }

    @Test
    void ownedModelClosesAutoCloseableDelegateOnlyOnce() {
        CloseableModel delegate = new CloseableModel();
        OwnedChatModel owned = OwnedChatModel.owned(delegate);

        owned.close();
        owned.close();

        org.assertj.core.api.Assertions.assertThat(delegate.closeCount).isOne();
    }

    private static final class CloseableModel extends ChatModelBase implements AutoCloseable {
        private int closeCount;

        @Override
        protected reactor.core.publisher.Flux<io.agentscope.core.model.ChatResponse> doStream(
                java.util.List<io.agentscope.core.message.Msg> messages,
                java.util.List<io.agentscope.core.model.ToolSchema> tools,
                io.agentscope.core.model.GenerateOptions options) {
            return reactor.core.publisher.Flux.empty();
        }

        @Override
        public String getModelName() {
            return "closeable-test";
        }

        @Override
        public void close() {
            closeCount++;
        }
    }
}
