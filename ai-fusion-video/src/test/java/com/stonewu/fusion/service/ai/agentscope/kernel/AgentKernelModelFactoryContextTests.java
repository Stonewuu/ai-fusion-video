package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeModelFactory;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.ai.provider.AiProviderService.AgentScopeModelProvision;
import io.agentscope.core.model.ChatModelBase;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.Map;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentKernelModelFactoryContextTests {

    @Test
    void existingAgentScopeFactoryIsTheOnlyKernelModelFactoryBean() {
        try (AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext()) {
            context.registerBean(AiProviderService.class, () -> mock(AiProviderService.class));
            context.register(AgentScopeModelFactory.class);
            context.refresh();

            Map<String, AgentKernelModelFactory> factories =
                    context.getBeansOfType(AgentKernelModelFactory.class);

            assertThat(factories).containsOnlyKeys("agentScopeModelFactory");
            assertThat(factories.get("agentScopeModelFactory"))
                    .isInstanceOf(AgentScopeModelFactory.class);
        }
    }

    @Test
    void createsANewOwnedModelForEveryKernelRequest() {
        AiProviderService provider = mock(AiProviderService.class);
        ChatModelBase firstModel = mock(ChatModelBase.class);
        ChatModelBase secondModel = mock(ChatModelBase.class);
        AiModel model = AiModel.builder().id(7L).code("model-a").build();
        String modelFingerprint = AgentKernelKey.modelRecordFingerprint(model);
        when(provider.provisionAgentScopeModel(org.mockito.ArgumentMatchers.any(AiModel.class)))
                .thenReturn(
                        new AgentScopeModelProvision(firstModel, modelFingerprint),
                        new AgentScopeModelProvision(secondModel, modelFingerprint));
        AgentScopeModelFactory factory = new AgentScopeModelFactory(provider);
        AgentKernelKey key = AgentKernelKey.create(
                "writer",
                modelFingerprint,
                AgentKernelKey.promptVersion("Write"),
                List.of(),
                "afv-tools-v1");
        AgentKernelSpec spec = new AgentKernelSpec(
                key, model, "writer", "Writer", "Writes", "Write", 10,
                List.of(), Set.of(), "afv-tools-v1");

        OwnedChatModel first = factory.create(spec);
        OwnedChatModel second = factory.create(spec);

        assertThat(first).isNotSameAs(second);
        assertThat(first.model()).isSameAs(firstModel);
        assertThat(second.model()).isSameAs(secondModel);
        verify(provider, times(2)).provisionAgentScopeModel(
                org.mockito.ArgumentMatchers.any(AiModel.class));
    }

    @Test
    void rejectsAndClosesAProvisionedModelWhenProviderConfigurationChanged() {
        AiProviderService provider = mock(AiProviderService.class);
        CloseableModel model = new CloseableModel();
        when(provider.provisionAgentScopeModel(org.mockito.ArgumentMatchers.any(AiModel.class)))
                .thenReturn(new AgentScopeModelProvision(model, "current-provider-fingerprint"));
        AgentScopeModelFactory factory = new AgentScopeModelFactory(provider);
        AiModel configuredModel = AiModel.builder().id(7L).code("model-a").build();
        AgentKernelKey key = AgentKernelKey.create(
                "writer", "stale-provider-fingerprint", AgentKernelKey.promptVersion("Write"),
                List.of(), "afv-tools-v1");
        AgentKernelSpec spec = new AgentKernelSpec(
                key, configuredModel, "writer", "Writer", "Writes", "Write", 10,
                List.of(), Set.of(), "afv-tools-v1");

        org.assertj.core.api.Assertions.assertThatThrownBy(() -> factory.create(spec))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("changed");
        assertThat(model.closeCount).isOne();
    }

    private static final class CloseableModel extends ChatModelBase implements AutoCloseable {
        private int closeCount;

        @Override
        protected reactor.core.publisher.Flux<io.agentscope.core.model.ChatResponse> doStream(
                List<io.agentscope.core.message.Msg> messages,
                List<io.agentscope.core.model.ToolSchema> tools,
                io.agentscope.core.model.GenerateOptions options) {
            return reactor.core.publisher.Flux.empty();
        }

        @Override
        public String getModelName() {
            return "closeable";
        }

        @Override
        public void close() {
            closeCount++;
        }
    }
}
