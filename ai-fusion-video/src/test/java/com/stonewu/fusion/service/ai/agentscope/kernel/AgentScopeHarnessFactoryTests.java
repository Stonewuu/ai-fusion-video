package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailureGuard;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import io.agentscope.core.tool.Toolkit;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class AgentScopeHarnessFactoryTests {

    @Test
    void keyAndCollectionContractsAreStableAndImmutable() {
        AgentKernelToolManifest alpha = manifest("alpha", "schema-a");
        AgentKernelToolManifest beta = manifest("beta", "schema-b");
        String whitelistVersion = AgentKernelKey.whitelistVersion("tools-v1", Set.of("beta", "alpha"));
        AgentKernelKey base = AgentKernelKey.create(
                "writer", "model-a", "prompt-v1", List.of(alpha, beta), whitelistVersion);

        assertThat(base.toolManifestFingerprint())
                .isEqualTo("b0a5e724a5889243a5ecb68004caeafee985f98e1bf77e171277c079c27788ee");
        assertThat(base).isEqualTo(AgentKernelKey.create(
                "writer", "model-a", "prompt-v1", List.of(beta, alpha), whitelistVersion));
        assertThat(AgentKernelKey.whitelistVersion("tools-v1", Set.of("alpha", "beta")))
                .isEqualTo(whitelistVersion);
        assertThat(Set.of(
                base,
                AgentKernelKey.create("reviewer", "model-a", "prompt-v1", List.of(alpha, beta), whitelistVersion),
                AgentKernelKey.create("writer", "model-b", "prompt-v1", List.of(alpha, beta), whitelistVersion),
                AgentKernelKey.create("writer", "model-a", "prompt-v2", List.of(alpha, beta), whitelistVersion),
                AgentKernelKey.create("writer", "model-a", "prompt-v1", List.of(alpha), whitelistVersion),
                AgentKernelKey.create("writer", "model-a", "prompt-v1", List.of(alpha, beta), "tools-v2")))
                .hasSize(6);

        AgentKernelSpec spec = spec(base, List.of(alpha, beta), Set.of("alpha", "beta"), whitelistVersion);
        assertThatThrownBy(() -> spec.toolManifest().add(alpha))
                .isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(() -> spec.toolWhitelist().add("gamma"))
                .isInstanceOf(UnsupportedOperationException.class);
        AiModel exposedModel = spec.model();
        exposedModel.setCode("mutated");
        assertThat(spec.model().getCode()).isEqualTo("model-a");
        assertThatThrownBy(() -> spec(
                base, List.of(alpha, beta), Set.of("alpha", "beta"), "mismatched"))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> spec(
                base, List.of(alpha, beta), Set.of("alpha"), whitelistVersion))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("exactly match");
    }

    @Test
    void rejectsKeysThatDoNotMatchThePromptContent() {
        AgentKernelToolManifest alpha = manifest("alpha", "schema-a");
        String whitelistVersion = AgentKernelKey.whitelistVersion("tools-v1", Set.of("alpha"));
        AiModel model = AiModel.builder().id(7L).code("model-a").build();
        AgentKernelKey validKey = AgentKernelKey.create(
                "writer",
                AgentKernelKey.modelRecordFingerprint(model),
                AgentKernelKey.promptVersion("You are a writer"),
                List.of(alpha),
                whitelistVersion);

        assertThatThrownBy(() -> new AgentKernelSpec(
                new AgentKernelKey(
                        validKey.agentDefinitionStableKey(),
                        validKey.modelConfigFingerprint(),
                        "wrong-prompt",
                        validKey.toolManifestFingerprint(),
                        validKey.toolWhitelistVersion()),
                model, "writer", "Writer", "Writes", "You are a writer", 10,
                List.of(alpha), Set.of("alpha"), whitelistVersion))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("prompt");
    }

    @Test
    void buildsMinimalHarnessOnlyThroughRegistryAndNeverClosesSharedStore() {
        CloseableModel delegate = new CloseableModel();
        AtomicInteger registryCalls = new AtomicInteger();
        AtomicInteger resourceCloses = new AtomicInteger();
        AgentKernelToolRegistry registry = (spec, toolkit) -> {
            registryCalls.incrementAndGet();
            toolkit.registerAgentTool(new TestTool());
            return resourceCloses::incrementAndGet;
        };
        AgentStateStore store = mock(AgentStateStore.class);
        StateStoreFailureGuard failures = mock(StateStoreFailureGuard.class);
        AgentScopeHarnessFactory factory = new AgentScopeHarnessFactory(
                ignored -> OwnedChatModel.owned(delegate), registry, store, failures);

        AgentKernelToolManifest testTool = manifest("test_tool", "{}");
        String whitelistVersion = AgentKernelKey.whitelistVersion("tools-v1", Set.of("test_tool"));
        AgentKernelKey key = AgentKernelKey.create(
                "writer", "model-a", "prompt-v1", List.of(testTool), whitelistVersion);
        AgentKernelResource resource = factory.create(
                spec(key, List.of(testTool), Set.of("test_tool"), whitelistVersion));

        assertThat(registryCalls).hasValue(1);
        assertThat(resource.agent().getToolkit().getToolNames()).containsExactly("test_tool");
        assertThat(resource.agent().getStateStore()).isSameAs(store);

        resource.close();
        resource.close();

        assertThat(resourceCloses).hasValue(1);
        assertThat(delegate.closeCount).isOne();
        verify(store, never()).close();
    }

    @Test
    void closesAcquiredResourcesWhenRegistryResultViolatesWhitelist() {
        CloseableModel delegate = new CloseableModel();
        AtomicInteger resourceCloses = new AtomicInteger();
        AgentKernelToolRegistry registry = (spec, toolkit) -> {
            toolkit.registerAgentTool(new TestTool());
            return resourceCloses::incrementAndGet;
        };
        AgentScopeHarnessFactory factory = new AgentScopeHarnessFactory(
                ignored -> OwnedChatModel.owned(delegate),
                registry,
                mock(AgentStateStore.class),
                mock(StateStoreFailureGuard.class));
        AgentKernelKey key = AgentKernelKey.create(
                "writer", "model-a", "prompt-v1", List.of(), "afv-tools-v1");

        assertThatThrownBy(() -> factory.create(
                spec(key, List.of(), Set.of(), "afv-tools-v1")))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("does not match kernel whitelist");
        assertThat(resourceCloses).hasValue(1);
        assertThat(delegate.closeCount).isOne();
    }

    private static AgentKernelSpec spec(
            AgentKernelKey key,
            List<AgentKernelToolManifest> manifest,
            Set<String> whitelist,
            String whitelistVersion) {
        AiModel model = AiModel.builder().id(7L).code("model-a").build();
        AgentKernelKey contentKey = new AgentKernelKey(
                key.agentDefinitionStableKey(),
                AgentKernelKey.modelRecordFingerprint(model),
                AgentKernelKey.promptVersion("You are a writer"),
                key.toolManifestFingerprint(),
                key.toolWhitelistVersion());
        return new AgentKernelSpec(
                contentKey,
                model,
                "writer",
                "Writer",
                "Writes content",
                "You are a writer",
                10,
                manifest,
                whitelist,
                whitelistVersion);
    }

    private static AgentKernelToolManifest manifest(String name, String schema) {
        return new AgentKernelToolManifest(
                name, AgentKernelToolManifest.schemaSha256(schema), false, false);
    }

    private static final class TestTool extends ToolBase {
        private TestTool() {
            super(ToolBase.builder()
                    .name("test_tool")
                    .description("test")
                    .inputSchema(Map.of("type", "object"))
                    .readOnly(true)
                    .concurrencySafe(true));
        }

        @Override
        public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
            return Mono.just(ToolResultBlock.text("ok"));
        }
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
            return "harness-test";
        }

        @Override
        public void close() {
            closeCount++;
        }
    }
}
