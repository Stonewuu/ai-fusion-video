package com.stonewu.fusion.build;

import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.AssistantMessage;
import io.agentscope.core.message.AudioBlock;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.ImageBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.URLSource;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.message.VideoBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import io.agentscope.harness.agent.HarnessAgent;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class AgentScopeGaApiContractTests {

    @Test
    void runtimeContextRetainsTypedValuesAndRemovesNullUpdates() {
        RuntimeContext context = RuntimeContext.builder()
                .userId("42")
                .sessionId("afv:v2:c:a")
                .put(String.class, "typed")
                .put("label", "plain")
                .build();

        assertThat(context.getUserId()).isEqualTo("42");
        assertThat(context.getSessionId()).isEqualTo("afv:v2:c:a");
        assertThat(context.get(String.class)).isEqualTo("typed");
        assertThat(context.get("label", String.class)).isEqualTo("plain");
        assertThat(context.get((Class<String>) null)).isNull();
        assertThat(context.<String>get((String) null)).isNull();

        context.put(String.class, null);
        context.put("label", null);

        assertThat(context.get(String.class)).isNull();
        assertThat(context.get("label", String.class)).isNull();
    }

    @Test
    void chatModelAndHarnessUseGaRuntimeContracts() {
        EchoModel model = new EchoModel();
        Msg request = Msg.builder()
                .role(MsgRole.USER)
                .content(TextBlock.builder().text("hello").build())
                .build();

        StepVerifier.create(model.stream(List.of(request), List.of(), GenerateOptions.builder().build()))
                .assertNext(response -> {
                    assertThat(response.getFinishReason()).isEqualTo("stop");
                    assertThat(response.getContent()).hasSize(1);
                    assertThat(response.getContent().getFirst()).isInstanceOf(TextBlock.class);
                })
                .verifyComplete();
        assertThat(model.getModelName()).isEqualTo("echo");

        assertThat(HarnessAgent.builder()
                .name("test")
                .model(model)
                .stateStore(new InMemoryAgentStateStore())
                .disableSessionPersistence())
                .isNotNull();
    }

    @Test
    void messagesPreserveStronglyTypedMediaBlocks() {
        URLSource imageSource = URLSource.builder()
                .url("https://example.test/image.png")
                .mimeType("image/png")
                .build();
        ImageBlock image = ImageBlock.builder()
                .source(imageSource)
                .minPixels(128)
                .maxPixels(1024)
                .build();
        AudioBlock audio = AudioBlock.builder()
                .source(Base64Source.builder()
                        .mediaType("audio/wav")
                        .data("YXVkaW8=")
                        .build())
                .build();
        VideoBlock video = VideoBlock.builder()
                .source(URLSource.builder()
                        .url("https://example.test/video.mp4")
                        .mimeType("video/mp4")
                        .build())
                .fps(24.0F)
                .maxFrames(12)
                .build();

        Msg message = Msg.builder()
                .role(MsgRole.USER)
                .content(TextBlock.builder().text("describe this media").build(), image, audio, video)
                .build();

        assertThat(message.getTextContent()).isEqualTo("describe this media");
        assertThat(message.getContentBlocks(ImageBlock.class)).containsExactly(image);
        assertThat(message.getContentBlocks(AudioBlock.class)).containsExactly(audio);
        assertThat(message.getContentBlocks(VideoBlock.class)).containsExactly(video);
        assertThat(image.getSource()).isSameAs(imageSource);
        assertThat(image.getMaxPixels()).isEqualTo(1024);
        assertThat(((Base64Source) audio.getSource()).getMediaType()).isEqualTo("audio/wav");
        assertThat(video.getFps()).isEqualTo(24.0F);
        assertThat(((URLSource) video.getSource()).getMimeType()).isEqualTo("video/mp4");
    }

    @Test
    void typedUserAndAssistantMessagesPreserveRolesAndContent() {
        UserMessage userFromBuilder = UserMessage.builder()
                .name("user-builder")
                .textContent("builder question")
                .build();
        UserMessage userFromConstructor = new UserMessage(
                "user-constructor", TextBlock.builder().text("constructor question").build());
        AssistantMessage assistantFromBuilder = AssistantMessage.builder()
                .name("assistant-builder")
                .textContent("builder answer")
                .build();
        AssistantMessage assistantFromConstructor = new AssistantMessage(
                "assistant-constructor", TextBlock.builder().text("constructor answer").build());

        assertThat(userFromBuilder.getRole()).isEqualTo(MsgRole.USER);
        assertThat(userFromConstructor.getRole()).isEqualTo(MsgRole.USER);
        assertThat(assistantFromBuilder.getRole()).isEqualTo(MsgRole.ASSISTANT);
        assertThat(assistantFromConstructor.getRole()).isEqualTo(MsgRole.ASSISTANT);
        assertThat(userFromBuilder.getTextContent()).isEqualTo("builder question");
        assertThat(userFromConstructor.getTextContent()).isEqualTo("constructor question");
        assertThat(assistantFromBuilder.getTextContent()).isEqualTo("builder answer");
        assertThat(assistantFromConstructor.getTextContent()).isEqualTo("constructor answer");
    }

    @Test
    void toolContractsExposeRuntimeContextAndBuilderStaysAbstract() throws Exception {
        RuntimeContext context = RuntimeContext.builder().put(String.class, "runtime-value").build();
        ToolUseBlock toolUse = ToolUseBlock.builder()
                .id("call-123")
                .name("echo_tool")
                .input(Map.of("value", "payload"))
                .build();
        ToolCallParam param = ToolCallParam.builder()
                .toolUseBlock(toolUse)
                .input(Map.of("value", "payload"))
                .runtimeContext(context)
                .build();
        EchoTool tool = new EchoTool();

        assertThat(ToolCallParam.class.getMethod("getRuntimeContext")).isNotNull();
        assertThat(Arrays.stream(ToolBase.Builder.class.getDeclaredMethods())
                .noneMatch(method -> method.getName().equals("build"))).isTrue();
        assertThat(param.getRuntimeContext()).isSameAs(context);
        assertThat(param.getInput()).containsEntry("value", "payload");
        assertThat(tool.getName()).isEqualTo("echo_tool");
        assertThat(tool.getDescription()).isEqualTo("returns its runtime context value");
        assertThat(tool.getParameters()).containsEntry("type", "object");
        assertThat(tool.isReadOnly()).isTrue();
        assertThat(tool.isConcurrencySafe()).isTrue();

        StepVerifier.create(tool.callAsync(param))
                .assertNext(result -> {
                    assertThat(result.getId()).isEqualTo("call-123");
                    assertThat(result.getName()).isEqualTo("echo_tool");
                    assertThat(((TextBlock) result.getOutput().getFirst()).getText())
                            .isEqualTo("runtime-value:payload");
                })
                .verifyComplete();
    }

    @SuppressWarnings("unused")
    private static void interruptTargeted(Agent agent, Msg message) {
        agent.interrupt(message);
    }

    private static final class EchoModel extends ChatModelBase {

        @Override
        protected Flux<ChatResponse> doStream(
                List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            return Flux.just(ChatResponse.builder()
                    .content(List.of(TextBlock.builder().text(messages.getFirst().getTextContent()).build()))
                    .finishReason("stop")
                    .build());
        }

        @Override
        public String getModelName() {
            return "echo";
        }
    }

    private static final class EchoTool extends ToolBase {

        private EchoTool() {
            super(ToolBase.builder()
                    .name("echo_tool")
                    .description("returns its runtime context value")
                    .inputSchema(Map.of("type", "object"))
                    .readOnly(true)
                    .concurrencySafe(true));
        }

        @Override
        public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
            String value = param.getRuntimeContext().get(String.class);
            return Mono.just(ToolResultBlock.text(value + ":" + param.getInput().get("value"))
                    .withIdAndName(param.getToolUseBlock().getId(), param.getToolUseBlock().getName()));
        }
    }
}
