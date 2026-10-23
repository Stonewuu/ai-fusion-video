package com.stonewu.fusion.integration;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.ai.agentscope.message.MediaResolvingChatModel;
import com.stonewu.fusion.service.media.AgentMediaInputResolver;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.system.SystemConfigService;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.*;
import io.agentscope.core.model.*;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Testcontainers;
import reactor.core.publisher.Flux;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Base64;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

@Testcontainers
class AgentMediaInputIT extends MediaResourceTestContext {
    static AgentMediaInputResolver resolver;
    static AiModel model = AiModel.builder().modelType(1).status(1).multimodalInputTypes(List.of("image", "video", "audio", "file"))
            .multimodalInputTransports(Map.of("image", List.of("url"), "video", List.of("url"), "audio", List.of("url"), "file", List.of("url"))).build();
    @BeforeAll static void inputs() {
        var system = mock(SystemConfigService.class);
        when(system.getPublicResourceBaseUrl()).thenReturn("https://media.example.test");
        context.getBeanFactory().registerSingleton("systemConfigService", system);
        context.register(MediaInputTransport.class, AgentMediaInputResolver.class);
        resolver = context.getBean(AgentMediaInputResolver.class);
    }

    @Test void actualModelBoundaryUsesFreshTicketsAndLeavesNestedPersistedMessagesUnchanged() throws Exception {
        var media = upload(id(), png, "reference.png", 71L, USER);
        String ref = MediaReference.of(media.id());
        var image = ImageBlock.builder().source(URLSource.builder().url(ref).mimeType("image/png").build()).minPixels(128).build();
        var tool = ToolResultBlock.builder().id("tool-call").name("read_asset").metadata(Map.of("version", 7)).output(List.of(image)).build();
        Msg original = Msg.builder().name("user").role(MsgRole.USER).content(List.of(image)).build();
        Msg toolMessage = Msg.builder().name("tool").role(MsgRole.TOOL).content(List.of(tool)).build();
        var runtime = runtime();
        var delegate = new CapturingModel();
        var wrapped = new MediaResolvingChatModel(delegate, model, resolver);
        for (int i = 0; i < 2; i++) wrapped.stream(List.of(original, toolMessage), List.of(), GenerateOptions.builder().build())
                .contextWrite(ctx -> ctx.put(AgentBase.RUNTIME_CONTEXT_KEY, runtime)).blockLast(Duration.ofSeconds(10));
        assertThat(delegate.calls).hasSize(2);
        URLSource first = (URLSource) ((ImageBlock) delegate.calls.get(0).getFirst().getContent().getFirst()).getSource();
        URLSource second = (URLSource) ((ImageBlock) delegate.calls.get(1).getFirst().getContent().getFirst()).getSource();
        assertThat(first.getUrl()).startsWith("https://media.example.test/api/media-content/").isNotEqualTo(second.getUrl());
        String token = first.getUrl().substring(first.getUrl().indexOf("?ticket=") + 8);
        assertThat(tickets.authorize(media.id(), token).resource().id()).isEqualTo(media.id());
        var nested = (ToolResultBlock) delegate.calls.getFirst().get(1).getContent().getFirst();
        assertThat(nested.getMetadata()).containsEntry("version", 7);
        assertThat(((URLSource) ((ImageBlock) nested.getOutput().getFirst()).getSource()).getUrl()).isEqualTo(first.getUrl());
        assertThat(((URLSource) image.getSource()).getUrl()).isEqualTo(ref);
        assertThat(((ImageBlock) delegate.calls.getFirst().getFirst().getContent().getFirst()).getMinPixels()).isEqualTo(128);
        assertThat(wrapped.getContextWindowSize()).isEqualTo(8192);
        wrapped.close(); wrapped.close();
        assertThat(delegate.closes).isEqualTo(1);
        jdbc.update("UPDATE afv_agent_run SET owner_epoch = owner_epoch + 1 WHERE run_id = ?", runtime.get(AgentRunContext.class).runId());
        assertThatThrownBy(() -> tickets.authorize(media.id(), token)).isInstanceOf(MediaException.class);
    }

    @Test void missingExpiredOrSupersededRunNeverReachesProvider() throws Exception {
        var media = upload(id(), png, "guard.png", 71L, USER);
        var input = messages(MediaReference.of(media.id()), "image/png");
        assertThatThrownBy(() -> resolver.resolve(input, null, model)).isInstanceOf(MediaException.class);
        var runtime = runtime();
        jdbc.update("UPDATE afv_agent_run SET owner_epoch = 2 WHERE run_id = ?", runtime.get(AgentRunContext.class).runId());
        var delegate = new CapturingModel();
        var wrapped = new MediaResolvingChatModel(delegate, model, resolver);
        assertThatThrownBy(() -> wrapped.stream(input, List.of(), GenerateOptions.builder().build())
                .contextWrite(ctx -> ctx.put(AgentBase.RUNTIME_CONTEXT_KEY, runtime)).blockLast(Duration.ofSeconds(10))).isInstanceOf(MediaException.class);
        assertThat(delegate.calls).isEmpty();
    }

    @Test void base64OnlyModelReceivesActualPrivateBytesAndPersistedReferenceIsUnchanged() throws Exception {
        var media = upload(id(), png, "private-observation.png", 71L, USER);
        var original = messages(MediaReference.of(media.id()), "image/png");
        var inlineModel = AiModel.builder().modelType(1).status(1).multimodalInputTypes(List.of("image"))
                .multimodalInputTransports(Map.of("image", List.of("base64"))).build();
        var delegate = new CapturingModel();
        var wrapped = new MediaResolvingChatModel(delegate, inlineModel, resolver);
        var runtime = runtime();
        wrapped.stream(original, List.of(), GenerateOptions.builder().build())
                .contextWrite(context -> context.put(AgentBase.RUNTIME_CONTEXT_KEY, runtime)).blockLast(Duration.ofSeconds(10));
        var block = (ImageBlock) delegate.calls.getFirst().getFirst().getContent().getFirst();
        assertThat(block.getSource()).isInstanceOf(Base64Source.class);
        assertThat(Base64.getDecoder().decode(((Base64Source) block.getSource()).getData())).isEqualTo(png);
        assertThat(((URLSource) ((ImageBlock) original.getFirst().getContent().getFirst()).getSource()).getUrl()).isEqualTo(MediaReference.of(media.id()));
        jdbc.update("UPDATE afv_agent_run SET cancel_requested_at=UTC_TIMESTAMP(3) WHERE run_id=?", runtime.get(AgentRunContext.class).runId());
        assertThatThrownBy(() -> resolver.resolve(original, runtime, inlineModel)).isInstanceOf(MediaException.class);
    }

    @Test void declarationMismatchAndRevokedProjectReadAreRejectedBeforeTicketIssuance() throws Exception {
        var media = upload(id(), png, "scope.png", 71L, USER);
        var runtime = runtime();
        assertThatThrownBy(() -> resolver.resolve(messages(MediaReference.of(media.id()), "image/jpeg"), runtime, model))
                .hasMessageContaining("实际类型不一致");
        jdbc.update("UPDATE afv_agent_run SET cancel_requested_at = UTC_TIMESTAMP(6) WHERE run_id = ?", runtime.get(AgentRunContext.class).runId());
        assertThatThrownBy(() -> resolver.resolve(messages(MediaReference.of(media.id()), "image/png"), runtime, model))
                .hasMessageContaining("运行身份已失效");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_access_ticket WHERE resource_id = ?", Integer.class, media.id())).isZero();
    }

    private List<Msg> messages(String url, String mime) {
        return List.of(Msg.builder().name("user").role(MsgRole.USER).content(List.of(ImageBlock.builder()
                .source(URLSource.builder().url(url).mimeType(mime).build()).build())).build());
    }
    private RuntimeContext runtime() {
        String id = id();
        jdbc.update("""
                INSERT INTO afv_agent_run(run_id, conversation_id, user_id, project_id, kernel_fingerprint,
                    agent_definition_snapshot_json, agent_state_session_id, status, owner_instance_id, owner_epoch,
                    lease_until, deadline_at, started_at)
                VALUES (?, ?, 71, 71, 'media-input', '{}', ?, 'RUNNING', 'input-owner', 1,
                    TIMESTAMPADD(MINUTE, 10, UTC_TIMESTAMP(6)), TIMESTAMPADD(HOUR, 1, UTC_TIMESTAMP(6)), UTC_TIMESTAMP(6))
                """, id, id, id);
        return RuntimeContext.builder().userId("71").sessionId(id)
                .put(AuthenticatedUserContext.class, new AuthenticatedUserContext(71L))
                .put(AgentRunContext.class, new AgentRunContext(id, "input-owner", 1, Instant.now().plusSeconds(3600))).build();
    }
    private static final class CapturingModel extends ChatModelBase implements AutoCloseable {
        final List<List<Msg>> calls = new ArrayList<>();
        int closes;
        CapturingModel() { setContextWindowSize(8192); }
        @Override public String getModelName() { return "capture"; }
        @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            calls.add(messages);
            return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("ok").build())).finishReason("stop").build());
        }
        @Override public void close() { closes++; }
    }
}
