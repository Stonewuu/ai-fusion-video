package com.stonewu.fusion.service.ai.provider;

import com.anthropic.backends.AnthropicBackend;
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.AnthropicClientImpl;
import com.anthropic.client.okhttp.OkHttpClient;
import com.anthropic.core.ClientOptions;
import com.anthropic.models.messages.MessageCreateParams;
import io.agentscope.core.message.Msg;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.extensions.model.anthropic.formatter.AnthropicChatFormatter;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

/**
 * 使用现有官方消息格式器，同时显式接入不可重放的传输。
 */
final class SingleRequestAnthropicModel extends ChatModelBase implements AutoCloseable {
    private final AnthropicClient client;
    private final String modelName;
    private final GenerateOptions defaults;
    private final AnthropicChatFormatter formatter = AnthropicAiProvider.agentScopeFormatter();

    SingleRequestAnthropicModel(AiProviderContext context, String baseUrl, GenerateOptions options) {
        this.modelName = context.getModelName();
        this.defaults = SingleRequestModelSupport.options(options);
        var backendBuilder = AnthropicBackend.builder().baseUrl(baseUrl);
        if (GatewayAuthentication.bearer(context.getConfig())) {
            backendBuilder.authToken(context.getApiKey());
        } else {
            backendBuilder.apiKey(context.getApiKey());
        }
        var backend = backendBuilder.build();
        client = new AnthropicClientImpl(ClientOptions.builder()
                .baseUrl(baseUrl).maxRetries(0).timeout(Duration.ofMinutes(25))
                .httpClient(new OkHttpClient(SingleRequestModelSupport.client(context.getApiConfig()), backend))
                .build());
    }

    @Override
    protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
        return Flux.defer(() -> {
            var started = Instant.now();
            var params = MessageCreateParams.builder().model(modelName).maxTokens(4096);
            formatter.applySystemMessage(params, messages);
            formatter.format(messages).forEach(params::addMessage);
            formatter.applyOptions(params, SingleRequestModelSupport.options(options), defaults);
            if (tools != null && !tools.isEmpty()) {
                formatter.applyTools(params, tools);
            }
            var responses = new StreamingAnthropicResponses(started);
            return Flux.using(() -> client.messages().createStreaming(params.build()),
                    stream -> Flux.fromStream(stream.stream()).concatMapIterable(responses::accept),
                    stream -> stream.close());
        }).subscribeOn(Schedulers.boundedElastic());
    }

    @Override
    public String getModelName() {
        return modelName;
    }

    @Override
    public void close() {
        client.close();
    }
}
