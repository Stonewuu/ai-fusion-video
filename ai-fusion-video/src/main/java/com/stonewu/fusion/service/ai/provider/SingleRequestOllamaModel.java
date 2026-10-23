package com.stonewu.fusion.service.ai.provider;

import io.agentscope.core.message.Msg;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.model.transport.HttpTransport;
import io.agentscope.extensions.model.ollama.OllamaHttpClient;
import io.agentscope.extensions.model.ollama.formatter.OllamaChatFormatter;
import io.agentscope.extensions.model.ollama.options.OllamaOptions;
import reactor.core.publisher.Flux;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

final class SingleRequestOllamaModel extends ChatModelBase implements AutoCloseable {
    private final HttpTransport transport;
    private final OllamaHttpClient client;
    private final String modelName;
    private final OllamaOptions defaults;
    private final OllamaChatFormatter formatter = new OllamaChatFormatter();

    SingleRequestOllamaModel(AiProviderContext context, String baseUrl, GenerateOptions options) {
        this.transport = SingleRequestModelSupport.transport(context);
        this.client = new OllamaHttpClient(transport, baseUrl);
        this.modelName = context.getModelName();
        this.defaults = OllamaOptions.fromGenerateOptions(options);
    }

    @Override
    protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
        return Flux.defer(() -> {
            var started = Instant.now();
            String responseId = UUID.randomUUID().toString();
            var request = formatter.buildRequest(modelName, formatter.format(messages), true,
                    OllamaOptions.fromGenerateOptions(options), defaults, tools,
                    options == null ? null : options.getToolChoice());
            return client.stream(request)
                    .map(response -> formatter.parseResponse(response, started).withId(responseId));
        });
    }

    @Override
    public String getModelName() {
        return modelName;
    }

    @Override
    public void close() {
        transport.close();
    }
}
