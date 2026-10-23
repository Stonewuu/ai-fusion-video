package com.stonewu.fusion.service.ai.provider;

import com.google.genai.GenerationContentStream;
import com.google.genai.GenerationHttpApiClient;
import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import io.agentscope.core.formatter.Formatter;
import io.agentscope.core.message.Msg;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

final class SingleRequestGeminiModel extends ChatModelBase implements AutoCloseable {
    private final GenerationHttpApiClient client;
    private final String modelName;
    private final GenerateOptions defaults;
    private final Formatter<Content, GenerateContentResponse, GenerateContentConfig.Builder> formatter = GeminiAiProvider.agentScopeFormatter();

    SingleRequestGeminiModel(GenerationHttpApiClient client, String modelName, GenerateOptions options) {
        this.client = client;
        this.modelName = modelName;
        this.defaults = SingleRequestModelSupport.options(options);
    }

    @Override
    protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
        return Flux.defer(() -> {
            var started = Instant.now();
            String responseId = UUID.randomUUID().toString();
            var config = GenerateContentConfig.builder();
            formatter.applyOptions(config, options, defaults);
            if (tools != null && !tools.isEmpty()) {
                formatter.applyTools(config, tools);
            }
            return GenerationContentStream.generate(client, modelName, formatter.format(messages), config.build())
                    .map(result -> {
                        ChatResponse parsed = formatter.parseResponse(result, started);
                        String finish = result.candidates().orElse(List.of()).stream().findFirst()
                                .flatMap(candidate -> candidate.finishReason()).map(Object::toString).orElse(null);
                        return new ChatResponse(responseId, parsed.getContent(), parsed.getUsage(),
                                parsed.getMetadata(), finish);
                    });
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
