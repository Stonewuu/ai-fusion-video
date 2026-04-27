package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.core.http.StreamResponse;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.MessageParam;
import com.anthropic.models.messages.RawMessageStreamEvent;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.formatter.anthropic.AnthropicBaseFormatter;
import io.agentscope.core.formatter.anthropic.AnthropicChatFormatter;
import io.agentscope.core.formatter.anthropic.AnthropicResponseParser;
import io.agentscope.core.message.Msg;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ModelException;
import io.agentscope.core.model.ModelUtils;
import io.agentscope.core.model.ToolSchema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.time.Instant;
import java.util.List;

/**
 * 当前 AgentScope 版本未暴露 Anthropic builder.proxy(ProxyConfig) 时的本地回退实现。
 */
final class ProxyAwareAnthropicChatModel extends ChatModelBase {

    private static final Logger log = LoggerFactory.getLogger(ProxyAwareAnthropicChatModel.class);

    private final String modelName;
    private final boolean streamEnabled;
    private final AnthropicClient client;
    private final GenerateOptions defaultOptions;
    private final AnthropicBaseFormatter formatter;

    ProxyAwareAnthropicChatModel(String baseUrl,
                                 String apiKey,
                                 String modelName,
                                 boolean streamEnabled,
                                 GenerateOptions defaultOptions,
                                 AnthropicBaseFormatter formatter,
                                 ApiConfig apiConfig) {
        this.modelName = modelName;
        this.streamEnabled = streamEnabled;
        this.defaultOptions = defaultOptions != null ? defaultOptions : GenerateOptions.builder().build();
        this.formatter = formatter != null ? formatter : new AnthropicChatFormatter();

        AnthropicOkHttpClient.Builder clientBuilder = AnthropicOkHttpClient.builder();
        if (apiKey != null) {
            clientBuilder.apiKey(apiKey);
        }
        if (baseUrl != null) {
            clientBuilder.baseUrl(baseUrl);
        }
        java.net.Proxy proxy = AiProxySupport.javaProxyOrNull(apiConfig);
        if (proxy != null) {
            clientBuilder.proxy(proxy);
        }
        if (apiConfig != null && StrUtil.isNotBlank(apiConfig.getProxyUsername())) {
            log.warn("[Anthropic] 当前兼容实现仅透传代理地址，代理认证需等待 AgentScope 官方 proxy(ProxyConfig) 支持: model={}",
                    modelName);
        }
        this.client = clientBuilder.build();
    }

    @Override
    protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
        Instant startTime = Instant.now();
        log.debug("Anthropic stream: model={}, messages={}, tools_present={}",
                modelName,
                messages != null ? messages.size() : 0,
                tools != null && !tools.isEmpty());

        Flux<ChatResponse> responseFlux = Flux.defer(() -> {
            try {
                MessageCreateParams.Builder paramsBuilder = MessageCreateParams.builder()
                        .model(modelName)
                        .maxTokens(4096);

                formatter.applySystemMessage(paramsBuilder, messages);

                List<MessageParam> formattedMessages = formatter.format(messages);
                for (MessageParam param : formattedMessages) {
                    paramsBuilder.addMessage(param);
                }

                formatter.applyOptions(paramsBuilder, options, defaultOptions);

                if (tools != null && !tools.isEmpty()) {
                    formatter.applyTools(paramsBuilder, tools);
                }

                MessageCreateParams params = paramsBuilder.build();

                if (streamEnabled) {
                    StreamResponse<RawMessageStreamEvent> streamResponse = client.messages().createStreaming(params);
                    return AnthropicResponseParser.parseStreamEvents(
                            Flux.fromStream(streamResponse.stream())
                                    .subscribeOn(Schedulers.boundedElastic()),
                            startTime)
                            .doFinally(signalType -> {
                                try {
                                    streamResponse.close();
                                } catch (Exception e) {
                                    log.debug("Error closing stream response", e);
                                }
                            });
                }

                return Mono.fromFuture(client.async().messages().create(params))
                        .map(message -> formatter.parseResponse(message, startTime))
                        .flux();
            } catch (Exception e) {
                return Flux.error(new ModelException(
                        "Failed to stream Anthropic API: " + e.getMessage(),
                        e,
                        modelName,
                        "anthropic"));
            }
        });

        return ModelUtils.applyTimeoutAndRetry(responseFlux, options, defaultOptions, modelName, "anthropic");
    }

    @Override
    public String getModelName() {
        return modelName;
    }
}