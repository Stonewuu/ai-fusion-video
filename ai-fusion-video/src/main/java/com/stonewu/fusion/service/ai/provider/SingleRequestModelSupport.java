package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import io.agentscope.core.message.Msg;
import io.agentscope.core.model.*;
import io.agentscope.core.model.transport.HttpTransport;
import io.agentscope.core.model.transport.HttpTransportConfig;
import io.agentscope.core.model.transport.OkHttpTransport;
import okhttp3.OkHttpClient;
import reactor.core.publisher.Flux;

import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 一个模型实例只承载一次持久化提交，禁止 SDK 在新 HTTP Call 内再次发送。
 */
public final class SingleRequestModelSupport {
    private SingleRequestModelSupport() {
    }

    public static OkHttpClient client(ApiConfig api) {
        var sent = new AtomicBoolean();
        var builder = GenerationHttpClients.providerRequestBuilder()
                .connectTimeout(Duration.ofMinutes(1)).readTimeout(Duration.ofMinutes(25))
                .writeTimeout(Duration.ofMinutes(1)).callTimeout(Duration.ofMinutes(25))
                .addInterceptor(chain -> {
                    if (!sent.compareAndSet(false, true)) {
                        throw new IOException("模型调用已经发送，必须核实原任务，不能自动重发");
                    }
                    return chain.proceed(chain.request());
                });
        return AiProxySupport.applyToOkHttp(builder, api).build();
    }

    public static HttpTransport transport(AiProviderContext context) {
        if (!context.isSingleRequest()) {
            return AiProxySupport.agentScopeHttpTransport(context.getApiConfig());
        }
        return new OkHttpTransport(client(context.getApiConfig()), HttpTransportConfig.builder()
                .connectTimeout(Duration.ofMinutes(1)).readTimeout(Duration.ofMinutes(25))
                .writeTimeout(Duration.ofMinutes(1)).build());
    }

    public static GenerateOptions options(GenerateOptions options) {
        return GenerateOptions.mergeOptions(GenerateOptions.builder()
                .stream(true).executionConfig(ExecutionConfig.builder()
                        .maxAttempts(1).timeout(Duration.ofMinutes(25)).build()).build(), options);
    }

    public static ChatModelBase own(ChatModelBase model, HttpTransport transport) {
        return new OwnedSingleRequestModel(model, transport);
    }

    private static final class OwnedSingleRequestModel extends ChatModelBase implements AutoCloseable {
        private final ChatModelBase model;
        private final HttpTransport transport;

        OwnedSingleRequestModel(ChatModelBase model, HttpTransport transport) {
            this.model = model;
            this.transport = transport;
        }

        @Override
        protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            return model.stream(messages, tools, SingleRequestModelSupport.options(options));
        }

        @Override
        public String getModelName() {
            return model.getModelName();
        }

        @Override
        public int getContextWindowSize() {
            return model.getContextWindowSize();
        }

        @Override
        public boolean supportsNativeStructuredOutput() {
            return model.supportsNativeStructuredOutput();
        }

        @Override
        public boolean supportsNativeStructuredOutputWithTools() {
            return model.supportsNativeStructuredOutputWithTools();
        }

        @Override
        public void close() {
            transport.close();
        }
    }
}
