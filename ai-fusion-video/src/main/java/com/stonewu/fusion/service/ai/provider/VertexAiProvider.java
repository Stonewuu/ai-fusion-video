package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.google.api.gax.core.FixedCredentialsProvider;
import com.google.api.gax.httpjson.InstantiatingHttpJsonChannelProvider;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.vertexai.Transport;
import com.google.cloud.vertexai.VertexAI;
import com.google.cloud.vertexai.api.LlmUtilityServiceClient;
import com.google.cloud.vertexai.api.LlmUtilityServiceSettings;
import com.google.cloud.vertexai.api.PredictionServiceClient;
import com.google.cloud.vertexai.api.PredictionServiceSettings;
import com.google.genai.GenerationHttpApiClient;
import com.google.genai.types.HttpOptions;
import com.google.genai.Client;
import com.google.genai.types.ClientOptions;
import com.google.genai.types.ProxyOptions;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.transport.ProxyConfig;
import io.agentscope.extensions.model.gemini.GeminiChatModel;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.google.genai.GoogleGenAiChatModel;
import org.springframework.ai.google.genai.GoogleGenAiChatOptions;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatModel;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Vertex AI 提供商。
 */
@Component
@Slf4j
public class VertexAiProvider extends AbstractAiProvider {

    public static final String API_KEY_PLATFORM = VertexAiAuthentication.API_KEY_PLATFORM;
    private static final OkHttpClient CATALOG_CLIENT = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS).readTimeout(60, TimeUnit.SECONDS)
            .callTimeout(90, TimeUnit.SECONDS).build();

    @Override
    public boolean supports(String platform) {
        if (platform == null) {
            return false;
        }
        String normalized = platform.toLowerCase();
        return "vertex_ai".equals(normalized) || "vertexai".equals(normalized) || API_KEY_PLATFORM.equals(normalized);
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        if (usesApiKey(context)) {
            requireApiKey(context.getApiKey(), "Vertex AI API Key");
            var options = GoogleGenAiChatOptions.builder().model(context.getModelName());
            applyDouble(context.getConfig(), "temperature", options::temperature);
            var builder = Client.builder().vertexAI(true).apiKey(context.getApiKey());
            var api = context.getApiConfig();
            if (AiProxySupport.isEnabled(api)) {
                var proxy = ProxyOptions.builder().type("socks5".equals(api.getProxyType()) ? "SOCKS" : "HTTP")
                        .host(api.getProxyHost()).port(api.getProxyPort());
                if (StrUtil.isNotBlank(api.getProxyUsername())) {
                    proxy.username(api.getProxyUsername()).password(api.getProxyPassword());
                }
                builder.clientOptions(ClientOptions.builder().proxyOptions(proxy.build()).build());
            }
            return GoogleGenAiChatModel.builder().genAiClient(builder.build()).defaultOptions(options.build()).build();
        }
        String projectId = getProjectId(context);
        String location = getLocation(context);
        if (StrUtil.isBlank(projectId)) {
            throw new BusinessException("Vertex AI 模型缺少 projectId 配置");
        }

        VertexAiGeminiChatOptions.Builder optionsBuilder = VertexAiGeminiChatOptions.builder()
                .model(context.getModelName());
        applyDouble(context.getConfig(), "temperature", optionsBuilder::temperature);

        String endpoint = resolveVertexEndpoint(location);
        VertexAI.Builder builder = new VertexAI.Builder()
                .setProjectId(projectId)
                .setLocation(location)
                .setApiEndpoint(endpoint)
                .setTransport(Transport.REST);

        GoogleCredentials credentials = loadGoogleCredentials(context);
        if (credentials != null) {
            builder.setCredentials(credentials);
        }
        if (AiProxySupport.isEnabled(context.getApiConfig())) {
            builder.setPredictionClientSupplier(
                    () -> createPredictionServiceClient(context.getApiConfig(), endpoint, credentials));
            builder.setLlmClientSupplier(
                    () -> createLlmUtilityServiceClient(context.getApiConfig(), endpoint, credentials));
        }

        VertexAI vertexAI = builder.build();

        return VertexAiGeminiChatModel.builder()
                .vertexAI(vertexAI)
                .defaultOptions(optionsBuilder.build())
                .build();
    }

    @Override
    public ChatModelBase createAgentScopeModel(AiProviderContext context) {
        if (usesApiKey(context)) {
            requireApiKey(context.getApiKey(), "Vertex AI API Key");
            var options = buildGeminiGenerateOptions(context);
            if (context.isSingleRequest()) {
                return new SingleRequestGeminiModel(new GenerationHttpApiClient(context.getApiKey(), null, null, null,
                        HttpOptions.builder().build(), context.getApiConfig()), context.getModelName(), options);
            }
            var builder = GeminiChatModel.builder().vertexAI(true).apiKey(context.getApiKey())
                    .modelName(context.getModelName()).formatter(GeminiAiProvider.agentScopeFormatter())
                    .streamEnabled(true);
            if (options != null) {
                builder.defaultOptions(options);
            }
            var proxy = AiProxySupport.agentScopeProxyConfig(context.getApiConfig());
            if (proxy != null) {
                builder.proxy(proxy);
            }
            return builder.build();
        }
        String projectId = getProjectId(context);
        if (StrUtil.isBlank(projectId)) {
            throw new BusinessException("Vertex AI 模型缺少 projectId 配置");
        }

        GoogleCredentials credentials = loadGoogleCredentials(context);
        GenerateOptions defaultOptions = buildGeminiGenerateOptions(context);
        if (context.isSingleRequest()) {
            return new SingleRequestGeminiModel(
                    new GenerationHttpApiClient(projectId, getLocation(context), credentials,
                            HttpOptions.builder().build(), context.getApiConfig()), context.getModelName(),
                    defaultOptions);
        }
        GeminiChatModel.Builder builder = GeminiChatModel.builder()
                .modelName(context.getModelName())
                .formatter(GeminiAiProvider.agentScopeFormatter())
                .project(projectId)
                .location(getLocation(context))
                .vertexAI(true)
                .streamEnabled(true);
        if (credentials != null) {
            builder.credentials(credentials);
        }
        if (defaultOptions != null) {
            builder.defaultOptions(defaultOptions);
        }
        ProxyConfig proxy = AiProxySupport.agentScopeProxyConfig(context.getApiConfig());
        if (proxy != null) {
            builder.proxy(proxy);
        }
        return builder.build();
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        Map<String, String> headers = VertexAiAuthentication.headers(context.getApiConfig());
        try {
            return new VertexModelCatalog().load(VertexModelCatalog.ENDPOINT,
                    url -> executeGet(url, headers, context.getApiConfig()));
        } catch (BusinessException failure) {
            if (usesApiKey(context) && failure.getMessage().contains("API keys are not supported")) {
                throw new BusinessException("Google 模型目录接口不接受 API Key。请切换到服务账号方式获取模型，已添加模型仍可使用 API Key 调用。");
            }
            throw failure;
        }
    }

    private boolean usesApiKey(AiProviderContext context) {
        return VertexAiAuthentication.usesApiKey(context.getApiConfig());
    }

    @Override
    protected String executeGet(String url, Map<String, String> headers, ApiConfig apiConfig) {
        var request = new Request.Builder().url(url).get();
        headers.forEach(request::header);
        try (var response = AiProxySupport.okHttpClient(CATALOG_CLIENT, apiConfig).newCall(request.build()).execute()) {
            String body = response.body() == null ? "" : response.body().string();
            if (!response.isSuccessful()) {
                String message = response.message();
                if (JSONUtil.isTypeJSONObject(body)) {
                    var error = JSONUtil.parseObj(body).getJSONObject("error");
                    if (error != null) {
                        message = error.getStr("message", message);
                    }
                }
                throw new BusinessException("获取 Google 模型目录失败：HTTP " + response.code() + " " + message);
            }
            return body;
        } catch (IOException failure) {
            throw new BusinessException("Google 模型目录连接失败，请检查网络或 API 配置中的代理：" + failure.getMessage());
        }
    }

    private String getProjectId(AiProviderContext context) {
        String projectId = getStr(context.getConfig(), "projectId", null);
        if (StrUtil.isNotBlank(projectId)) {
            return projectId;
        }
        projectId = getStr(context.getConfig(), "project", null);
        if (StrUtil.isNotBlank(projectId)) {
            return projectId;
        }
        return context.getApiConfig() != null ? context.getApiConfig().getAppId() : null;
    }

    private String getLocation(AiProviderContext context) {
        String location = getStr(context.getConfig(), "location", null);
        if (StrUtil.isNotBlank(location)) {
            return location;
        }
        if (context.getApiConfig() != null && StrUtil.isNotBlank(context.getApiConfig().getApiUrl())) {
            return context.getApiConfig().getApiUrl();
        }
        return "us-central1";
    }

    private String resolveVertexEndpoint(String location) {
        String resolvedLocation = StrUtil.blankToDefault(location, "us-central1");
        if ("global".equalsIgnoreCase(resolvedLocation)) {
            return "aiplatform.googleapis.com:443";
        }
        return resolvedLocation + "-aiplatform.googleapis.com:443";
    }

    private GoogleCredentials loadGoogleCredentials(AiProviderContext context) {
        return VertexAiAuthentication.credentials(context.getApiConfig());
    }

    private PredictionServiceClient createPredictionServiceClient(ApiConfig apiConfig, String endpoint,
                                                                  GoogleCredentials credentials) {
        try {
            var transportProvider = InstantiatingHttpJsonChannelProvider.newBuilder()
                    .setEndpoint(endpoint)
                    .setHttpTransport(AiProxySupport.googleHttpTransport(apiConfig))
                    .build();
            PredictionServiceSettings.Builder settingsBuilder = PredictionServiceSettings.newHttpJsonBuilder()
                    .setEndpoint(endpoint)
                    .setTransportChannelProvider(transportProvider);
            if (credentials != null) {
                settingsBuilder.setCredentialsProvider(FixedCredentialsProvider.create(credentials));
            }
            return PredictionServiceClient.create(settingsBuilder.build());
        } catch (IOException e) {
            throw new BusinessException("创建 Vertex AI Prediction 代理客户端失败: " + e.getMessage());
        }
    }

    private LlmUtilityServiceClient createLlmUtilityServiceClient(ApiConfig apiConfig, String endpoint,
                                                                  GoogleCredentials credentials) {
        try {
            var transportProvider = InstantiatingHttpJsonChannelProvider.newBuilder()
                    .setEndpoint(endpoint)
                    .setHttpTransport(AiProxySupport.googleHttpTransport(apiConfig))
                    .build();
            LlmUtilityServiceSettings.Builder settingsBuilder = LlmUtilityServiceSettings.newHttpJsonBuilder()
                    .setEndpoint(endpoint)
                    .setTransportChannelProvider(transportProvider);
            if (credentials != null) {
                settingsBuilder.setCredentialsProvider(FixedCredentialsProvider.create(credentials));
            }
            return LlmUtilityServiceClient.create(settingsBuilder.build());
        } catch (IOException e) {
            throw new BusinessException("创建 Vertex AI LLM 代理客户端失败: " + e.getMessage());
        }
    }
}
