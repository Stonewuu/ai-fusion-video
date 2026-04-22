package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.google.genai.Client;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import io.agentscope.core.model.GeminiChatModel;
import io.agentscope.core.model.Model;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.google.genai.GoogleGenAiChatModel;
import org.springframework.ai.google.genai.GoogleGenAiChatOptions;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Gemini Developer API 提供商。
 */
@Component
@Slf4j
public class GeminiAiProvider extends AbstractAiProvider {

    @Override
    public boolean supports(String platform) {
        if (platform == null) {
            return false;
        }
        return "gemini".equals(platform.trim().toLowerCase(Locale.ROOT));
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Gemini");

        GoogleGenAiChatOptions.Builder optionsBuilder = GoogleGenAiChatOptions.builder()
                .model(context.getModelName());
        applyDouble(context.getConfig(), "temperature", optionsBuilder::temperature);

        Client genAiClient = Client.builder()
                .apiKey(context.getApiKey())
                .build();

        return GoogleGenAiChatModel.builder()
                .genAiClient(genAiClient)
                .defaultOptions(optionsBuilder.build())
                .build();
    }

    @Override
    public Model createAgentScopeModel(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Gemini");
        return GeminiChatModel.builder()
                .apiKey(context.getApiKey())
                .modelName(context.getModelName())
                .streamEnabled(true)
                .vertexAI(false)
                .build();
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Gemini");
        String apiBaseUrl = resolveGeminiApiBaseUrl(context.getBaseUrl());
        String url = joinUrl(apiBaseUrl, "/v1beta/models?pageSize=1000");
        log.info("[GeminiAiProvider] 获取 Gemini 远程模型列表: {}", url);

        String response = executeGet(url, Map.of("x-goog-api-key", context.getApiKey()));
        return parseGeminiModels(response);
    }

    private String resolveGeminiApiBaseUrl(String baseUrl) {
        return StrUtil.isBlank(baseUrl)
                ? "https://generativelanguage.googleapis.com"
                : normalizeBaseUrl(baseUrl);
    }
}