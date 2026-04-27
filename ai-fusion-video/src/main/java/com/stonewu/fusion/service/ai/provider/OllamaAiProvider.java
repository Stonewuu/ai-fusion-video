package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import io.agentscope.core.model.Model;
import io.agentscope.core.model.OllamaChatModel;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.ollama.OllamaChatModel.Builder;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaChatOptions;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Ollama 提供商。
 */
@Component
@Slf4j
public class OllamaAiProvider extends AbstractAiProvider {

    @Override
    public boolean supports(String platform) {
        return platform != null && "ollama".equalsIgnoreCase(platform);
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        String baseUrl = normalizeBaseUrl(context.getBaseUrl());
        if (StrUtil.isBlank(baseUrl)) {
            baseUrl = "http://localhost:11434";
        }

        OllamaChatOptions.Builder optionsBuilder = OllamaChatOptions.builder().model(context.getModelName());
        applyDouble(context.getConfig(), "temperature", optionsBuilder::temperature);
        applyDouble(context.getConfig(), "topP", optionsBuilder::topP);

        return org.springframework.ai.ollama.OllamaChatModel.builder()
                .ollamaApi(OllamaApi.builder().baseUrl(baseUrl).build())
                .defaultOptions(optionsBuilder.build())
                .build();
    }

    @Override
    public Model createAgentScopeModel(AiProviderContext context) {
        String baseUrl = normalizeBaseUrl(context.getBaseUrl());
        if (StrUtil.isBlank(baseUrl)) {
            baseUrl = "http://localhost:11434";
        }
        return OllamaChatModel.builder()
                .modelName(context.getModelName())
                .baseUrl(baseUrl)
                .build();
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        String baseUrl = normalizeBaseUrl(context.getBaseUrl());
        if (StrUtil.isBlank(baseUrl)) {
            baseUrl = "http://localhost:11434";
        }
        String url = baseUrl + "/api/tags";
        log.info("[OllamaAiProvider] 获取远程模型列表: {}", url);
        String response = executeGet(url, java.util.Map.of(), context.getApiConfig());
        return parseOllamaTags(response);
    }
}