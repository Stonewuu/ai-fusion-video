package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.vertexai.VertexAI;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import io.agentscope.core.model.GeminiChatModel;
import io.agentscope.core.model.Model;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatModel;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;

/**
 * Vertex AI 提供商。
 */
@Component
@Slf4j
public class VertexAiProvider extends AbstractAiProvider {

    private static final String VERTEX_AI_SCOPE = "https://www.googleapis.com/auth/cloud-platform";

    @Override
    public boolean supports(String platform) {
        if (platform == null) {
            return false;
        }
        String normalized = platform.toLowerCase();
        return "vertex_ai".equals(normalized) || "vertexai".equals(normalized);
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        String projectId = getProjectId(context);
        String location = getLocation(context);
        if (StrUtil.isBlank(projectId)) {
            throw new BusinessException("Vertex AI 模型缺少 projectId 配置");
        }

        VertexAiGeminiChatOptions.Builder optionsBuilder = VertexAiGeminiChatOptions.builder()
                .model(context.getModelName());
        applyDouble(context.getConfig(), "temperature", optionsBuilder::temperature);

        VertexAI.Builder builder = new VertexAI.Builder()
            .setProjectId(projectId)
            .setLocation(location);

        GoogleCredentials credentials = loadServiceAccountCredentials(context);
        if (credentials != null) {
            builder.setCredentials(credentials);
        }

        VertexAI vertexAI = builder.build();

        return VertexAiGeminiChatModel.builder()
                .vertexAI(vertexAI)
                .defaultOptions(optionsBuilder.build())
                .build();
    }

    @Override
    public Model createAgentScopeModel(AiProviderContext context) {
        String projectId = getProjectId(context);
        if (StrUtil.isBlank(projectId)) {
            throw new BusinessException("Vertex AI 模型缺少 projectId 配置");
        }

        GeminiChatModel.Builder builder = GeminiChatModel.builder()
                .modelName(context.getModelName())
                .streamEnabled(true);

        builder.project(projectId)
                .location(getLocation(context))
                .vertexAI(true);

        GoogleCredentials credentials = loadServiceAccountCredentials(context);
        if (credentials != null) {
            builder.credentials(credentials);
        }
        return builder.build();
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        throw new BusinessException("Vertex AI 暂未接入自动获取模型列表，请手动填写模型 code");
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

    private GoogleCredentials loadServiceAccountCredentials(AiProviderContext context) {
        String appSecret = context.getApiConfig() != null ? context.getApiConfig().getAppSecret() : null;
        if (StrUtil.isBlank(appSecret)) {
            return null;
        }
        try {
            return GoogleCredentials.fromStream(
                    new ByteArrayInputStream(appSecret.getBytes(StandardCharsets.UTF_8)))
                    .createScoped(Collections.singletonList(VERTEX_AI_SCOPE));
        } catch (IOException e) {
            throw new BusinessException("Vertex AI 服务账号 JSON Key 无效: " + e.getMessage());
        }
    }
}