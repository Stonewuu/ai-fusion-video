package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import io.agentscope.core.model.ChatModelBase;
import org.springframework.ai.chat.model.ChatModel;

import java.util.List;

/**
 * 统一的 AI 提供商接口。
 *
 * 每个提供商负责自身的平台能力：
 * - 创建 Spring AI ChatModel
 * - 创建 AgentScope Model
 * - 获取远程可用模型列表
 */
public interface AiProvider {

    /**
     * 是否支持当前平台标识
     */
    boolean supports(String platform);

    /**
     * 创建 Spring AI ChatModel
     */
    ChatModel createChatModel(AiProviderContext context);

    /**
     * 创建 AgentScope Model
     */
    ChatModelBase createAgentScopeModel(AiProviderContext context);

    /**
     * 获取远程可用模型列表
     */
    List<RemoteModelVO> listRemoteModels(AiProviderContext context);
}
