package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * AI Agent 配置服务
 * <p>
 * 从 {@link AiAgentRegistry} 读取代码中定义的 Agent 配置。
 */
@Service
@RequiredArgsConstructor
public class AiAgentService {

    private final AiAgentRegistry agentRegistry;

    /**
     * 获取所有 Agent 定义
     */
    public List<AiAgentDefinition> getAll() {
        return agentRegistry.getAll();
    }

    /**
     * 根据类型获取 Agent 定义
     *
     * @param type Agent 类型
     * @return Agent 定义，不存在返回 null
     */
    public AiAgentDefinition getByType(String type) {
        return agentRegistry.getByType(type);
    }

    /**
     * 根据类型获取 Agent 定义（不存在则抛异常）
     *
     * @param type Agent 类型
     * @return Agent 定义
     * @throws BusinessException 如果类型不存在
     */
    public AiAgentDefinition getRequiredByType(String type) {
        AiAgentDefinition definition = agentRegistry.getByType(type);
        if (definition == null) {
            throw new BusinessException("Agent 类型不存在: " + type);
        }
        return definition;
    }
}
