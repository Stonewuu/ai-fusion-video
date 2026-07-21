package com.stonewu.fusion.service.ai.agentscope;

import com.stonewu.fusion.entity.ai.AiModel;
import io.agentscope.core.model.ChatModelBase;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * AgentScope 模型适配工厂
 * <p>
 * 从数据库 AiModel 配置动态创建 AgentScope 的 Model 实例。
 * 与现有 ChatModelFactory 并行存在，两者使用不同的模型体系：
 * - ChatModelFactory → Spring AI ChatModel
 * - AgentScopeModelFactory → AgentScope Model
 */
@Component
@RequiredArgsConstructor
public class AgentScopeModelFactory {

    private final AiProviderService aiProviderService;

    /** 缓存已创建的 AgentScope Model 实例（按 AiModel.id 缓存） */
    private final Map<Long, ChatModelBase> modelCache = new ConcurrentHashMap<>();

    /**
     * 根据 AiModel 配置创建或返回缓存的 AgentScope Model
     */
    public ChatModelBase getOrCreate(AiModel model) {
        return modelCache.computeIfAbsent(model.getId(), id -> createModel(model));
    }

    public void evict(Long modelId) {
        modelCache.remove(modelId);
    }

    public void evictAll() {
        modelCache.clear();
    }

    private ChatModelBase createModel(AiModel model) {
        return aiProviderService.createAgentScopeModel(model);
    }
}
