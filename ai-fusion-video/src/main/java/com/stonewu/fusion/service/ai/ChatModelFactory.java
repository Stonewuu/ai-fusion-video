package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
@RequiredArgsConstructor
public class ChatModelFactory {

    private final AiProviderService aiProviderService;

    /**
     * 缓存已创建的 ChatModel 实例（按 AiModel.id 缓存）
     * <p>
     * 修改模型配置后需调用 {@link #evict(Long)} 清除缓存
     */
    private final Map<Long, ChatModel> modelCache = new ConcurrentHashMap<>();

    /**
     * 根据 AiModel 配置创建或返回缓存的 ChatModel
     */
    public ChatModel getOrCreate(AiModel model) {
        return modelCache.computeIfAbsent(model.getId(), id -> createChatModel(model));
    }

    /**
     * 清除指定模型缓存
     */
    public void evict(Long modelId) {
        modelCache.remove(modelId);
    }

    /**
     * 清除全部缓存
     */
    public void evictAll() {
        modelCache.clear();
    }

    private ChatModel createChatModel(AiModel model) {
        return aiProviderService.createChatModel(model);
    }
}
