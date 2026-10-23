package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.List;
import java.time.Duration;
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

    /** 使用与画布、助手一致的单次请求协议，包含仅支持 Responses 的模型。 */
    @Transactional(propagation = Propagation.NEVER)
    public String checkConnectivity(AiModel model, String message) {
        try (var owned = OwnedChatModel.owned(aiProviderService.provisionSingleRequestModel(model).model())) {
            var input = Msg.builder().name("user").role(MsgRole.USER)
                    .content(TextBlock.builder().text(message).build()).build();
            var parts = owned.model().stream(List.of(input), List.of(), null)
                    .concatMapIterable(response -> response.getContent()).ofType(TextBlock.class)
                    .map(TextBlock::getText).collectList().block(Duration.ofSeconds(60));
            return String.join("", parts);
        }
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
