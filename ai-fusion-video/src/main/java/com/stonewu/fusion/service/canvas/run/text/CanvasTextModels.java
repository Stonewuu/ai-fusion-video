package com.stonewu.fusion.service.canvas.run.text;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasTextModels {
    private final AiModelService models;
    private final AiProviderService providers;
    private final GenerationModelSnapshot apis;
    private final CanvasModelCatalog catalog;
    private final CanvasJson json;

    public ObjectNode freeze(long modelId) {
        var model = require(modelId);
        if (!catalog.require(modelId).kind().equals("text")) {
            throw CanvasRunException.invalid("请选择文本模型");
        }
        return json.object(Map.of("model", model, "providerHash", providers.agentScopeModelFingerprint(model)));
    }

    public void verify(CanvasTextData.Input input) {
        require(input.model().getId());
        apis.requireApi(input.model().getApiConfigId());
        if (!providers.agentScopeModelFingerprint(input.model()).equals(input.providerHash())) {
            throw CanvasRunException.conflict("文本模型的供应商配置已变化，请重新确认运行计划");
        }
    }

    public OwnedChatModel provision(CanvasTextData.Input input) {
        verify(input);
        var provision = providers.provisionSingleRequestModel(input.model());
        var owned = OwnedChatModel.owned(provision.model());
        if (!input.providerHash().equals(provision.modelConfigFingerprint())) {
            owned.close();
            throw CanvasRunException.conflict("建立文本调用时供应商配置发生变化，未发送请求");
        }
        return owned;
    }

    private AiModel require(long id) {
        var model = models.getById(id);
        if (model == null || !Integer.valueOf(1).equals(model.getStatus()) || !Integer.valueOf(1)
                .equals(model.getModelType())) {
            throw CanvasRunException.invalid("文本模型不存在、已停用或类型不匹配");
        }
        if (model.getMaxConcurrency() == null || model.getMaxConcurrency() < 1) {
            throw CanvasRunException.invalid("文本模型必须设置正数并发上限");
        }
        apis.requireApi(model.getApiConfigId());
        return json.read(json.write(model), AiModel.class);
    }
}
