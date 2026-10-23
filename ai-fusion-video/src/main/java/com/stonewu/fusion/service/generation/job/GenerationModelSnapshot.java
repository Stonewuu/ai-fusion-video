package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ProviderConfigurationFingerprint;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class GenerationModelSnapshot {
    private final AiModelService models;
    private final ApiConfigService apis;
    private final AiModelMetadataResolver metadata;
    private final GenerationModelCapabilityService capabilities;
    private final CanvasJson json;

    public AiModel freeze(Long id, GenerationKind kind) {
        return freeze(id, kind, null);
    }

    public AiModel freeze(Long id, GenerationKind kind, Long workflowVersionId) {
        if (id == null) {
            throw GenerationJobException.invalid("请选择生成模型");
        }
        var current = models.getById(id);
        int expectedType = kind == GenerationKind.IMAGE ? 2 : 3;
        if (current == null || !Integer.valueOf(1).equals(current.getStatus())
                || !Integer.valueOf(expectedType).equals(current.getModelType())) {
            throw GenerationJobException.invalid("所选生成模型不存在、已停用或类型不匹配");
        }
        ApiConfig api = requireApi(current.getApiConfigId());
        if (current.getMaxConcurrency() == null || current.getMaxConcurrency() < 1) {
            throw GenerationJobException.invalid("模型必须配置正数并发上限");
        }
        AiModel copy = json.read(json.write(current), AiModel.class);
        copy.setConfig(capabilities.getMergedModelConfig(current, workflowVersionId).toString());
        copy.setGenerationConfigSnapshot(true);
        copy.setModelProtocol(metadata.resolve(current).modelProtocol());
        copy.setCapabilityPresetCode(null); // 已展开的能力不会在执行期间随预设变化。
        if (copy.getModelProtocol() == null || copy.getModelProtocol().isBlank()
                || copy.getCode() == null || copy.getCode().isBlank()) {
            throw GenerationJobException.invalid("模型缺少已登记的请求协议或模型代码");
        }
        if (!api.getId().equals(copy.getApiConfigId())) {
            throw GenerationJobException.invalid("供应商配置不一致");
        }
        return copy;
    }

    public ApiConfig requireApi(Long id) {
        if (id == null) {
            throw GenerationJobException.invalid("模型缺少供应商配置");
        }
        var api = apis.getById(id);
        if (api == null || !Integer.valueOf(1).equals(api.getStatus())) {
            throw new GenerationJobException(409, "GENERATION_PROVIDER_UNAVAILABLE", "供应商配置已停用或不存在");
        }
        return api;
    }

    public AiModel frozen(AiModel snapshot, GenerationKind kind, String apiHash) {
        var current = models.getById(snapshot.getId());
        if (current == null || !Integer.valueOf(1).equals(current.getStatus())
                || !Integer.valueOf(kind == GenerationKind.IMAGE ? 2 : 3).equals(snapshot.getModelType())) {
            throw GenerationJobException.invalid("运行计划中的模型已停用或类型不匹配");
        }
        if (!fingerprint(requireApi(snapshot.getApiConfigId())).equals(apiHash)) {
            throw GenerationJobException.invalid("运行计划中的供应商连接已变化，请重新确认");
        }
        AiModel copy = json.read(json.write(snapshot), AiModel.class);
        copy.setGenerationConfigSnapshot(true);
        return copy;
    }

    public ApiConfig forExecution(GenerationSubmission submission) {
        var api = requireApi(submission.apiConfigId());
        if (!fingerprint(api).equals(submission.apiConfigHash())) {
            throw new GenerationJobException(409, "GENERATION_PROVIDER_CHANGED",
                    "供应商连接配置已变化，请核实原任务后继续");
        }
        var current = models.getById(submission.model().getId());
        if (current == null || !Integer.valueOf(1).equals(current.getStatus())) {
            throw new GenerationJobException(409, "GENERATION_MODEL_DISABLED", "模型已停用，任务需要核实处理");
        }
        return api;
    }

    public String fingerprint(ApiConfig api) {
        return ProviderConfigurationFingerprint.of(api, json);
    }
}
