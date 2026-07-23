package com.stonewu.fusion.service.ai;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.ai.vo.AiModelConnectivityRespVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import cn.hutool.core.util.StrUtil;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class AiModelService {

    private static final int MODEL_TYPE_TEXT = 1;
    private static final String CONNECTIVITY_TEST_MESSAGE = "Connectivity test. Reply with OK only.";

    private final AiModelMapper aiModelMapper;
    private final ApiConfigService apiConfigService;
    private final ModelPresetService modelPresetService;
    private final AiModelMetadataResolver aiModelMetadataResolver;
    private final ChatModelFactory chatModelFactory;

    @Transactional
    public Long createAiModel(AiModel aiModel) {
        validateApiConfig(aiModel.getApiConfigId(), true);
        validateUniqueCode(null, aiModel.getApiConfigId(), aiModel.getCode());
        normalizeMetadata(aiModel);
        validateRequestProtocol(aiModel);
        validateCapabilityPreset(aiModel);
        try {
            aiModelMapper.insert(aiModel);
        } catch (DuplicateKeyException e) {
            throwDuplicateCodeException(aiModel.getApiConfigId(), e);
        }
        if (Boolean.TRUE.equals(aiModel.getDefaultModel())) {
            clearOtherDefaults(aiModel.getModelType(), aiModel.getId());
        }
        return aiModel.getId();
    }

    @Transactional
    public void updateAiModel(Long id, String name, String code, String modelProtocol,
                               String capabilityPresetCode, Integer modelType, String icon,
                               String description, Integer sort, Integer status,
                               String config, Boolean defaultModel, Long apiConfigId,
                               Integer maxConcurrency, Boolean supportVision,
                               Boolean supportReasoning, Integer contextWindow) {
        AiModel model = aiModelMapper.selectById(id);
        if (model == null) throw new BusinessException(404, "AI模型不存在");
        Long nextApiConfigId = apiConfigId != null ? apiConfigId : model.getApiConfigId();
        String nextCode = code != null ? code : model.getCode();
        validateApiConfig(apiConfigId, false);
        validateUniqueCode(id, nextApiConfigId, nextCode);
        if (name != null) model.setName(name);
        if (code != null) model.setCode(code);
        if (modelProtocol != null) model.setModelProtocol(aiModelMetadataResolver.normalizeProtocol(modelProtocol));
        if (capabilityPresetCode != null) model.setCapabilityPresetCode(normalizeCapabilityPresetCode(capabilityPresetCode));
        if (modelType != null) model.setModelType(modelType);
        if (icon != null) model.setIcon(icon);
        if (description != null) model.setDescription(description);
        if (sort != null) model.setSort(sort);
        if (status != null) model.setStatus(status);
        if (config != null) model.setConfig(config);
        if (maxConcurrency != null) model.setMaxConcurrency(maxConcurrency > 0 ? maxConcurrency : 5);
        if (defaultModel != null) model.setDefaultModel(defaultModel);
        if (supportVision != null) model.setSupportVision(supportVision);
        if (supportReasoning != null) model.setSupportReasoning(supportReasoning);
        if (contextWindow != null) model.setContextWindow(contextWindow > 0 ? contextWindow : null);
        if (apiConfigId != null) model.setApiConfigId(apiConfigId);
        normalizeMetadata(model);
        validateRequestProtocol(model);
        validateCapabilityPreset(model);
        try {
            aiModelMapper.updateById(model);
        } catch (DuplicateKeyException e) {
            throwDuplicateCodeException(nextApiConfigId, e);
        }
        if (Boolean.TRUE.equals(model.getDefaultModel())) {
            clearOtherDefaults(model.getModelType(), model.getId());
        }
        chatModelFactory.evict(id);
    }

    @Transactional
    public void deleteAiModel(Long id) {
        aiModelMapper.softDeleteById(id);
        chatModelFactory.evict(id);
    }

    public AiModel getById(Long id) {
        return aiModelMapper.selectById(id);
    }

    public PageResult<AiModel> getPage(String name, String code, Integer modelType, Integer status,
                                        int pageNo, int pageSize) {
        LambdaQueryWrapper<AiModel> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(name != null, AiModel::getName, name)
                .like(code != null, AiModel::getCode, code)
                .eq(modelType != null, AiModel::getModelType, modelType)
                .eq(status != null, AiModel::getStatus, status)
                .orderByAsc(AiModel::getSort)
                .orderByDesc(AiModel::getId);
        return PageResult.of(aiModelMapper.selectPage(new Page<>(pageNo, pageSize), wrapper));
    }

    public List<AiModel> getEnabledList() {
        return aiModelMapper.selectList(new LambdaQueryWrapper<AiModel>().eq(AiModel::getStatus, 1));
    }

    public List<AiModel> getListByType(Integer modelType) {
        return aiModelMapper.selectList(new LambdaQueryWrapper<AiModel>()
                .eq(AiModel::getStatus, 1)
                .eq(AiModel::getModelType, modelType));
    }

    public AiModel getDefaultByType(Integer modelType) {
        return aiModelMapper.selectOne(new LambdaQueryWrapper<AiModel>()
                .eq(AiModel::getDefaultModel, true)
                .eq(AiModel::getModelType, modelType)
                .eq(AiModel::getStatus, 1)
                .orderByAsc(AiModel::getSort)
                .last("LIMIT 1"));
    }

    public AiModelConnectivityRespVO testTextModelConnectivity(Long id) {
        AiModel model = aiModelMapper.selectById(id);
        if (model == null) {
            throw new BusinessException(404, "AI模型不存在");
        }
        if (model.getModelType() == null || model.getModelType() != MODEL_TYPE_TEXT) {
            throw new BusinessException(400, "仅支持文本模型连通性检测");
        }

        long startTime = System.currentTimeMillis();
        try {
            ChatModel chatModel = chatModelFactory.getOrCreate(model);
            ChatResponse response = chatModel.call(new Prompt(CONNECTIVITY_TEST_MESSAGE));

            AiModelConnectivityRespVO respVO = new AiModelConnectivityRespVO();
            respVO.setModelId(model.getId());
            respVO.setModelName(model.getName());
            respVO.setResponseText(StrUtil.blankToDefault(extractResponseText(response), "模型已响应，但未返回文本内容"));
            respVO.setDurationMs(System.currentTimeMillis() - startTime);
            respVO.setTestedAt(LocalDateTime.now());
            return respVO;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException("模型连通性检测失败: "
                    + StrUtil.blankToDefault(e.getMessage(), e.getClass().getSimpleName()));
        }
    }

    private void validateApiConfig(Long apiConfigId, boolean required) {
        if (apiConfigId == null) {
            if (required) {
                throw new BusinessException(400, "请选择 API 配置");
            }
            return;
        }
        if (apiConfigService.getById(apiConfigId) == null) {
            throw new BusinessException(404, "API 配置不存在");
        }
    }

    private void validateUniqueCode(Long currentId, Long apiConfigId, String code) {
        if (StrUtil.isBlank(code)) {
            return;
        }
        LambdaQueryWrapper<AiModel> wrapper = new LambdaQueryWrapper<AiModel>()
                .eq(AiModel::getCode, code);
        if (apiConfigId != null) {
            wrapper.eq(AiModel::getApiConfigId, apiConfigId);
        } else {
            wrapper.isNull(AiModel::getApiConfigId);
        }
        if (currentId != null) {
            wrapper.ne(AiModel::getId, currentId);
        }
        if (aiModelMapper.exists(wrapper)) {
            throw new BusinessException(400,
                    apiConfigId != null ? "同一 API 配置下模型标识已存在" : "未绑定 API 配置的模型标识已存在");
        }
    }

    private void throwDuplicateCodeException(Long apiConfigId, DuplicateKeyException e) {
        throw new BusinessException(400,
                apiConfigId != null ? "同一 API 配置下模型标识已存在" : "未绑定 API 配置的模型标识已存在");
    }

    private void normalizeMetadata(AiModel model) {
        if (model == null) {
            return;
        }
        model.setModelProtocol(aiModelMetadataResolver.normalizeProtocol(model.getModelProtocol()));
        model.setCapabilityPresetCode(normalizeCapabilityPresetCode(model.getCapabilityPresetCode()));
    }

    private String normalizeCapabilityPresetCode(String code) {
        return StrUtil.isBlank(code) ? null : code.trim();
    }

    private void validateCapabilityPreset(AiModel model) {
        if (model == null || StrUtil.isBlank(model.getCapabilityPresetCode())) {
            return;
        }
        var preset = modelPresetService.getPreset(model.getCapabilityPresetCode());
        if (preset == null) {
            throw new BusinessException(400, "模型能力预设不存在: " + model.getCapabilityPresetCode());
        }
        Integer presetModelType = preset.getInt("modelType");
        if (presetModelType != null && !presetModelType.equals(model.getModelType())) {
            throw new BusinessException(400, "模型能力预设与模型类型不匹配");
        }
        String presetProtocol = aiModelMetadataResolver.normalizeProtocol(preset.getStr("modelProtocol"));
        String effectiveProtocol = aiModelMetadataResolver.resolve(model).modelProtocol();
        if (StrUtil.isNotBlank(presetProtocol) && !presetProtocol.equals(effectiveProtocol)) {
            throw new BusinessException(400, "模型能力预设与当前请求协议不匹配");
        }
    }

    private void validateRequestProtocol(AiModel model) {
        if (model == null || model.getModelType() == null || model.getModelType() < 1 || model.getModelType() > 3) {
            return;
        }
        var metadata = aiModelMetadataResolver.resolve(model);
        if (StrUtil.isBlank(metadata.modelProtocol()) || "generic".equals(metadata.modelProtocol())) {
            String capability = switch (model.getModelType()) {
                case 1 -> "文本";
                case 2 -> "图片";
                case 3 -> "视频";
                default -> "当前";
            };
            throw new BusinessException(400, capability
                    + "模型未配置有效请求协议：请在模型中设置覆盖协议，或在 API 配置中设置对应的默认协议");
        }
    }

    private void clearOtherDefaults(Integer modelType, Long excludeId) {
        if (modelType == null || excludeId == null) {
            return;
        }
        aiModelMapper.update(null, new LambdaUpdateWrapper<AiModel>()
                .set(AiModel::getDefaultModel, false)
                .eq(AiModel::getDefaultModel, true)
                .eq(AiModel::getModelType, modelType)
                .ne(AiModel::getId, excludeId));
    }

    private String extractResponseText(ChatResponse response) {
        if (response == null || response.getResult() == null) {
            return "";
        }
        AssistantMessage assistantMessage = response.getResult().getOutput();
        if (assistantMessage == null) {
            return "";
        }
        return StrUtil.trim(assistantMessage.getText());
    }
}
