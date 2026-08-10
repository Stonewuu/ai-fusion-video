package com.stonewu.fusion.service.ai.comfyui;

import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiWorkflowMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiWorkflowVersionMapper;
import com.stonewu.fusion.service.ai.ApiConfigService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Locale;

/** CRUD, immutable versioning and publication rules for ComfyUI workflows. */
@Service
@RequiredArgsConstructor
public class ComfyUiWorkflowService {

    public static final String PLATFORM = "comfyui";
    private static final int MODEL_TYPE_IMAGE = 2;
    private static final int MODEL_TYPE_VIDEO = 3;

    private final ComfyUiWorkflowMapper workflowMapper;
    private final ComfyUiWorkflowVersionMapper versionMapper;
    private final AiModelMapper aiModelMapper;
    private final ApiConfigService apiConfigService;
    private final ComfyUiWorkflowDocumentService documentService;

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public Long createWorkflow(Long apiConfigId,
                               String name,
                               String code,
                               Integer modelType,
                               String description,
                               Integer status) {
        requireComfyUiApiConfig(apiConfigId);
        requireNameAndCode(name, code);
        requireModelType(modelType);
        validateUniqueCode(null, apiConfigId, code);
        ComfyUiWorkflow workflow = ComfyUiWorkflow.builder()
                .apiConfigId(apiConfigId)
                .name(name.trim())
                .code(normalizeCode(code))
                .modelType(modelType)
                .description(StrUtil.trim(description))
                .status(status != null ? status : 1)
                .build();
        try {
            workflowMapper.insert(workflow);
        } catch (DuplicateKeyException e) {
            throw new BusinessException(400, "同一 ComfyUI 配置下工作流标识已存在");
        }
        return workflow.getId();
    }

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void updateWorkflow(Long id,
                               Long apiConfigId,
                               String name,
                               String code,
                               Integer modelType,
                               String description,
                               Integer status) {
        ComfyUiWorkflow workflow = requireWorkflow(id);
        if (apiConfigId != null && !apiConfigId.equals(workflow.getApiConfigId())) {
            throw new BusinessException(400, "ComfyUI 工作流创建后不能更换所属供应商");
        }
        Long nextApiConfigId = workflow.getApiConfigId();
        String nextCode = code != null ? code : workflow.getCode();
        Integer nextModelType = modelType != null ? modelType : workflow.getModelType();
        requireComfyUiApiConfig(nextApiConfigId);
        requireModelType(nextModelType);
        if (workflow.getActiveVersionId() != null
                && !nextModelType.equals(workflow.getModelType())) {
            throw new BusinessException(400, "已发布工作流不能更换模型类型");
        }
        validateUniqueCode(id, nextApiConfigId, nextCode);
        if (name != null) {
            if (name.isBlank()) throw new BusinessException(400, "工作流名称不能为空");
            workflow.setName(name.trim());
        }
        if (code != null) {
            if (code.isBlank()) throw new BusinessException(400, "工作流标识不能为空");
            workflow.setCode(normalizeCode(code));
        }
        if (modelType != null) workflow.setModelType(modelType);
        if (description != null) workflow.setDescription(StrUtil.trim(description));
        if (status != null) workflow.setStatus(status);
        workflowMapper.updateById(workflow);
    }

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void deleteWorkflow(Long id) {
        ComfyUiWorkflow workflow = requireWorkflow(id);
        boolean referenced = aiModelMapper.exists(new LambdaQueryWrapper<AiModel>()
                .eq(AiModel::getComfyuiWorkflowId, workflow.getId()));
        if (referenced) {
            throw new BusinessException(400, "工作流已被模型引用，请先解除模型绑定");
        }
        workflowMapper.softDeleteById(id);
    }

    @Cacheable(value = "comfyUiWorkflow", key = "#id", unless = "#result == null")
    public ComfyUiWorkflow getById(Long id) {
        return id == null ? null : workflowMapper.selectById(id);
    }

    public PageResult<ComfyUiWorkflow> getPage(String name,
                                               String code,
                                               Long apiConfigId,
                                               Integer modelType,
                                               Integer status,
                                               int pageNo,
                                               int pageSize) {
        LambdaQueryWrapper<ComfyUiWorkflow> query = new LambdaQueryWrapper<>();
        query.like(StrUtil.isNotBlank(name), ComfyUiWorkflow::getName, name)
                .like(StrUtil.isNotBlank(code), ComfyUiWorkflow::getCode, code)
                .eq(apiConfigId != null, ComfyUiWorkflow::getApiConfigId, apiConfigId)
                .eq(modelType != null, ComfyUiWorkflow::getModelType, modelType)
                .eq(status != null, ComfyUiWorkflow::getStatus, status)
                .orderByDesc(ComfyUiWorkflow::getId);
        return PageResult.of(workflowMapper.selectPage(new Page<>(pageNo, pageSize), query));
    }

    @Cacheable(value = "comfyUiWorkflow", key = "'list:' + #apiConfigId + ':' + #modelType")
    public List<ComfyUiWorkflow> getEnabledList(Long apiConfigId, Integer modelType) {
        return workflowMapper.selectList(new LambdaQueryWrapper<ComfyUiWorkflow>()
                .eq(ComfyUiWorkflow::getStatus, 1)
                .eq(apiConfigId != null, ComfyUiWorkflow::getApiConfigId, apiConfigId)
                .eq(modelType != null, ComfyUiWorkflow::getModelType, modelType)
                .isNotNull(ComfyUiWorkflow::getActiveVersionId)
                .orderByAsc(ComfyUiWorkflow::getName));
    }

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public Long createVersion(Long workflowId,
                              String uiWorkflowJson,
                              String apiWorkflowJson,
                              String inputBindingsJson,
                              String outputBindingsJson) {
        ComfyUiWorkflow workflow = lockWorkflow(workflowId);
        ComfyUiWorkflowDefinition definition = documentService.normalize(
                workflow.getModelType(), uiWorkflowJson, apiWorkflowJson,
                inputBindingsJson, outputBindingsJson);
        ComfyUiWorkflowVersion latest = versionMapper.selectOne(
                new LambdaQueryWrapper<ComfyUiWorkflowVersion>()
                        .eq(ComfyUiWorkflowVersion::getWorkflowId, workflowId)
                        .orderByDesc(ComfyUiWorkflowVersion::getVersionNo)
                        .last("LIMIT 1"));
        int nextVersion = latest == null ? 1 : latest.getVersionNo() + 1;
        ComfyUiWorkflowVersion version = buildVersion(workflowId, nextVersion, definition);
        versionMapper.insert(version);
        return version.getId();
    }

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void updateDraftVersion(Long versionId,
                                   String uiWorkflowJson,
                                   String apiWorkflowJson,
                                   String inputBindingsJson,
                                   String outputBindingsJson) {
        ComfyUiWorkflowVersion version = requireVersion(versionId);
        if (Boolean.TRUE.equals(version.getPublished())) {
            throw new BusinessException(400, "已发布版本不可修改，请创建新版本");
        }
        ComfyUiWorkflow workflow = requireWorkflow(version.getWorkflowId());
        ComfyUiWorkflowDefinition definition = documentService.normalize(
                workflow.getModelType(), uiWorkflowJson, apiWorkflowJson,
                inputBindingsJson, outputBindingsJson);
        applyDefinition(version, definition);
        version.setValidationStatus(ComfyUiWorkflowVersion.VALIDATION_PENDING);
        version.setValidationMessage(null);
        version.setTestStatus(ComfyUiWorkflowVersion.TEST_PENDING);
        version.setTestMessage(null);
        version.setLastTestTime(null);
        versionMapper.updateById(version);
    }

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void deleteDraftVersion(Long versionId) {
        ComfyUiWorkflowVersion version = requireVersion(versionId);
        ComfyUiWorkflow workflow = requireWorkflow(version.getWorkflowId());
        if (Boolean.TRUE.equals(version.getPublished())
                || versionId.equals(workflow.getActiveVersionId())) {
            throw new BusinessException(400, "已发布版本不可删除");
        }
        versionMapper.deleteById(versionId);
    }

    @Cacheable(value = "comfyUiWorkflowVersion", key = "#id", unless = "#result == null")
    public ComfyUiWorkflowVersion getVersion(Long id) {
        return id == null ? null : versionMapper.selectById(id);
    }

    public List<ComfyUiWorkflowVersion> getVersions(Long workflowId) {
        requireWorkflow(workflowId);
        return versionMapper.selectList(new LambdaQueryWrapper<ComfyUiWorkflowVersion>()
                .eq(ComfyUiWorkflowVersion::getWorkflowId, workflowId)
                .orderByDesc(ComfyUiWorkflowVersion::getVersionNo));
    }

    @Cacheable(value = "comfyUiWorkflowVersion", key = "'active:' + #workflowId", unless = "#result == null")
    public ComfyUiWorkflowVersion getActiveVersion(Long workflowId) {
        ComfyUiWorkflow workflow = requireWorkflow(workflowId);
        if (workflow.getActiveVersionId() == null) {
            return null;
        }
        return requireVersion(workflow.getActiveVersionId());
    }

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void recordValidation(Long versionId, boolean valid, String message) {
        ComfyUiWorkflowVersion version = requireVersion(versionId);
        version.setValidationStatus(valid
                ? ComfyUiWorkflowVersion.VALIDATION_VALID
                : ComfyUiWorkflowVersion.VALIDATION_INVALID);
        version.setValidationMessage(StrUtil.trim(message));
        versionMapper.updateById(version);
    }

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void recordExecutionTest(Long versionId, boolean passed, String message) {
        ComfyUiWorkflowVersion version = requireVersion(versionId);
        version.setTestStatus(passed
                ? ComfyUiWorkflowVersion.TEST_PASSED
                : ComfyUiWorkflowVersion.TEST_FAILED);
        version.setTestMessage(StrUtil.trim(message));
        version.setLastTestTime(LocalDateTime.now());
        versionMapper.updateById(version);
    }

    @Transactional
    @CacheEvict(value = {"comfyUiWorkflow", "comfyUiWorkflowVersion"}, allEntries = true)
    public void publishVersion(Long workflowId, Long versionId) {
        ComfyUiWorkflow workflow = lockWorkflow(workflowId);
        ComfyUiWorkflowVersion version = requireVersion(versionId);
        if (!workflowId.equals(version.getWorkflowId())) {
            throw new BusinessException(400, "工作流版本不属于指定工作流");
        }
        if (!Integer.valueOf(ComfyUiWorkflowVersion.VALIDATION_VALID)
                .equals(version.getValidationStatus())) {
            throw new BusinessException(400, "工作流版本必须通过目标 ComfyUI 在线验证后才能发布");
        }
        if (!Integer.valueOf(ComfyUiWorkflowVersion.TEST_PASSED).equals(version.getTestStatus())) {
            throw new BusinessException(400, "工作流版本必须在目标 ComfyUI 试运行成功后才能发布");
        }
        version.setPublished(true);
        versionMapper.updateById(version);
        workflow.setActiveVersionId(versionId);
        workflow.setStatus(1);
        workflowMapper.updateById(workflow);
    }

    public void validateModelBinding(AiModel model, ApiConfig apiConfig) {
        if (model == null || apiConfig == null || !PLATFORM.equals(normalize(apiConfig.getPlatform()))) {
            return;
        }
        if (model.getModelType() == null
                || (model.getModelType() != MODEL_TYPE_IMAGE && model.getModelType() != MODEL_TYPE_VIDEO)) {
            throw new BusinessException(400, "ComfyUI 仅支持图片或视频模型");
        }
        if (model.getComfyuiWorkflowId() == null) {
            throw new BusinessException(400, "ComfyUI 模型必须选择已发布工作流");
        }
        String effectiveProtocol = StrUtil.isNotBlank(model.getModelProtocol())
                ? model.getModelProtocol()
                : (model.getModelType() == MODEL_TYPE_IMAGE
                ? apiConfig.getImageProtocol() : apiConfig.getVideoProtocol());
        if (!PLATFORM.equals(normalize(effectiveProtocol))) {
            throw new BusinessException(400, "ComfyUI 模型请求协议必须为 comfyui");
        }
        ComfyUiWorkflow workflow = requireWorkflow(model.getComfyuiWorkflowId());
        if (!apiConfig.getId().equals(workflow.getApiConfigId())) {
            throw new BusinessException(400, "模型与工作流必须使用同一个 ComfyUI API 配置");
        }
        if (!model.getModelType().equals(workflow.getModelType())) {
            throw new BusinessException(400, "模型类型与 ComfyUI 工作流类型不一致");
        }
        if (!Integer.valueOf(1).equals(workflow.getStatus()) || workflow.getActiveVersionId() == null) {
            throw new BusinessException(400, "请选择已启用且已发布的 ComfyUI 工作流");
        }
    }

    public ComfyUiWorkflow requireWorkflow(Long id) {
        ComfyUiWorkflow workflow = getById(id);
        if (workflow == null) {
            throw new BusinessException(404, "ComfyUI 工作流不存在");
        }
        return workflow;
    }

    public ComfyUiWorkflowVersion requireVersion(Long id) {
        ComfyUiWorkflowVersion version = getVersion(id);
        if (version == null) {
            throw new BusinessException(404, "ComfyUI 工作流版本不存在");
        }
        return version;
    }

    private ComfyUiWorkflow lockWorkflow(Long id) {
        ComfyUiWorkflow workflow = workflowMapper.selectOne(
                new LambdaQueryWrapper<ComfyUiWorkflow>()
                        .eq(ComfyUiWorkflow::getId, id)
                        .last("FOR UPDATE"));
        if (workflow == null) {
            throw new BusinessException(404, "ComfyUI 工作流不存在");
        }
        return workflow;
    }

    private ComfyUiWorkflowVersion buildVersion(Long workflowId,
                                                int versionNo,
                                                ComfyUiWorkflowDefinition definition) {
        ComfyUiWorkflowVersion version = ComfyUiWorkflowVersion.builder()
                .workflowId(workflowId)
                .versionNo(versionNo)
                .build();
        applyDefinition(version, definition);
        return version;
    }

    private void applyDefinition(ComfyUiWorkflowVersion version,
                                 ComfyUiWorkflowDefinition definition) {
        version.setUiWorkflowJson(definition.uiWorkflowJson());
        version.setApiWorkflowJson(definition.apiWorkflowJson());
        version.setInputBindingsJson(definition.inputBindingsJson());
        version.setOutputBindingsJson(definition.outputBindingsJson());
        version.setRequiredNodesJson(definition.requiredNodesJson());
        version.setWorkflowHash(definition.workflowHash());
    }

    private void requireComfyUiApiConfig(Long apiConfigId) {
        if (apiConfigId == null) {
            throw new BusinessException(400, "请选择 ComfyUI API 配置");
        }
        ApiConfig apiConfig = apiConfigService.getById(apiConfigId);
        if (apiConfig == null) {
            throw new BusinessException(404, "API 配置不存在");
        }
        if (!PLATFORM.equals(normalize(apiConfig.getPlatform()))) {
            throw new BusinessException(400, "工作流只能关联 platform=comfyui 的 API 配置");
        }
    }

    private void requireNameAndCode(String name, String code) {
        if (StrUtil.isBlank(name)) throw new BusinessException(400, "工作流名称不能为空");
        if (StrUtil.isBlank(code)) throw new BusinessException(400, "工作流标识不能为空");
    }

    private void requireModelType(Integer modelType) {
        if (modelType == null || (modelType != MODEL_TYPE_IMAGE && modelType != MODEL_TYPE_VIDEO)) {
            throw new BusinessException(400, "ComfyUI 工作流类型只能是图片或视频");
        }
    }

    private void validateUniqueCode(Long currentId, Long apiConfigId, String code) {
        LambdaQueryWrapper<ComfyUiWorkflow> query = new LambdaQueryWrapper<ComfyUiWorkflow>()
                .eq(ComfyUiWorkflow::getApiConfigId, apiConfigId)
                .eq(ComfyUiWorkflow::getCode, normalizeCode(code));
        if (currentId != null) query.ne(ComfyUiWorkflow::getId, currentId);
        if (workflowMapper.exists(query)) {
            throw new BusinessException(400, "同一 ComfyUI 配置下工作流标识已存在");
        }
    }

    private String normalizeCode(String value) {
        return value.trim().toLowerCase(Locale.ROOT).replace(' ', '-');
    }

    private String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }
}
