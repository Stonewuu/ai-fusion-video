package com.stonewu.fusion.controller.ai;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.ai.vo.*;
import com.stonewu.fusion.convert.ai.AiModelConvert;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.common.CommonResult.success;

@Tag(name = "AI模型管理")
@RestController
@RequestMapping("/api/ai/model")
@RequiredArgsConstructor
public class AiModelController {

    private final AiModelService aiModelService;
    private final ModelPresetService modelPresetService;

    @PostMapping("/create")
    @Operation(summary = "创建AI模型")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Long> create(@Valid @RequestBody AiModelCreateReqVO reqVO) {
        AiModel model = AiModel.builder()
                .name(reqVO.getName())
                .code(reqVO.getCode())
                .modelProtocol(reqVO.getModelProtocol())
                .capabilityPresetCode(reqVO.getCapabilityPresetCode())
                .modelType(reqVO.getModelType())
                .icon(reqVO.getIcon()).description(reqVO.getDescription())
                .sort(reqVO.getSort() != null ? reqVO.getSort() : 0)
                .config(reqVO.getConfig())
            .maxConcurrency(reqVO.getMaxConcurrency() != null && reqVO.getMaxConcurrency() > 0
                ? reqVO.getMaxConcurrency() : 5)
                .defaultModel(reqVO.getDefaultModel() != null ? reqVO.getDefaultModel() : false)
            .supportVision(Boolean.TRUE.equals(reqVO.getSupportVision()))
            .multimodalInputTypes(reqVO.getMultimodalInputTypes())
            .multimodalInputTransports(reqVO.getMultimodalInputTransports())
            .supportReasoning(Boolean.TRUE.equals(reqVO.getSupportReasoning()))
            .contextWindow(reqVO.getContextWindow() != null && reqVO.getContextWindow() > 0
                ? reqVO.getContextWindow() : null)
                .apiConfigId(reqVO.getApiConfigId())
                .build();
        return success(aiModelService.createAiModel(model));
    }

    @PutMapping("/update")
    @Operation(summary = "更新AI模型")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> update(@Valid @RequestBody AiModelUpdateReqVO reqVO) {
        aiModelService.updateAiModel(reqVO.getId(), reqVO.getName(), reqVO.getCode(),
                reqVO.getModelProtocol(), reqVO.getCapabilityPresetCode(), reqVO.getModelType(),
                reqVO.getIcon(), reqVO.getDescription(),
                reqVO.getSort(), reqVO.getStatus(), reqVO.getConfig(), reqVO.getDefaultModel(),
                reqVO.getApiConfigId(), reqVO.getMaxConcurrency(), reqVO.getMultimodalInputTypes(),
                reqVO.getMultimodalInputTransports(),
                reqVO.getSupportReasoning(), reqVO.getContextWindow());
        return success(true);
    }

    @DeleteMapping("/delete")
    @Operation(summary = "删除AI模型")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> delete(@RequestParam("id") Long id) {
        aiModelService.deleteAiModel(id);
        return success(true);
    }

    @GetMapping("/get")
    @Operation(summary = "获取AI模型详情")
    @Parameter(name = "id", description = "模型ID", required = true)
    public CommonResult<AiModelRespVO> get(@RequestParam("id") Long id) {
        AiModel model = aiModelService.getById(id);
        return success(model == null ? null : AiModelConvert.INSTANCE.convert(model));
    }

    @GetMapping("/page")
    @Operation(summary = "AI模型分页列表")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<PageResult<AiModelRespVO>> page(@Valid AiModelPageReqVO reqVO) {
        return success(aiModelService.getPage(reqVO.getName(), reqVO.getCode(),
                reqVO.getModelType(), reqVO.getStatus(), reqVO.getPageNo(), reqVO.getPageSize())
                .map(AiModelConvert.INSTANCE::convert));
    }

    @GetMapping("/list")
    @Operation(summary = "获取启用的AI模型列表")
    public CommonResult<List<AiModelRespVO>> list() {
        return success(AiModelConvert.INSTANCE.convertList(aiModelService.getEnabledList()));
    }

    @GetMapping("/list-by-type")
    @Operation(summary = "按类型获取AI模型列表")
    @Parameter(name = "type", description = "模型类型", required = true)
    public CommonResult<List<AiModelRespVO>> listByType(@RequestParam("type") Integer type) {
        return success(AiModelConvert.INSTANCE.convertList(aiModelService.getListByType(type)));
    }

    @GetMapping("/presets")
    @Operation(summary = "获取模型能力预设列表")
    @Parameter(name = "type", description = "模型类型（可选）")
    public CommonResult<List<JSONObject>> presets(@RequestParam(value = "type", required = false) Integer type) {
        if (type != null) {
            return success(modelPresetService.getPresetsByType(type));
        }
        return success(modelPresetService.getAllPresets());
    }

    @GetMapping("/preset-config")
    @Operation(summary = "获取模型能力预设配置")
    @Parameter(name = "code", description = "能力预设代码", required = true)
    public CommonResult<String> presetConfig(@RequestParam("code") String code) {
        return success(modelPresetService.getPresetConfig(code));
    }

    @PostMapping("/test-text-connectivity")
    @Operation(summary = "检测文本模型连通性")
    @Parameter(name = "id", description = "模型ID", required = true)
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<AiModelConnectivityRespVO> testTextConnectivity(@RequestParam("id") Long id) {
        return success(aiModelService.testTextModelConnectivity(id));
    }
}

