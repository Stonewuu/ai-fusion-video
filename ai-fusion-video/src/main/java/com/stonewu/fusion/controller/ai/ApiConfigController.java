package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.ai.vo.ApiConfigPageReqVO;
import com.stonewu.fusion.controller.ai.vo.ApiConfigRespVO;
import com.stonewu.fusion.controller.ai.vo.ApiConfigSaveReqVO;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.convert.ai.ApiConfigConvert;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.common.CommonResult.success;

@Tag(name = "API配置管理")
@RestController
@RequestMapping("/ai/api-config")
@RequiredArgsConstructor
public class ApiConfigController {

    private final ApiConfigService apiConfigService;
    private final AiProviderService aiProviderService;

    @PostMapping("/create")
    @Operation(summary = "创建API配置")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Long> create(@Valid @RequestBody ApiConfigSaveReqVO reqVO) {
        ApiConfig config = ApiConfig.builder()
                .name(reqVO.getName()).platform(reqVO.getPlatform())
                .apiUrl(reqVO.getApiUrl())
                .autoAppendV1Path(reqVO.getAutoAppendV1Path() != null ? reqVO.getAutoAppendV1Path() : true)
                .proxyType(reqVO.getProxyType())
                .proxyHost(reqVO.getProxyHost())
                .proxyPort(reqVO.getProxyPort())
                .proxyUsername(reqVO.getProxyUsername())
                .proxyPassword(reqVO.getProxyPassword())
                .apiKey(reqVO.getApiKey()).appId(reqVO.getAppId()).appSecret(reqVO.getAppSecret())
                .modelId(reqVO.getModelId()).status(reqVO.getStatus() != null ? reqVO.getStatus() : 1)
                .remark(reqVO.getRemark())
                .build();
        return success(apiConfigService.createApiConfig(config));
    }

    @PutMapping("/update")
    @Operation(summary = "更新API配置")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> update(@Valid @RequestBody ApiConfigSaveReqVO reqVO) {
        apiConfigService.updateApiConfig(reqVO.getId(), reqVO.getName(), reqVO.getPlatform(),
            reqVO.getApiUrl(), reqVO.getAutoAppendV1Path(), reqVO.getProxyType(),
            reqVO.getProxyHost(), reqVO.getProxyPort(), reqVO.getProxyUsername(),
            reqVO.getProxyPassword(), reqVO.getApiKey(), reqVO.getAppId(), reqVO.getAppSecret(),
            reqVO.getModelId(), reqVO.getStatus(), reqVO.getRemark());
        return success(true);
    }

    @DeleteMapping("/delete")
    @Operation(summary = "删除API配置")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> delete(@RequestParam("id") Long id) {
        apiConfigService.deleteApiConfig(id);
        return success(true);
    }

    @GetMapping("/get")
    @Operation(summary = "获取API配置详情")
    @Parameter(name = "id", description = "配置ID", required = true)
    public CommonResult<ApiConfigRespVO> get(@RequestParam("id") Long id) {
        ApiConfig config = apiConfigService.getById(id);
        return success(config == null ? null : ApiConfigConvert.INSTANCE.convert(config));
    }

    @GetMapping("/page")
    @Operation(summary = "API配置分页列表")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<PageResult<ApiConfigRespVO>> page(@Valid ApiConfigPageReqVO reqVO) {
        return success(apiConfigService.getPage(reqVO.getName(), reqVO.getPlatform(),
                reqVO.getStatus(), reqVO.getPageNo(), reqVO.getPageSize())
                .map(ApiConfigConvert.INSTANCE::convert));
    }

    @GetMapping("/list")
    @Operation(summary = "获取启用的API配置列表")
    public CommonResult<List<ApiConfigRespVO>> list() {
        return success(ApiConfigConvert.INSTANCE.convertList(apiConfigService.getEnabledList()));
    }

    @GetMapping("/remote-models")
    @Operation(summary = "获取远程可用模型列表")
    @Parameter(name = "id", description = "API配置ID", required = true)
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<List<RemoteModelVO>> remoteModels(@RequestParam("id") Long id) {
        return success(aiProviderService.listRemoteModels(id));
    }
}
