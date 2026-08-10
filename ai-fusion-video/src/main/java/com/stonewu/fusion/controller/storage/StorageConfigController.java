package com.stonewu.fusion.controller.storage;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.storage.vo.StorageConfigRespVO;
import com.stonewu.fusion.controller.storage.vo.StorageConfigSaveReqVO;
import com.stonewu.fusion.controller.storage.vo.StorageConfigTestRespVO;
import com.stonewu.fusion.convert.storage.StorageConfigConvert;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.StorageConfigOptions;
import com.stonewu.fusion.service.storage.StorageConnectionTestResult;
import com.stonewu.fusion.service.storage.StorageConnectionTestService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.common.CommonResult.success;

/**
 * 存储配置管理
 */
@Tag(name = "存储配置管理")
@RestController
@RequestMapping("/api/storage/config")
@RequiredArgsConstructor
public class StorageConfigController {

    private final StorageConfigService storageConfigService;
    private final StorageConnectionTestService storageConnectionTestService;

    @PostMapping("/create")
    @Operation(summary = "创建存储配置")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Long> create(@Valid @RequestBody StorageConfigSaveReqVO reqVO) {
        StorageConfig config = buildConfig(reqVO, null);
        return success(storageConfigService.create(config));
    }

    @PutMapping("/update")
    @Operation(summary = "更新存储配置")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> update(@Valid @RequestBody StorageConfigSaveReqVO reqVO) {
        StorageConfig config = buildConfig(reqVO, storageConfigService.getById(reqVO.getId()));
        storageConfigService.update(config);
        return success(true);
    }

    @DeleteMapping("/delete")
    @Operation(summary = "删除存储配置")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> delete(@RequestParam("id") Long id) {
        storageConfigService.delete(id);
        return success(true);
    }

    @GetMapping("/get")
    @Operation(summary = "获取存储配置详情")
    @Parameter(name = "id", description = "配置ID", required = true)
    public CommonResult<StorageConfigRespVO> get(@RequestParam("id") Long id) {
        StorageConfig config = storageConfigService.getById(id);
        return success(StorageConfigConvert.INSTANCE.convert(config));
    }

    @GetMapping("/list")
    @Operation(summary = "获取启用的存储配置列表")
    public CommonResult<List<StorageConfigRespVO>> list() {
        return success(StorageConfigConvert.INSTANCE.convertList(storageConfigService.getEnabledList()));
    }

    @PutMapping("/set-default")
    @Operation(summary = "设置默认存储配置")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> setDefault(@RequestParam("id") Long id) {
        storageConfigService.setDefault(id);
        return success(true);
    }

    @PostMapping("/test")
    @Operation(summary = "测试存储配置")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<StorageConfigTestRespVO> test(@Valid @RequestBody StorageConfigSaveReqVO reqVO) {
        StorageConfig existing = reqVO.getId() != null ? storageConfigService.getById(reqVO.getId()) : null;
        StorageConfig config = buildConfig(reqVO, existing);
        StorageConnectionTestResult result = storageConnectionTestService.test(config);
        return success(new StorageConfigTestRespVO(result.success(), result.message(), result.publicUrl()));
    }

    private StorageConfig buildConfig(StorageConfigSaveReqVO reqVO, StorageConfig existing) {
        StorageConfig.StorageConfigBuilder builder = StorageConfig.builder()
                .id(existing != null ? existing.getId() : reqVO.getId())
                .name(valueOrExisting(reqVO.getName(), existing != null ? existing.getName() : null))
                .type(valueOrExisting(reqVO.getType(), existing != null ? existing.getType() : null))
                .provider(valueOrExisting(reqVO.getProvider(), existing != null ? existing.getProvider() : null))
                .endpoint(valueOrExisting(reqVO.getEndpoint(), existing != null ? existing.getEndpoint() : null))
                .bucketName(valueOrExisting(reqVO.getBucketName(), existing != null ? existing.getBucketName() : null))
                .region(valueOrExisting(reqVO.getRegion(), existing != null ? existing.getRegion() : null))
                .basePath(valueOrExisting(reqVO.getBasePath(), existing != null ? existing.getBasePath() : null))
                .customDomain(valueOrExisting(reqVO.getCustomDomain(), existing != null ? existing.getCustomDomain() : null))
                .isDefault(reqVO.getIsDefault() != null ? reqVO.getIsDefault()
                        : existing != null ? existing.getIsDefault() : false)
                .status(reqVO.getStatus() != null ? reqVO.getStatus()
                        : existing != null ? existing.getStatus() : 1)
                .remark(valueOrExisting(reqVO.getRemark(), existing != null ? existing.getRemark() : null))
                .options(reqVO.getOptions() != null
                        ? StorageConfigOptions.toJson(reqVO.getOptions())
                        : existing != null ? existing.getOptions() : null);

        String accessKey = reqVO.getAccessKey();
        if (existing != null && (StrUtil.isBlank(accessKey) || isMaskedCredential(accessKey))) {
            accessKey = existing.getAccessKey();
        }

        String secretKey = reqVO.getSecretKey();
        if (existing != null && StrUtil.isBlank(secretKey)) {
            secretKey = existing.getSecretKey();
        }

        return builder
                .accessKey(accessKey)
                .secretKey(secretKey)
                .build();
    }

    private String valueOrExisting(String value, String existing) {
        return value != null ? value : existing;
    }

    private boolean isMaskedCredential(String value) {
        return value != null && value.contains("****");
    }
}
