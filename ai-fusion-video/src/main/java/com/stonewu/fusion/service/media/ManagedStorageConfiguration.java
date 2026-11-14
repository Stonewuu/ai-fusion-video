package com.stonewu.fusion.service.media;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import com.stonewu.fusion.service.storage.StorageTypes;
import com.stonewu.fusion.service.storage.BuiltInLocalStorage;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/** 提交前检查与媒体登记时的锁定检查共用同一份存储契约。 */
@Component
@RequiredArgsConstructor
public class ManagedStorageConfiguration {
    private final StorageConfigMapper configurations;
    private final S3StorageConfigResolver s3;
    private final BuiltInLocalStorage builtIn;

    public StorageConfig requireDefault() {
        return validate(configurations.selectPage(new Page<>(1, 2, false),
                new LambdaQueryWrapper<StorageConfig>().eq(StorageConfig::getIsDefault, true)
                        .eq(StorageConfig::getStatus, 1).orderByAsc(StorageConfig::getId)).getRecords());
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public StorageConfig lockDefault() {
        return validate(configurations.lockActiveDefaults());
    }

    private StorageConfig validate(List<StorageConfig> defaults) {
        if (defaults.isEmpty() && !configurations.exists(new LambdaQueryWrapper<>())) {
            return builtIn.configuration();
        }
        if (defaults.size() != 1) {
            throw MediaException.storageConfiguration("请设置且只设置一个启用的默认存储配置");
        }
        StorageConfig config = defaults.getFirst();
        String backend = StorageTypes.normalizeType(config.getType());
        if (StorageTypes.LOCAL.equals(backend)) {
            if (config.getBasePath() == null || config.getBasePath().isBlank()) {
                throw MediaException.storageConfiguration("请填写默认本地存储的根目录");
            }
        } else if (StorageTypes.S3.equals(backend)) {
            try {
                s3.resolve(config);
            } catch (BusinessException | IllegalArgumentException invalid) {
                throw MediaException.storageConfiguration("默认对象存储配置不完整，请检查存储设置");
            }
        } else {
            throw MediaException.storageConfiguration("默认存储类型不支持保存媒体，请选择本地存储或对象存储");
        }
        return config;
    }
}
