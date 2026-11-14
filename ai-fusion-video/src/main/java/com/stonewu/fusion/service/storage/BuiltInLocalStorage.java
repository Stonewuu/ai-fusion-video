package com.stonewu.fusion.service.storage;

import com.stonewu.fusion.entity.storage.StorageConfig;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * 零配置时的内置本地后端，不写入存储配置表。
 * 媒体对象以 0 标识此后端；用户配置使用数据库自增的正数 ID。
 */
@Component
public class BuiltInLocalStorage {
    public static final long CONFIG_ID = 0L;
    private final String basePath;

    public BuiltInLocalStorage(@Value("${app.storage.local-base-path}") String basePath) {
        this.basePath = basePath;
    }

    public StorageConfig configuration() {
        return StorageConfig.builder().id(CONFIG_ID).name("本地存储").type(StorageTypes.LOCAL)
                .basePath(basePath).status(1).isDefault(true).build();
    }
}
