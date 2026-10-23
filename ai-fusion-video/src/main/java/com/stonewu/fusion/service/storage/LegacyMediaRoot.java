package com.stonewu.fusion.service.storage;

import org.springframework.beans.factory.annotation.Value;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import org.springframework.stereotype.Component;

import java.nio.file.Path;

/**
 * 历史 /media 路由在启动时固定一次根目录；导入与静态资源使用同一个实际定位。
 */
@Component
public class LegacyMediaRoot {
    private final StorageConfigMapper configurations;
    private final String configuredRoot;
    private Path root;

    public LegacyMediaRoot(StorageConfigMapper configurations,
                           @Value("${app.storage.local-base-path:./data/media}") String configuredRoot) {
        this.configurations = configurations;
        this.configuredRoot = configuredRoot;
    }

    public synchronized Path path() {
        if (root != null) {
            return root;
        }
        var defaults = configurations.selectList(new LambdaQueryWrapper<StorageConfig>()
                .select(StorageConfig::getType, StorageConfig::getBasePath)
                .eq(StorageConfig::getIsDefault, true).eq(StorageConfig::getStatus, 1)
                .orderByAsc(StorageConfig::getId));
        if (defaults.size() > 1) {
            throw new IllegalStateException("默认存储配置不唯一，无法确定历史媒体目录");
        }
        String directory = configuredRoot;
        if (!defaults.isEmpty() && "local".equals(defaults.getFirst().getType())) {
            String base = defaults.getFirst().getBasePath();
            // 这是历史 /media 协议已有的路径规则，不在多目录之间搜索或吞掉数据库错误。
            if (base != null && !base.isBlank()) {
                directory = base;
            }
        }
        root = Path.of(directory).toAbsolutePath().normalize();
        return root;
    }
}
