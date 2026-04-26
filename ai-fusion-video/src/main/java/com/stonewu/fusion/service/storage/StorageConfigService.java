package com.stonewu.fusion.service.storage;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * 存储配置服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class StorageConfigService {

    private final StorageConfigMapper storageConfigMapper;

    @Cacheable(value = "storageConfig", key = "#id")
    public StorageConfig getById(Long id) {
        StorageConfig config = storageConfigMapper.selectById(id);
        if (config == null) throw new BusinessException("存储配置不存在: " + id);
        return config;
    }

    /**
     * 获取默认存储配置，未设置默认时返回 null（将回退到本地存储）
     */
    @Cacheable(value = "storageConfig", key = "'default'", unless = "#result == null")
    public StorageConfig getDefaultConfig() {
        return storageConfigMapper.selectOne(
                new LambdaQueryWrapper<StorageConfig>()
                        .eq(StorageConfig::getIsDefault, true)
                        .eq(StorageConfig::getStatus, 1)
                        .last("LIMIT 1"));
    }

    public List<StorageConfig> getEnabledList() {
        return storageConfigMapper.selectList(
                new LambdaQueryWrapper<StorageConfig>()
                        .eq(StorageConfig::getStatus, 1)
                        .orderByAsc(StorageConfig::getId));
    }

    public PageResult<StorageConfig> page(String name, String type, Integer status, int pageNo, int pageSize) {
        LambdaQueryWrapper<StorageConfig> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(name != null, StorageConfig::getName, name)
                .eq(type != null, StorageConfig::getType, type)
                .eq(status != null, StorageConfig::getStatus, status)
                .orderByDesc(StorageConfig::getIsDefault)
                .orderByDesc(StorageConfig::getId);
        return PageResult.of(storageConfigMapper.selectPage(new Page<>(pageNo, pageSize), wrapper));
    }

    @CacheEvict(value = "storageConfig", allEntries = true)
    @Transactional
    public Long create(StorageConfig config) {
        storageConfigMapper.insert(config);
        // 如果是默认配置，清除其他默认
        if (Boolean.TRUE.equals(config.getIsDefault())) {
            clearOtherDefaults(config.getId());
        }
        return config.getId();
    }

    @CacheEvict(value = "storageConfig", allEntries = true)
    @Transactional
    public void update(StorageConfig config) {
        storageConfigMapper.updateById(config);
        // 如果设为默认，清除其他默认
        if (Boolean.TRUE.equals(config.getIsDefault())) {
            clearOtherDefaults(config.getId());
        }
    }

    @CacheEvict(value = "storageConfig", allEntries = true)
    @Transactional
    public void delete(Long id) {
        storageConfigMapper.deleteById(id);
    }

    /**
     * 设置某个配置为默认
     */
    @CacheEvict(value = "storageConfig", allEntries = true)
    @Transactional
    public void setDefault(Long id) {
        // 先清除所有默认标记
        storageConfigMapper.update(null,
                new LambdaUpdateWrapper<StorageConfig>()
                        .set(StorageConfig::getIsDefault, false)
                        .eq(StorageConfig::getIsDefault, true));
        // 设置指定配置为默认
        StorageConfig config = getById(id);
        config.setIsDefault(true);
        storageConfigMapper.updateById(config);
    }

    /**
     * 清除指定 ID 之外的其他默认标记
     */
    private void clearOtherDefaults(Long excludeId) {
        storageConfigMapper.update(null,
                new LambdaUpdateWrapper<StorageConfig>()
                        .set(StorageConfig::getIsDefault, false)
                        .eq(StorageConfig::getIsDefault, true)
                        .ne(StorageConfig::getId, excludeId));
    }
}
