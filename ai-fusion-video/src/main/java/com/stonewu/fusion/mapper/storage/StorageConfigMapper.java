package com.stonewu.fusion.mapper.storage;

import java.util.List;

import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.storage.StorageConfig;
import org.apache.ibatis.annotations.Mapper;

/**
 * 存储配置 Mapper
 */
@Mapper
public interface StorageConfigMapper extends BaseMapper<StorageConfig> {
    StorageConfig selectForUpdate(Long id);

    List<StorageConfig> lockActiveDefaults();
}
