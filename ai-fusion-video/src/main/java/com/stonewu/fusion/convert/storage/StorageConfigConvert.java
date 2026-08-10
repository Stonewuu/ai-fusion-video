package com.stonewu.fusion.convert.storage;

import com.stonewu.fusion.controller.storage.vo.StorageConfigRespVO;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.StorageConfigOptions;
import org.mapstruct.AfterMapping;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.MappingTarget;
import org.mapstruct.factory.Mappers;

import java.util.List;

/**
 * 存储配置 MapStruct 转换器
 */
@Mapper
public interface StorageConfigConvert {

    StorageConfigConvert INSTANCE = Mappers.getMapper(StorageConfigConvert.class);

    @Mapping(target = "options", ignore = true)
    StorageConfigRespVO convert(StorageConfig config);

    List<StorageConfigRespVO> convertList(List<StorageConfig> list);

    @AfterMapping
    default void maskSecrets(StorageConfig config, @MappingTarget StorageConfigRespVO respVO) {
        respVO.setAccessKey(maskCredential(config.getAccessKey()));
        respVO.setSecretKey(null);
        respVO.setOptions(StorageConfigOptions.toMap(config.getOptions()));
    }

    private static String maskCredential(String value) {
        if (value == null || value.isBlank()) {
            return value;
        }
        if (value.length() <= 8) {
            return "****";
        }
        return value.substring(0, 4) + "****" + value.substring(value.length() - 4);
    }
}
