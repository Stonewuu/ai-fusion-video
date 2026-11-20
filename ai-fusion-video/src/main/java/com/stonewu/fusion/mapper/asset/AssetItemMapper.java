package com.stonewu.fusion.mapper.asset;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.asset.AssetItem;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;
import com.baomidou.mybatisplus.core.metadata.IPage;

@Mapper
public interface AssetItemMapper extends BaseMapper<AssetItem> {
    List<Map<String, Object>> summaries(IPage<?> page, @Param("assetId") long assetId);

    List<Map<String, Object>> detailsByAssetIds(@Param("assetIds") List<Long> assetIds);

    List<Map<String, Object>> summariesByAssetIds(@Param("assetIds") List<Long> assetIds,
                                                @Param("offset") long offset, @Param("end") long end);
    List<Long> lockNamed(@Param("assetId") long assetId, @Param("name") String name);

    List<Long> lockChildren(@Param("assetId") long assetId);

    long parentAssetId(@Param("id") long id);
}
