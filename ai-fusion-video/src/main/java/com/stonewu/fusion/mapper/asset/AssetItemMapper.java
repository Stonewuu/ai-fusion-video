package com.stonewu.fusion.mapper.asset;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.asset.AssetItem;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface AssetItemMapper extends BaseMapper<AssetItem> {
    List<Long> lockNamed(@Param("assetId") long assetId, @Param("name") String name);

    List<Long> lockChildren(@Param("assetId") long assetId);

    long parentAssetId(@Param("id") long id);
}
