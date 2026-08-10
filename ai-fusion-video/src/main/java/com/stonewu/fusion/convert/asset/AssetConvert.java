package com.stonewu.fusion.convert.asset;

import com.stonewu.fusion.controller.asset.vo.AssetCreateReqVO;
import com.stonewu.fusion.controller.asset.vo.AssetItemCreateReqVO;
import com.stonewu.fusion.controller.asset.vo.AssetItemUpdateReqVO;
import com.stonewu.fusion.controller.asset.vo.AssetUpdateReqVO;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.factory.Mappers;

/**
 * 资产 Convert
 */
@Mapper
public interface AssetConvert {

    AssetConvert INSTANCE = Mappers.getMapper(AssetConvert.class);

    @Mapping(target = "id", ignore = true)
    @Mapping(target = "userId", ignore = true)
    @Mapping(target = "ownerType", ignore = true)
    @Mapping(target = "ownerId", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "sourceType", ignore = true)
    Asset convert(AssetCreateReqVO reqVO);

    @Mapping(target = "userId", ignore = true)
    @Mapping(target = "projectId", ignore = true)
    @Mapping(target = "ownerType", ignore = true)
    @Mapping(target = "ownerId", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "sourceType", ignore = true)
    Asset convert(AssetUpdateReqVO reqVO);

    @Mapping(target = "id", ignore = true)
    @Mapping(target = "sourceType", ignore = true)
    AssetItem convert(AssetItemCreateReqVO reqVO);

    @Mapping(target = "assetId", ignore = true)
    @Mapping(target = "sourceType", ignore = true)
    AssetItem convert(AssetItemUpdateReqVO reqVO);
}
