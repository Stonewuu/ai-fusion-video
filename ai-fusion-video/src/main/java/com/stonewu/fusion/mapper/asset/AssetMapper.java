package com.stonewu.fusion.mapper.asset;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.asset.Asset;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import com.baomidou.mybatisplus.core.metadata.IPage;

import java.util.List;

@Mapper
public interface AssetMapper extends BaseMapper<Asset> {
    record AccessFilter(long userId, Long teamId, List<Long> projects, Long projectId, String type, String keyword) {
    }

    record TypeCount(String type, long total) {
    }

    List<Long> accessibleIds(IPage<?> page, @Param("filter") AccessFilter filter);

    long countAccessible(@Param("filter") AccessFilter filter);

    List<TypeCount> countAccessibleTypes(@Param("filter") AccessFilter filter);

    Asset selectOwnerRecord(@Param("id") long id, @Param("includeDeleted") boolean includeDeleted);

    List<Long> lockNamed(@Param("projectId") Long projectId, @Param("ownerType") int ownerType,
                         @Param("ownerId") long ownerId, @Param("type") String type, @Param("name") String name);
}
