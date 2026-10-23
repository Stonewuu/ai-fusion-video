package com.stonewu.fusion.mapper.resource;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;

import com.stonewu.fusion.entity.resource.ResourceRevisionEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ResourceRevisionMapper extends BaseMapper<ResourceRevisionEntity> {
    int capture(@Param("row") ResourceRevisionEntity row);

}
