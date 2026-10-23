package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.ApiConfig;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ApiConfigMapper extends BaseMapper<ApiConfig> {
    ApiConfig selectByIdForUpdate(@Param("id") Long id);
}
