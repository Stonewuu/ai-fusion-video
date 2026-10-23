package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasMcpContractEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasMcpContractMapper extends BaseMapper<CanvasMcpContractEntity> {
    CanvasMcpContractEntity lock(@Param("id") String id, @Param("userId") long userId);

    int create(@Param("id") String id, @Param("userId") long userId, @Param("name") String name,
               @Param("toolName") String toolName, @Param("serverName") String serverName,
               @Param("definition") String definition, @Param("hash") String hash, @Param("enabled") boolean enabled);

    int revise(@Param("name") String name, @Param("toolName") String toolName, @Param("serverName") String serverName,
               @Param("definition") String definition, @Param("hash") String hash, @Param("enabled") boolean enabled,
               @Param("id") String id);

    int remove(@Param("id") String id);
}
