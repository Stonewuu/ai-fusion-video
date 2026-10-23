package com.stonewu.fusion.mapper.generation;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface GenerationWithdrawalMapper {
    String selectKind(@Param("scope") String scope, @Param("requestId") String requestId);

    int insert(@Param("scope") String scope, @Param("requestId") String requestId, @Param("kind") String kind,
               @Param("userId") long userId, @Param("teamId") Long teamId);
}
