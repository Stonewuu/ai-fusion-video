package com.stonewu.fusion.mapper.execution;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.execution.ExecutionRecoveryDecisionEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ExecutionRecoveryDecisionMapper extends BaseMapper<ExecutionRecoveryDecisionEntity> {
    ExecutionRecoveryDecisionEntity lockRequest(@Param("admin") long admin, @Param("requestId") String requestId);

    int record(@Param("row") ExecutionRecoveryDecisionEntity row);
}
