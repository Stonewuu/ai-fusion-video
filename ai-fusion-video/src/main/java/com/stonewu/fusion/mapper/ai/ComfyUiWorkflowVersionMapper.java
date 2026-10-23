package com.stonewu.fusion.mapper.ai;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ComfyUiWorkflowVersionMapper extends BaseMapper<ComfyUiWorkflowVersion> {
    int recordTest(@Param("status") int status, @Param("message") String message, @Param("apiHash") String apiHash,
                   @Param("versionId") long versionId, @Param("runId") String runId,
                   @Param("workflowHash") String workflowHash, @Param("modelType") Integer modelType,
                   @Param("apiConfigId") long apiConfigId);

    ComfyUiWorkflowVersion selectForUpdate(@Param("id") Long id);
}
