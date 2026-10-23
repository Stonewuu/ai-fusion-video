package com.stonewu.fusion.mapper.generation;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface GenerationSubmissionMapper extends BaseMapper<GenerationSubmissionEntity> {
    String lock(@Param("id") String id);

    long countActiveJobs(@Param("kind") String kind, @Param("taskId") long taskId);

    GenerationSubmissionEntity selectAgentInvocation(@Param("scope") String scope,
                                                     @Param("agentRunId") String agentRunId,
                                                     @Param("toolCallId") String toolCallId);

    int insertQueued(@Param("row") GenerationSubmissionEntity row);

    int aggregate(@Param("state") String state, @Param("id") String id);

    String endedNode(@Param("id") String id, @Param("nodeRunId") String nodeRunId);
}
