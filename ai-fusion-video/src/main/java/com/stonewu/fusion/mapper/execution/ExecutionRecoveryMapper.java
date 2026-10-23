package com.stonewu.fusion.mapper.execution;


import org.apache.ibatis.annotations.Param;

import java.util.List;

import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Ref;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Kind;
import com.stonewu.fusion.service.generation.GenerationKind;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ExecutionRecoveryMapper {
    record Observation(String id, String status, long revision, long actorUserId, Long actorTeamId,
                       Long projectId, String canvasId, String runId, boolean permitHeld, boolean cancelled,
                       boolean started, boolean received) {
    }

    List<Ref> candidates(@Param("offset") int offset);

    Observation agentModel(@Param("id") String id);

    Observation mcp(@Param("id") String id);

    Observation text(@Param("id") String id);

    String lockCanvas(@Param("id") String id);

    String lockRun(@Param("id") String id);

    Long lockProject(@Param("id") long id);

    Long lockWorkflow(@Param("id") long id);

    String lockExecution(@Param("kind") Kind kind, @Param("id") String id);

    Long lockGenerationTask(@Param("kind") GenerationKind kind, @Param("id") long id);
}
