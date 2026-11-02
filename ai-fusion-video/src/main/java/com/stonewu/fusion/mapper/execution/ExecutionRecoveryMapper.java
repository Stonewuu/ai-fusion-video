package com.stonewu.fusion.mapper.execution;


import org.apache.ibatis.annotations.Param;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ExecutionRecoveryMapper {
    record Observation(String id, String status, long revision, long actorUserId, Long actorTeamId,
                       Long projectId, String canvasId, String runId, boolean permitHeld, boolean cancelled,
                       boolean started, boolean received) {
    }


    Observation agentModel(@Param("id") String id);

    Observation mcp(@Param("id") String id);

    Observation text(@Param("id") String id);

}
