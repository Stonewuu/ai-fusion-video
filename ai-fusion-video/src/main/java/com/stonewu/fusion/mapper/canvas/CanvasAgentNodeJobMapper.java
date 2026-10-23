package com.stonewu.fusion.mapper.canvas;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasAgentNodeJobEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface CanvasAgentNodeJobMapper extends BaseMapper<CanvasAgentNodeJobEntity> {
    List<String> previousScopes(@Param("parent") String parent, @Param("attempt") int attempt,
                                @Param("itemIndex") int itemIndex);

    String lockActiveScope(@Param("nodeRunId") String nodeRunId);

    long countAgentRuns(@Param("nodeRunId") String nodeRunId);

    List<String> pendingChildren(@Param("instance") String instance);

    CanvasAgentNodeJobEntity lock(@Param("nodeRunId") String nodeRunId);

    int insertQueued(@Param("nodeRunId") String nodeRunId, @Param("canvasRunId") String canvasRunId,
                     @Param("agentRunId") String agentRunId, @Param("conversationId") String conversationId,
                     @Param("snapshot") String snapshot, @Param("input") String input, @Param("scope") String scope);

    List<String> pending(@Param("instance") String instance);

    int dispatching(@Param("nodeRunId") String nodeRunId);

    int dispatched(@Param("nodeRunId") String nodeRunId, @Param("epoch") long epoch);

    int updateScope(@Param("scope") String scope, @Param("nodeRunId") String nodeRunId);

    int requestCancel(@Param("nodeRunId") String nodeRunId);

    List<String> agentRunIds(@Param("nodeRunId") String nodeRunId);

    List<String> unresolvedModelIds(@Param("nodeRunId") String nodeRunId);

    List<String> childStates(@Param("nodeRunId") String nodeRunId, @Param("runId") String runId);

    String ended(@Param("id") String id, @Param("nodeRunId") String nodeRunId);
}
