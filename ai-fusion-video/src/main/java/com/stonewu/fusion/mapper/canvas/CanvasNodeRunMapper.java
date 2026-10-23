package com.stonewu.fusion.mapper.canvas;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasNodeRunEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface CanvasNodeRunMapper extends BaseMapper<CanvasNodeRunEntity> {
    int insertBatchItem(@Param("id") String id, @Param("index") int index, @Param("attempt") int attempt,
                        @Param("fingerprint") String fingerprint, @Param("plan") String plan,
                        @Param("input") String input, @Param("output") String output,
                        @Param("parentId") String parentId);

    int insertPlanned(@Param("row") CanvasNodeRunEntity row);

    CanvasNodeRunEntity lock(@Param("id") String id);

    CanvasNodeRunEntity latestSucceeded(@Param("canvasId") String canvasId, @Param("nodeId") String nodeId);

    List<CanvasNodeRunEntity> latestNodes(@Param("canvasId") String canvasId);

    int updateState(@Param("status") String status, @Param("input") String input, @Param("output") String output,
                    @Param("checkpoint") String checkpoint, @Param("kind") String kind,
                    @Param("externalId") String externalId, @Param("error") String error, @Param("delay") int delay,
                    @Param("id") String id);

    int defer(@Param("seconds") int seconds, @Param("id") String id);

    int runningCount(@Param("runId") String runId);

    String claimCandidate(@Param("runId") String runId, @Param("externalCount") int externalCount,
                          @Param("parallelism") int parallelism);

    int claim(@Param("owner") String owner, @Param("id") String id);

    CanvasNodeRunEntity leased(@Param("id") String id, @Param("runId") String runId, @Param("owner") String owner,
                               @Param("epoch") long epoch);

    int heartbeat(@Param("id") String id, @Param("runId") String runId, @Param("owner") String owner,
                  @Param("epoch") long epoch);

    boolean hasActiveNodes(@Param("canvasId") String canvasId, @Param("nodeIds") List<String> nodeIds);
}
