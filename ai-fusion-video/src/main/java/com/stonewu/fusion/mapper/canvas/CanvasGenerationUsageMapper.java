package com.stonewu.fusion.mapper.canvas;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface CanvasGenerationUsageMapper {
    record Tokens(long id, Long inputTokens, Long outputTokens) {
    }

    List<Tokens> tokensForRun(@Param("runId") String runId);

    record Usage(String inputJson, String modelSnapshotJson) {
    }

    List<Usage> selectNodeUsage(@Param("nodeRunId") String nodeRunId, @Param("kind") String kind);

    List<String> successfulImages(@Param("nodeRunId") String nodeRunId);

    List<String> successfulVideos(@Param("nodeRunId") String nodeRunId);
}
