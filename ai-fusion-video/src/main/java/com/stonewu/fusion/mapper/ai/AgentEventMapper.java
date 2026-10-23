package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentEvent;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentEventMapper extends BaseMapper<AgentEvent> {

    AgentEvent selectByRunAndSequence(
            @Param("runId") String runId,
            @Param("sequenceNo") long sequenceNo);

    List<AgentEvent> selectPendingPublishCandidatesForUpdate(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    List<AgentEvent> selectExpiredPublishCandidatesForUpdate(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    int claimPublishCandidate(
            @Param("eventId") long eventId,
            @Param("claimOwner") String claimOwner,
            @Param("claimUntil") LocalDateTime claimUntil,
            @Param("now") LocalDateTime now);

    int markPublished(
            @Param("eventId") long eventId,
            @Param("claimOwner") String claimOwner);

    int releasePublishForRetry(
            @Param("eventId") long eventId,
            @Param("claimOwner") String claimOwner,
            @Param("nextAttemptAt") LocalDateTime nextAttemptAt,
            @Param("lastPublishError") String lastPublishError);

    long countOutstandingPublish();

    List<AgentEvent> selectProjectionRange(
            @Param("runId") String runId,
            @Param("afterSequence") long afterSequence,
            @Param("throughSequence") long throughSequence,
            @Param("limit") int limit);

    List<AgentEvent> selectReplayRange(
            @Param("runId") String runId,
            @Param("afterSequence") long afterSequence,
            @Param("throughSequence") long throughSequence,
            @Param("limit") int limit);

    long selectLastContextBoundarySequence(
            @Param("runId") String runId,
            @Param("beforeSequence") long beforeSequence,
            @Param("parentToolCallId") String parentToolCallId);

    List<AgentEvent> selectContextDeltas(
            @Param("runId") String runId,
            @Param("afterSequence") long afterSequence,
            @Param("beforeSequence") long beforeSequence,
            @Param("parentToolCallId") String parentToolCallId);

    List<AgentEvent> selectToolDeltas(
            @Param("runId") String runId,
            @Param("toolCallId") String toolCallId,
            @Param("throughSequence") long throughSequence);

    AgentEvent selectConfirmationCandidate(
            @Param("runId") String runId,
            @Param("replyId") String replyId);

    AgentEvent selectLatestConfirmationCandidate(@Param("runId") String runId);

    AgentEvent selectPendingExternalExecution(
            @Param("runId") String runId,
            @Param("toolCallId") String toolCallId);
}
