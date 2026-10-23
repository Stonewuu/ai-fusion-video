package com.stonewu.fusion.mapper.ai;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentRunMapper extends BaseMapper<AgentRun> {
    String lockActiveMutation(@Param("context") ResourceCommandContext context, @Param("projectId") Long projectId);

    AgentRun activeMediaScope(@Param("runId") String runId, @Param("userId") long userId,
                              @Param("owner") String owner, @Param("epoch") long epoch);

    boolean hasPendingCanvasNode(@Param("runId") String runId);

    AgentRun selectByRunIdForUpdate(@Param("runId") String runId);

    AgentRun selectByRunId(@Param("runId") String runId);

    List<AgentRun> selectProjectionRecoveryCandidates(@Param("limit") int limit);

    AgentRun selectAuthorizedByRunId(
            @Param("runId") String runId,
            @Param("userId") long userId);

    AgentRun selectAuthorizedPreferredRootByConversation(
            @Param("conversationId") String conversationId,
            @Param("userId") long userId);

    AgentRun selectLatestRootByConversation(
            @Param("conversationId") String conversationId);

    List<AgentRun> selectAuthorizedRunningRoots(@Param("userId") long userId);

    List<String> selectStateSessionIdsByConversation(
            @Param("conversationId") String conversationId);

    LocalDateTime selectDatabaseNow();

    AgentRun selectByParentAndToolCallForUpdate(
            @Param("parentRunId") String parentRunId,
            @Param("parentToolCallId") String parentToolCallId);

    List<AgentRun> selectActiveChildren(@Param("parentRunId") String parentRunId);

    int advanceRunningSequence(
            @Param("runId") long runId,
            @Param("ownerEpoch") long ownerEpoch,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    int enterWaitingConfirmation(
            @Param("runId") long runId,
            @Param("ownerEpoch") long ownerEpoch,
            @Param("replyId") String replyId,
            @Param("expiresAt") LocalDateTime expiresAt,
            @Param("pausedThroughSequence") long pausedThroughSequence,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    int enterWaitingExternal(
            @Param("runId") long runId,
            @Param("ownerEpoch") long ownerEpoch,
            @Param("toolCallId") String toolCallId,
            @Param("toolName") String toolName,
            @Param("expiresAt") LocalDateTime expiresAt,
            @Param("deadlineAt") LocalDateTime deadlineAt,
            @Param("pausedThroughSequence") long pausedThroughSequence,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    int resumeConfirmation(
            @Param("runId") long runId,
            @Param("oldOwnerEpoch") long oldOwnerEpoch,
            @Param("replyId") String replyId,
            @Param("newOwnerInstanceId") String newOwnerInstanceId,
            @Param("newOwnerEpoch") long newOwnerEpoch,
            @Param("leaseUntil") LocalDateTime leaseUntil,
            @Param("deadlineAt") LocalDateTime deadlineAt,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    int resumeExternal(
            @Param("runId") long runId,
            @Param("oldOwnerEpoch") long oldOwnerEpoch,
            @Param("toolCallId") String toolCallId,
            @Param("newOwnerInstanceId") String newOwnerInstanceId,
            @Param("newOwnerEpoch") long newOwnerEpoch,
            @Param("leaseUntil") LocalDateTime leaseUntil,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    int recordCanvasResumeDispatch(@Param("runId") String runId, @Param("epoch") long epoch,
                                   @Param("payload") String payload, @Param("sequence") long sequence,
                                   @Param("now") LocalDateTime now, @Param("deadline") LocalDateTime deadline,
                                   @Param("leaseUntil") LocalDateTime leaseUntil);

    int renewOwnedLease(
            @Param("runId") String runId,
            @Param("ownerInstanceId") String ownerInstanceId,
            @Param("ownerEpoch") long ownerEpoch,
            @Param("leaseMicros") long leaseMicros);

    int countValidOwnedLease(
            @Param("runId") String runId,
            @Param("ownerInstanceId") String ownerInstanceId,
            @Param("ownerEpoch") long ownerEpoch);

    int acknowledgeOwnedCancellation(
            @Param("runId") String runId,
            @Param("ownerInstanceId") String ownerInstanceId,
            @Param("ownerEpoch") long ownerEpoch);

    int markCancellationBroadcast(
            @Param("runId") String runId,
            @Param("now") LocalDateTime now,
            @Param("nextAttemptAt") LocalDateTime nextAttemptAt);

    List<AgentRun> selectExpiredLeaseCandidates(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    List<AgentRun> selectCancellationRetryCandidates(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    List<String> waitingForCanvas();

    List<String> dispatchingCanvasResume(@Param("instance") String instance);

    int deferCanvasCheck(@Param("id") String id);

    int clearCanvasResume(@Param("id") String id, @Param("epoch") long epoch, @Param("sequence") long sequence);
}
