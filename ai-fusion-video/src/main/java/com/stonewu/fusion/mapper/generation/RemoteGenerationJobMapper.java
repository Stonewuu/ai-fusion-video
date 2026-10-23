package com.stonewu.fusion.mapper.generation;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.generation.RemoteGenerationJobEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface RemoteGenerationJobMapper extends BaseMapper<RemoteGenerationJobEntity> {
    List<RemoteGenerationJobEntity> selectClaimCandidates();

    int insertQueued(@Param("row") RemoteGenerationJobEntity row);

    int deferForCapacity(@Param("id") String id);

    int claim(@Param("owner") String owner, @Param("permitHeld") boolean permitHeld, @Param("id") String id);

    int heartbeat(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    RemoteGenerationJobEntity selectLeasedForUpdate(@Param("id") String id, @Param("owner") String owner,
                                                    @Param("epoch") long epoch);

    int schedulePoll(@Param("phase") String phase, @Param("seconds") int seconds, @Param("errors") int errors,
                     @Param("errorCode") String errorCode, @Param("errorMessage") String errorMessage,
                     @Param("id") String id);

    int receiveResult(@Param("result") String result, @Param("id") String id);

    int prepareAttempt(@Param("attemptId") String attemptId, @Param("id") String id);

    int beginSubmission(@Param("providerTaskId") String providerTaskId, @Param("id") String id);

    int applyReceipt(@Param("providerTaskId") String providerTaskId, @Param("result") String result,
                     @Param("phase") String phase, @Param("permitHeld") boolean permitHeld, @Param("id") String id);

    int markReview(@Param("state") String state, @Param("code") String code, @Param("message") String message,
                   @Param("id") String id);

    int finish(@Param("state") String state, @Param("code") String code, @Param("message") String message,
               @Param("id") String id);

    int requestCancellation(@Param("submissionId") String submissionId);

    int cancelQueued(@Param("id") String id);

    String lock(@Param("id") String id);

    int restoreReceipt(@Param("attemptId") String attemptId, @Param("id") String id);

    int recheck(@Param("id") String id);

    int retryIngest(@Param("id") String id);

    int retrySubmission(@Param("id") String id);

    int confirmFinished(@Param("state") String state, @Param("message") String message, @Param("id") String id);

    int adminCancel(@Param("message") String message, @Param("id") String id);
}
