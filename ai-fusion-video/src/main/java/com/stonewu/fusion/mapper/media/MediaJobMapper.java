package com.stonewu.fusion.mapper.media;


import java.util.List;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.media.MediaJobEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MediaJobMapper extends BaseMapper<MediaJobEntity> {
    Long previewIngestEpoch(@Param("id") String id);

    int previewIngestOwner(@Param("epoch") long epoch, @Param("id") String id);

    int activePreviews();

    List<String> nextPreview();

    int claimPreview(@Param("owner") String owner, @Param("id") String id);

    int completePreview(@Param("checkpoint") String checkpoint, @Param("id") String id);

    int waitForPreviewIngest(@Param("id") String id);

    List<String> lockPreviewOwner(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int stopPreview(@Param("status") String status, @Param("phase") String phase, @Param("error") String error,
                    @Param("id") String id);

    int insertPreview(@Param("id") String id, @Param("projectId") Long projectId, @Param("actor") long actor,
                      @Param("requestId") String requestId, @Param("requestHash") String requestHash,
                      @Param("context") String context, @Param("input") String input);

    List<MediaJobEntity> nextCompose();

    int claimCompose(@Param("owner") String owner, @Param("id") String id);

    int acquireEpisodeAudioPermit(@Param("id") String id);

    int heartbeat(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int checkpoint(@Param("checkpoint") String checkpoint, @Param("phase") String phase, @Param("id") String id);

    int waitForProvider(@Param("seconds") int seconds, @Param("id") String id);

    List<String> lockEpisodeAudioPermit(@Param("id") String id);

    List<String> importAgentScope(@Param("runId") String runId, @Param("userId") long userId,
                                  @Param("teamId") Long teamId);

    List<String> importNodeRuns(@Param("id") String id, @Param("canvasId") String canvasId);

    List<String> activeCanvasNodes(@Param("id") String id, @Param("canvasId") String canvasId,
                                   @Param("userId") long userId, @Param("teamId") Long teamId,
                                   @Param("allowCancel") boolean allowCancel);

    List<String> activeImportParents(@Param("nodeRunId") String nodeRunId, @Param("id") String id,
                                     @Param("canvasId") String canvasId, @Param("userId") long userId,
                                     @Param("teamId") Long teamId, @Param("allowCancel") boolean allowCancel);

    List<String> lockOwner(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch,
                           @Param("allowCancel") boolean allowCancel);

    int countWorkflowTesterRoles(@Param("actor") long actor);

    MediaJobEntity lock(@Param("id") String id);

    int recordFailure(@Param("error") String error, @Param("id") String id, @Param("owner") String owner,
                      @Param("epoch") long epoch);

    int terminal(@Param("status") String status, @Param("phase") String phase, @Param("error") String error,
                 @Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int requestComposeCancel(@Param("id") String id);

    int activeImports();

    List<String> nextImport();

    int waitForImportIngest(@Param("id") String id);

    List<String> lockImportOwner(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int stopImport(@Param("message") String message, @Param("checkpoint") String checkpoint, @Param("id") String id);

    int importProgress(@Param("status") String status, @Param("checkpoint") String checkpoint,
                       @Param("phase") String phase, @Param("id") String id);

    int importPlacement(@Param("checkpoint") String checkpoint, @Param("id") String id);

    MediaJobEntity lockRequest(@Param("actor") long actor, @Param("requestId") String requestId);

    int insertImport(@Param("id") String id, @Param("projectId") Long projectId, @Param("actor") long actor,
                     @Param("requestId") String requestId, @Param("requestHash") String requestHash,
                     @Param("context") String context, @Param("input") String input,
                     @Param("checkpoint") String checkpoint);

    int cancelImport(@Param("status") String status, @Param("checkpoint") String checkpoint, @Param("id") String id);

    int retryImport(@Param("checkpoint") String checkpoint, @Param("id") String id);

    int insertCompose(@Param("id") String id, @Param("projectId") Long projectId, @Param("episodeId") long episodeId,
                      @Param("version") int version, @Param("actor") long actor, @Param("requestId") String requestId,
                      @Param("requestHash") String requestHash, @Param("context") String context,
                      @Param("input") String input);

    int advanceRevision(@Param("id") String id);

    int waitForAudio(@Param("status") String status, @Param("phase") String phase, @Param("error") String error,
                     @Param("seconds") int seconds, @Param("id") String id);

    int resumeReviewedAudio(@Param("id") String id);

    List<String> dueComfyTests();

    int heartbeatComfyTest(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int submittingComfyTest(@Param("id") String id);

    int ingestingComfyTest(@Param("id") String id);

    int waitForComfyTest(@Param("phase") String phase, @Param("error") String error, @Param("seconds") int seconds,
                         @Param("id") String id);

    int finishComfyTest(@Param("status") String status, @Param("error") String error, @Param("id") String id);

    List<String> lockComfyTestOwner(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int insertComfyTest(@Param("id") String id, @Param("actor") long actor, @Param("requestId") String requestId,
                        @Param("requestHash") String requestHash, @Param("context") String context,
                        @Param("input") String input, @Param("minutes") int minutes);

    int cancelComfyTest(@Param("status") String status, @Param("error") String error, @Param("id") String id);

    int reviewComfyTest(@Param("status") String status, @Param("error") String error, @Param("id") String id);

    int resumeComfyTest(@Param("id") String id);

    String endedCanvas(@Param("id") String id, @Param("nodeRunId") String nodeRunId);

    int insertCanvasMedia(@Param("id") String id, @Param("projectId") Long projectId, @Param("actor") long actor,
                          @Param("requestId") String requestId, @Param("hash") String hash,
                          @Param("context") String context, @Param("input") String input);

    List<String> nextCanvasMedia();

    int claimCanvasMedia(@Param("owner") String owner, @Param("id") String id);

    int waitForCanvasMedia(@Param("phase") String phase, @Param("seconds") int seconds, @Param("id") String id);

    int finishCanvasMedia(@Param("status") String status, @Param("phase") String phase, @Param("error") String error,
                          @Param("checkpoint") String checkpoint, @Param("id") String id, @Param("owner") String owner,
                          @Param("epoch") long epoch);

    int cancelCanvasMedia(@Param("id") String id);

    int reviewCanvasMedia(@Param("status") String status, @Param("error") String error, @Param("id") String id);

    int resumeCanvasMedia(@Param("id") String id);

    int adminCancel(@Param("message") String message, @Param("id") String id);
}
