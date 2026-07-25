package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.controller.ai.vo.PipelineRunStatusRespVO;
import com.stonewu.fusion.controller.ai.vo.RunningPipelineRunRespVO;
import com.stonewu.fusion.service.ai.agentscope.AgentScopePipelineRunService;
import com.stonewu.fusion.service.ai.run.AgentRunQueryService;
import com.stonewu.fusion.service.ai.run.AgentRunReplayService;
import com.stonewu.fusion.service.ai.run.CancellationCoordinator;
import com.stonewu.fusion.service.ai.run.PipelineCursorParser;
import com.stonewu.fusion.service.ai.run.model.RunCursor;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;

import static com.stonewu.fusion.security.SecurityUtils.requireCurrentUserId;

/** Durable, user-authorized pipeline HTTP and SSE API. */
@Tag(name = "AI Pipeline")
@RestController
@RequestMapping("/api/ai/pipeline")
@RequiredArgsConstructor
public class AiPipelineController {

    private final AgentScopePipelineRunService pipelineRuns;
    private final AgentRunQueryService runQueries;
    private final AgentRunReplayService replayService;
    private final PipelineCursorParser cursorParser;
    private final CancellationCoordinator cancellations;

    @Operation(summary = "启动 Pipeline（SSE 流式）")
    @PostMapping(value = "/run", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<AiChatStreamRespVO>> run(
            @RequestBody AiChatReqVO request) {
        long currentUserId = requireCurrentUserId();
        return runQueries.authorizeConversationForStart(
                request.getConversationId(), currentUserId)
                .thenMany(Flux.defer(() -> pipelineRuns.stream(
                        request, currentUserId)))
                .map(this::toSse);
    }

    @Operation(summary = "取消 Pipeline")
    @PostMapping("/cancel")
    public Mono<CommonResult<Boolean>> cancel(
            @RequestParam(required = false) String runId,
            @RequestParam(required = false) String conversationId) {
        long currentUserId = requireCurrentUserId();
        return runQueries.resolveAuthorizedTarget(
                runId, conversationId, currentUserId)
                .flatMap(run -> cancellations.cancel(
                        run.getRunId(), currentUserId))
                .thenReturn(CommonResult.success(true));
    }

    @Operation(summary = "重连 Pipeline")
    @GetMapping(value = "/reconnect", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<AiChatStreamRespVO>> reconnect(
            @RequestParam(required = false) String runId,
            @RequestParam(required = false) String conversationId,
            @RequestParam(required = false) Long afterSequence,
            @RequestHeader(name = "Last-Event-ID", required = false) String lastEventId) {
        long currentUserId = requireCurrentUserId();
        return runQueries.resolveAuthorizedTarget(
                runId, conversationId, currentUserId)
                .flatMapMany(run -> {
                    RunCursor cursor = cursorParser.parse(
                            run.getRunId(), afterSequence, lastEventId);
                    return replayService.replayThenLive(
                            cursor.runId(), cursor.afterSequence())
                            .concatMap(event -> runQueries.project(run, event))
                            .map(this::toSse);
                });
    }

    @Operation(summary = "查询 Pipeline 运行状态")
    @GetMapping("/status")
    public Mono<CommonResult<PipelineRunStatusRespVO>> getStatus(
            @RequestParam(required = false) String runId,
            @RequestParam(required = false) String conversationId) {
        long currentUserId = requireCurrentUserId();
        return runQueries.status(runId, conversationId, currentUserId)
                .map(CommonResult::success);
    }

    @Operation(summary = "查询运行中的 Pipeline 列表")
    @GetMapping("/running")
    public Mono<CommonResult<List<RunningPipelineRunRespVO>>> listRunning() {
        long currentUserId = requireCurrentUserId();
        return runQueries.listRunning(currentUserId)
                .map(CommonResult::success);
    }

    private ServerSentEvent<AiChatStreamRespVO> toSse(
            AiChatStreamRespVO event) {
        ServerSentEvent.Builder<AiChatStreamRespVO> builder = ServerSentEvent.<AiChatStreamRespVO>builder(event)
                .event("pipeline-event");
        if (event.getRunId() == null && event.getSequence() == null) {
            return builder.build();
        }
        if (event.getRunId() == null || event.getRunId().isBlank()
                || event.getSequence() == null || event.getSequence() <= 0) {
            throw new IllegalStateException(
                    "Pipeline SSE event has incomplete durable identity");
        } else {
            builder.id(event.getRunId() + ':' + event.getSequence());
        }
        return builder.build();
    }
}
