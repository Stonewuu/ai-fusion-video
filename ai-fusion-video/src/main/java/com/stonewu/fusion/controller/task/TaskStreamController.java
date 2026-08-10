package com.stonewu.fusion.controller.task;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.service.task.TaskStreamService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

import java.util.List;

import static com.stonewu.fusion.security.SecurityUtils.requireCurrentUserId;

/**
 * 通用任务流 Controller。
 */
@Tag(name = "任务流")
@RestController
@RequestMapping("/api/task-stream")
@RequiredArgsConstructor
public class TaskStreamController {

    private final TaskStreamService taskStreamService;

    @Operation(summary = "重连任务流（页面刷新后恢复 SSE）")
    @GetMapping(value = "/reconnect", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<AiChatStreamRespVO> reconnect(@RequestParam String taskId) {
        return taskStreamService.reconnect(taskId);
    }

    @Operation(summary = "查询任务流状态")
    @GetMapping("/status")
    public CommonResult<String> getStatus(@RequestParam String taskId) {
        return CommonResult.success(taskStreamService.getStatus(taskId));
    }

    @Operation(summary = "查询运行中的任务流列表")
    @GetMapping("/running")
    public CommonResult<List<AgentConversation>> listRunning() {
        Long userId = requireCurrentUserId();
        return CommonResult.success(taskStreamService.listRunning(userId));
    }
}