package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageParam;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.List;

import static com.stonewu.fusion.security.SecurityUtils.requireCurrentUserId;

/**
 * AI 助手 Controller（对话管理 + 后续多轮 Chat）
 * <p>
 * Pipeline 相关接口已迁移至 {@link AiPipelineController}
 */
@Tag(name = "AI 助手")
@RestController
@RequestMapping("/api/ai/assistant")
@RequiredArgsConstructor
public class AiAssistantController {

    private final AgentConversationService conversationService;
    private final AgentMessageService messageService;

    // ========== 对话管理 ==========

    @Operation(summary = "获取对话列表（当前用户）")
    @GetMapping("/conversations")
    public CommonResult<PageResult<AgentConversation>> listConversations(
            PageParam pageParam,
            @RequestParam(required = false) String category) {
        Long userId = requireCurrentUserId();
        PageResult<AgentConversation> result;
        if (category != null) {
            result = conversationService.listByUserAndCategory(userId, category,
                    pageParam.getPageNo(), pageParam.getPageSize());
        } else {
            result = conversationService.listByUser(userId,
                    pageParam.getPageNo(), pageParam.getPageSize());
        }
        return CommonResult.success(result);
    }

    @Operation(summary = "获取对话消息列表")
    @GetMapping("/conversations/{conversationId}/messages")
    public CommonResult<List<AgentMessage>> listMessages(@PathVariable String conversationId) {
        long currentUserId = requireCurrentUserId();
        if (conversationService.getOwnedByConversationId(conversationId, currentUserId) == null) {
            // Return the same not-found response for an unknown id and an id
            // owned by another user; this avoids turning the endpoint into an
            // ownership oracle.
            throw new BusinessException(404, "对话不存在");
        }
        return CommonResult.success(messageService.listByConversation(conversationId));
    }

    @Operation(summary = "删除对话")
    @DeleteMapping("/conversations/{id}")
    public Mono<CommonResult<Boolean>> deleteConversation(@PathVariable Long id) {
        long currentUserId = requireCurrentUserId();
        return conversationService.deleteConversation(id, currentUserId)
                .thenReturn(CommonResult.success(true));
    }
}

