package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 工具执行上下文
 * <p>
 * 封装工具执行时所需的用户身份和权限信息。
 * 通过方法参数显式传递，不依赖 SecurityContext（ThreadLocal），
 * 解决 Reactor 异步线程中上下文丢失的问题，同时兼容未来微服务化场景（可通过 HTTP/RPC 序列化传递）。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ToolExecutionContext {

    /**
     * 当前操作用户 ID
     */
    private Long userId;

    /**
     * 所有者类型：1=个人（预留扩展团队/企业）
     */
    private Integer ownerType;

    /**
     * 所有者 ID（个人模式下等于 userId）
     */
    private Long ownerId;

    /**
     * AgentScope 从可信运行上下文投影的业务写入身份，不能由模型参数构造。
     */
    private ResourceCommandContext mutationContext;

    private CanvasAgentScope canvasScope;
    /**
     * 当前内核已固定的模型，仅用于校验工具的多模态结果投影。
     */
    private AiModel model;

    public static ToolExecutionContext forInvocation(Long userId, Integer ownerType, Long ownerId,
                                                     ResourceCommandContext mutationContext) {
        return ToolExecutionContext.builder().userId(userId).ownerType(ownerType).ownerId(ownerId)
                .mutationContext(mutationContext).build();
    }

    public ResourceCommandContext requireMutationContext() {
        if (mutationContext == null) {
            throw new IllegalStateException("缺少可信的工具调用上下文");
        }
        return mutationContext;
    }
}
