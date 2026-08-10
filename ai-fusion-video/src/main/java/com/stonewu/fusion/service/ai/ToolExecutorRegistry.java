package com.stonewu.fusion.service.ai;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 工具执行器注册中心
 * <p>
 * 自动收集所有 {@link ToolExecutor} Bean，按 toolName 建立索引，
 * 提供 AgentScope V2 kernel 按名称查找严格白名单工具的能力。
 */
@Component
@Slf4j
public class ToolExecutorRegistry {

    private final Map<String, ToolExecutor> executorMap;

    public ToolExecutorRegistry(List<ToolExecutor> toolExecutors) {
        this.executorMap = toolExecutors.stream()
                .collect(Collectors.toMap(ToolExecutor::getToolName, Function.identity()));
        log.info("[ToolExecutorRegistry] 已注册 {} 个工具执行器: {}",
                executorMap.size(), executorMap.keySet());
    }

    /**
     * 查找工具执行器
     *
     * @param toolName 工具名称
     * @return 工具执行器，不存在返回 null
     */
    public ToolExecutor findExecutor(String toolName) {
        return executorMap.get(toolName);
    }
}
