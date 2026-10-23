package com.stonewu.fusion.service.ai;

import java.util.Map;

/**
 * 服务端运行范围的工具边界；提示词或模型参数不能关闭这些校验。
 */
public interface AgentToolInvocationPolicy {
    void before(String toolName, Map<String, Object> input, ToolExecutionContext context);
}
