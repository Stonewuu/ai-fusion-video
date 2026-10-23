package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;

/**
 * 由服务器生成的审批说明；只用于展示，执行仍使用原始且持久化的工具调用。
 */
public interface AgentConfirmationPreviewProvider {
    boolean supports(String toolName);

    JsonNode preview(String toolName, JsonNode input, AgentScopeRuntimeContextRequest runtime);
}
