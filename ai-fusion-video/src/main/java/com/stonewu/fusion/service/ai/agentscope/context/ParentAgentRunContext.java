package com.stonewu.fusion.service.ai.agentscope.context;

/** Immutable identity used to durably expose child events on the parent run stream. */
public record ParentAgentRunContext(
        String runId,
        String ownerInstanceId,
        long ownerEpoch,
        String toolCallId,
        String agentName) {

    public ParentAgentRunContext {
        runId = ContextValues.requireText(runId, "runId");
        ownerInstanceId = ContextValues.requireText(
                ownerInstanceId, "ownerInstanceId");
        ownerEpoch = ContextValues.requirePositive(ownerEpoch, "ownerEpoch");
        toolCallId = ContextValues.requireText(toolCallId, "toolCallId");
        agentName = ContextValues.requireText(agentName, "agentName");
    }
}
