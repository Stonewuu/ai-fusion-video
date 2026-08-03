package com.stonewu.fusion.controller.ai.vo;

/** Authoritative status snapshot for one durable root or child run. */
public record PipelineRunStatusRespVO(
        String runId,
        String status,
        long lastSequence,
        String waitingReplyId,
        AiChatStreamRespVO terminalEvent) {
}
