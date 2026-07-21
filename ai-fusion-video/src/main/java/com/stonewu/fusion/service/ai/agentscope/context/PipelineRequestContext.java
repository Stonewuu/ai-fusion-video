package com.stonewu.fusion.service.ai.agentscope.context;

import java.util.Objects;

public record PipelineRequestContext(String requestId, Kind kind) {

    public PipelineRequestContext {
        requestId = ContextValues.requireText(requestId, "requestId");
        kind = Objects.requireNonNull(kind, "kind must not be null");
    }

    public enum Kind {
        CHAT,
        PIPELINE
    }
}
