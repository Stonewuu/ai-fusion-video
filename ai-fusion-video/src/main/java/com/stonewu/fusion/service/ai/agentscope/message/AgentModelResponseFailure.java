package com.stonewu.fusion.service.ai.agentscope.message;

/** 连接已建立后模型返回的失败事件，保留错误代码，不把 HTTP 200 改写为 HTTP 500。 */
public final class AgentModelResponseFailure extends RuntimeException {
    private final String code;

    public AgentModelResponseFailure(String code, String message) {
        super(message);
        this.code = code;
    }

    public String code() {
        return code;
    }
}
