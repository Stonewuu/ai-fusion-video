package com.stonewu.fusion.service.ai.run.stream;

/** 一次运行的终态消息已经保存，提交后检查整条共享流的保留期限。 */
public record AgentStreamMessagesCommitted(String streamRunId) {
}
