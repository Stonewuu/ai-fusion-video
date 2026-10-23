package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * 在付费提交前保存可用于对账的客户端标识和请求；不允许包含凭据。
 */
public record GenerationPreparedRequest(String providerTaskId, ObjectNode payload) {
    public static GenerationPreparedRequest ordinary() {
        return new GenerationPreparedRequest(null, null);
    }
}
