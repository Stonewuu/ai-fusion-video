package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.common.BusinessException;

import java.util.Map;

/** 兼容网关可明确要求 Bearer；原生平台继续使用各自 SDK 的鉴权约定。 */
final class GatewayAuthentication {
    private GatewayAuthentication() {
    }

    static boolean bearer(Map<String, Object> config) {
        Object mode = config.get("gatewayAuthMode");
        if (mode == null || "api_key".equals(mode)) {
            return false;
        }
        if ("bearer".equals(mode)) {
            return true;
        }
        throw new BusinessException("模型的网关鉴权方式无效");
    }
}
