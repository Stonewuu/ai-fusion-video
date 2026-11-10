package com.stonewu.fusion.service.ai.google;

import cn.hutool.json.JSONObject;

public final class GoogleRequestErrors {
    private GoogleRequestErrors() {
    }

    static String message(int status, String reason) {
        if (reason != null && reason.contains("API keys are not supported")) {
            return "此 Google 接口需要服务账号，请切换鉴权方式";
        }
        if (status == 403 && reason != null && reason.contains("aiplatform.interactions.")) {
            return "此模型需要 Interactions 权限，请联系 Google Cloud 项目管理员开通";
        }
        return switch (status) {
            case 401 -> "Google 凭据未通过验证，请检查 API 配置";
            case 403 -> "Google 拒绝了模型调用，请检查项目权限与结算状态";
            case 404 -> "模型在当前区域不可用，请检查模型 ID 和区域";
            case 429 -> "Google 配额不足，请检查配额或稍后重试";
            case 400 -> "Google 未接受当前输入，请检查模型参数和参考素材";
            default -> "Google 服务暂时不可用，请稍后重试";
        };
    }

    public static String operationMessage(JSONObject error) {
        int status = switch (error.getInt("code", 13)) {
            case 3 -> 400;
            case 5 -> 404;
            case 7 -> 403;
            case 8 -> 429;
            case 16 -> 401;
            default -> 503;
        };
        if (status == 429 && error.getStr("message", "").contains("high load")) {
            return "Google 视频服务繁忙，请稍后重试";
        }
        return message(status, error.getStr("message", ""));
    }
}
