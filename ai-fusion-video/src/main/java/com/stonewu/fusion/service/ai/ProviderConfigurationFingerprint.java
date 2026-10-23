package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.canvas.CanvasJson;

import java.util.List;

/**
 * 只保存连接摘要；密钥、代理凭据与连接配置不复制进任务快照。
 */
public final class ProviderConfigurationFingerprint {
    private ProviderConfigurationFingerprint() {
    }

    public static String of(ApiConfig api, CanvasJson json) {
        var fields = json.object(api);
        fields.remove(List.of("name", "remark", "status", "createTime", "updateTime", "creator", "updater", "deleted",
                "modelId"));
        return json.hash(fields);
    }
}
