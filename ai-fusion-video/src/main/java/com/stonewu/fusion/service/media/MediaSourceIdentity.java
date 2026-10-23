package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.model.MediaFacts;
import com.stonewu.fusion.service.media.model.MediaSourceSnapshot;

import java.util.Map;

/**
 * 导入与执行读取共用同一身份摘要；展示名称不改变文件身份。
 */
public final class MediaSourceIdentity {
    private MediaSourceIdentity() {
    }

    public static String hash(CanvasJson json, MediaSourceSnapshot source, MediaFacts facts) {
        var identity = json.object(source);
        identity.remove("name");
        return json.hash(Map.of("source", identity, "content", facts));
    }
}
