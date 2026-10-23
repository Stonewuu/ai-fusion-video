package com.stonewu.fusion.service.canvas.run.media;

import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;

public final class CanvasMediaRules {
    private CanvasMediaRules() {
    }

    public static void config(CanvasNode node) {
        var config = node.config();
        if (node.kind() == CanvasNodeKind.IMAGE_TRANSFORM) {
            int width = config.path("width").asInt(), height = config.path("height").asInt();
            if (width <= 0 || height <= 0 || (long) width * height > 40_000_000) {
                throw CanvasRunException.invalid("请填写输出宽高，图片像素总数不能超过 4000 万");
            }
        }
        if (node.kind() == CanvasNodeKind.VIDEO_COMPOSE && config.has("resolution")
                && !config.path("resolution").asText().matches("[1-9][0-9]{1,3}x[1-9][0-9]{1,3}")) {
            throw CanvasRunException.invalid("合成分辨率请使用宽x高，例如 1920x1080");
        }
    }

    public static void inputs(CanvasNodeKind operation, Ports inputs) {
        if (operation == CanvasNodeKind.VIDEO_COMPOSE) {
            boolean clips = !inputs.at("videos").isEmpty(), episode = !inputs.at("episode").isEmpty();
            if (clips == episode) {
                throw CanvasRunException.invalid("合成视频必须选择有序视频列表或一集分镜，不能同时连接或全部留空");
            }
        }
    }
}
