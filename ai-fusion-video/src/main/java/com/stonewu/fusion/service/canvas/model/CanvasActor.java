package com.stonewu.fusion.service.canvas.model;

/**
 * 由认证边界构造，异步任务保留同一团队作用域；不得从画布请求体反序列化。
 */
public record CanvasActor(long userId, Long teamId) {
    public CanvasActor {
        if (userId <= 0 || (teamId != null && teamId <= 0)) {
            throw new IllegalArgumentException("画布执行身份无效");
        }
    }
}
