package com.stonewu.fusion.service.canvas.model;

import com.fasterxml.jackson.annotation.JsonUnwrapped;

/**
 * 列表显示项目名称，画布本身的版本与存储协议保持独立。
 */
public record CanvasListItem(@JsonUnwrapped CanvasDocument document, String projectName) {
}
