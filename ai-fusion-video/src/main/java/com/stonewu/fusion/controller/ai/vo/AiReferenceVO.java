package com.stonewu.fusion.controller.ai.vo;

import lombok.Data;
import lombok.experimental.Accessors;

/**
 * AI 引用数据 VO
 * <p>
 * 用于前端传递上下文引用（自动引用、手动引用）
 */
@Data
@Accessors(chain = true)
public class AiReferenceVO {

    /** 引用类型（如 script、storyboard、asset、episode 等） */
    private String type;

    /** 引用 ID */
    private Long id;

    /** 引用标题（用于展示和构建 context key） */
    private String title;

    /** 扩展元数据 JSON（如 fullText、startLine、endLine 等） */
    private String metadata;
}
