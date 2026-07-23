package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * AI 模型实体
 * <p>
 * 对应数据库表：afv_ai_model
 * 管理系统中可用的 AI 模型配置，包括文本模型和图片模型。
 */
@TableName("afv_ai_model")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AiModel extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 模型显示名称 */
    private String name;

    /** 模型代码标识，如 deepseek-chat、qwen-vl-max */
    private String code;

    /** 模型级请求协议覆盖；为空时继承 API 配置中对应能力类型的默认协议 */
    private String modelProtocol;

    /** 模型能力预设代码；为空表示使用自定义能力配置 */
    private String capabilityPresetCode;

    /** 模型类型：1-文本对话 2-图片生成 3-视频生成 */
    private Integer modelType;

    /** 模型图标URL */
    private String icon;

    /** 模型描述说明 */
    private String description;

    /** 排列顺序 */
    @Builder.Default
    private Integer sort = 0;

    /** 状态：0-禁用 1-启用 */
    @Builder.Default
    private Integer status = 1;

    /** 模型特定配置 JSON，如 temperature、top_p 等参数 */
    private String config;

    /** 最大并发请求数 */
    @Builder.Default
    private Integer maxConcurrency = 5;

    /** 关联的 API 配置ID，指向 ApiConfig */
    private Long apiConfigId;

    /** 是否为默认模型 */
    @Builder.Default
    private Boolean defaultModel = false;

    // ========== 模型能力标识 ==========

    /** 是否支持视觉理解（传图片） */
    @Builder.Default
    private Boolean supportVision = false;

    /** 是否支持深度思考（reasoning） */
    @Builder.Default
    private Boolean supportReasoning = false;

    /** 上下文窗口大小（token 数） */
    private Integer contextWindow;

    /**
     * 逻辑删除隔离标识。
     * 0 表示未删除；逻辑删除后写入当前记录 ID，用于避免唯一索引被已删除数据占用。
     */
    @Builder.Default
    private Long deletedId = 0L;
}
