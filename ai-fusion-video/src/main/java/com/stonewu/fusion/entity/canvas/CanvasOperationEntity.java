package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.TableField;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_canvas_operation")
public class CanvasOperationEntity {
    /**
     * 命令提交标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 画布标识
     */
    private String canvasId;

    /**
     * 经过认证的操作用户编号
     */
    private Long actorUserId;

    /**
     * 请求幂等标识
     */
    private String requestId;

    /**
     * 请求规范化摘要
     */
    private String requestHash;

    /**
     * 操作来源
     */
    private String origin;

    /**
     * 关联智能体运行标识
     */
    private String agentRunId;

    /**
     * 关联工具调用标识
     */
    private String toolCallId;

    /**
     * 提交后的文档版本
     */
    private Long committedRevision;

    /**
     * 已验证的命令内容
     */
    private String requestJson;

    /**
     * 幂等提交回执
     */
    private String resultJson;

    /**
     * 有条件撤销所需的变更前值
     */
    private String undoJson;

    /**
     * 提交时间
     */
    private LocalDateTime createTime;
    @TableField(exist = false)
    private String eventType;
    @TableField(exist = false)
    private String actorName;
}
