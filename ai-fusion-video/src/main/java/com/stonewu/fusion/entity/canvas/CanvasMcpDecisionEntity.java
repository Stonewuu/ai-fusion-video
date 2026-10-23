package com.stonewu.fusion.entity.canvas;

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
@TableName("afv_canvas_mcp_decision")
public class CanvasMcpDecisionEntity {
    /**
     * 人工核实决定标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 被核实的原调用
     */
    private String jobId;

    /**
     * 核实人账号
     */
    private Long userId;

    /**
     * 同一核实的幂等请求标识
     */
    private String requestId;

    /**
     * 核实内容摘要
     */
    private String requestHash;

    /**
     * 采用已接收结果或确认原调用结束
     */
    private String action;

    /**
     * 核实依据
     */
    private String note;

    /**
     * 原核实回执
     */
    private String receiptJson;

    /**
     * 核实时间
     */
    private LocalDateTime createdAt;
}
