package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_canvas_agent_node_job")
public class CanvasAgentNodeJobEntity {
    /**
     * 画布节点执行标识
     */
    @TableId(type = IdType.INPUT)
    private String nodeRunId;

    /**
     * 画布运行标识
     */
    private String canvasRunId;

    /**
     * 专业智能体根运行标识
     */
    private String agentRunId;

    /**
     * 专业智能体会话标识
     */
    private String conversationId;

    /**
     * 固定的智能体内核快照
     */
    private String snapshotJson;

    /**
     * 固定的结构化输入与提示内容
     */
    private String inputJson;

    /**
     * 输入资源范围与当前受保护版本
     */
    private String scopeJson;

    /**
     * 启动投递状态
     */
    private String dispatchState;

    /**
     * 是否请求停止
     */
    private Boolean cancelRequested;

    /**
     * 启动错误说明
     */
    private String errorMessage;

    /**
     * 受理时间
     */
    private LocalDateTime createdAt;

    /**
     * 最后更新时间
     */
    private LocalDateTime updatedAt;
}
