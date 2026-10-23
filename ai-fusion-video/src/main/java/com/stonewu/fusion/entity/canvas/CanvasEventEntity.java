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
@TableName("afv_canvas_event")
public class CanvasEventEntity {
    /**
     * 事件唯一标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 画布标识
     */
    private String canvasId;

    /**
     * 画布内提交序号
     */
    private Long sequenceNo;

    /**
     * 关联命令标识
     */
    private String operationId;

    /**
     * 事件类型
     */
    private String eventType;

    /**
     * 已提交的事件数据
     */
    private String payloadJson;

    /**
     * 事件提交时间
     */
    private LocalDateTime createTime;
}
