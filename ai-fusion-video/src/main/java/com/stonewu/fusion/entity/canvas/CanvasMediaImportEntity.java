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
@TableName("afv_canvas_media_import")
public class CanvasMediaImportEntity {
    /**
     * 共享媒体导入任务标识
     */
    @TableId(type = IdType.INPUT)
    private String jobId;

    /**
     * 发起导入的画布标识
     */
    private String canvasId;

    /**
     * 发起时固定的团队作用域
     */
    private Long actorTeamId;

    /**
     * 工作流内部导入对应的节点运行，手工导入为空
     */
    private String nodeRunId;

    /**
     * 明确请求的画布落点及自动放入意图
     */
    private String placementJson;

    /**
     * 导入批次创建时间
     */
    private LocalDateTime createdAt;
}
