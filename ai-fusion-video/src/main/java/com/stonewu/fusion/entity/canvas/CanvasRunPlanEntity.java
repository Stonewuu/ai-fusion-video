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
@TableName("afv_canvas_run_plan")
public class CanvasRunPlanEntity {
    /**
     * 运行计划标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属画布
     */
    private String canvasId;

    /**
     * 计划发起用户
     */
    private Long actorUserId;

    /**
     * 计划发起团队作用域
     */
    private Long actorTeamId;

    /**
     * 预览对应文档版本
     */
    private Long documentRevision;

    /**
     * 完整计划摘要
     */
    private String planHash;

    /**
     * 冻结的图、业务版本、输入、模型与预算
     */
    private String snapshotJson;

    /**
     * 计划确认截止时间
     */
    private LocalDateTime expiresAt;

    /**
     * 计划创建时间
     */
    private LocalDateTime createdAt;
}
