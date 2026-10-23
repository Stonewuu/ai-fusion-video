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
@TableName("afv_canvas_value_snapshot")
public class CanvasValueSnapshotEntity {
    /**
     * 固定结果快照标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 快照所属画布标识
     */
    private String canvasId;

    /**
     * 副本中对应的来源节点标识
     */
    private String nodeId;

    /**
     * 固定的输出端口
     */
    private String portId;

    /**
     * 快照协议版本
     */
    private Integer schemaVersion;

    /**
     * 经权限校验冻结的完整类型化结果，不含访问票据或密钥
     */
    private String valueJson;

    /**
     * 原始画布标识，模板导入时为空
     */
    private String sourceCanvasId;

    /**
     * 原始节点运行标识
     */
    private String sourceNodeRunId;

    /**
     * 原始结果标识
     */
    private String sourceOutputId;

    /**
     * 复制前的固定结果快照标识
     */
    private String sourceSnapshotId;

    /**
     * 创建快照的用户标识
     */
    private Long createdBy;

    /**
     * 创建时间
     */
    private LocalDateTime createdAt;

    /**
     * 固定结果来源：画布复制或模板导入
     */
    private String sourceKind;

    /**
     * 导入模板的内容摘要
     */
    private String templateHash;
}
