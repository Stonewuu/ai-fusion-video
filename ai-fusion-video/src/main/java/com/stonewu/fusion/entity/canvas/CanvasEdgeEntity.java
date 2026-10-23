package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.FieldStrategy;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
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
@TableName("afv_canvas_edge")
public class CanvasEdgeEntity {
    /**
     * 连线唯一标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属画布标识
     */
    private String canvasId;

    /**
     * 关系类型：执行、业务引用或注释
     */
    private String kind;

    /**
     * 来源节点标识
     */
    private String sourceNodeId;

    /**
     * 来源输出端口
     */
    private String sourcePort;

    /**
     * 目标节点标识
     */
    private String targetNodeId;

    /**
     * 目标输入端口
     */
    private String targetPort;

    /**
     * 目标端口中的显式输入顺序
     */
    private Integer inputIndex;

    /**
     * 固定的运行与输出资源引用
     */
    private String outputJson;

    /**
     * 连线版本
     */
    private Long version;

    /**
     * 是否已删除连线
     */
    private Boolean deleted;

    /**
     * 用于约束有效输入位置的目标节点
     */
    @TableField(insertStrategy = FieldStrategy.NEVER, updateStrategy = FieldStrategy.NEVER)
    private String activeTargetNodeId;

    /**
     * 创建时间
     */
    private LocalDateTime createTime;

    /**
     * 更新时间
     */
    private LocalDateTime updateTime;
}
