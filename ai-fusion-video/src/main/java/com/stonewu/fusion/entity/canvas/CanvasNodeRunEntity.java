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
@TableName("afv_canvas_node_run")
public class CanvasNodeRunEntity {
    /**
     * 节点运行标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属画布运行
     */
    private String runId;

    /**
     * 原始节点标识，删除节点仍保留记录
     */
    private String nodeId;

    /**
     * 批量处理父运行
     */
    private String parentNodeRunId;

    /**
     * 冻结批量目标序号，普通节点为负一
     */
    private Integer itemIndex;

    /**
     * 明确发起的执行次数
     */
    private Integer attempt;

    /**
     * 计划中的稳定拓扑次序
     */
    private Integer ordinalNo;

    /**
     * 配置与上游输入版本摘要
     */
    private String fingerprint;

    /**
     * 节点配置、依赖与执行器快照
     */
    private String planJson;

    /**
     * 解析并冻结后的实际端口输入
     */
    private String inputJson;

    /**
     * 端口输出及精确资源版本
     */
    private String outputJson;

    /**
     * 可恢复阶段与子任务回执
     */
    private String checkpointJson;

    /**
     * 节点运行状态
     */
    private String status;

    /**
     * 失败、阻塞或待核实原因
     */
    private String errorMsg;

    /**
     * 关联的生成、媒体或智能体任务类型
     */
    private String externalKind;

    /**
     * 关联的持久化任务标识
     */
    private String externalId;

    /**
     * 当前执行实例
     */
    private String ownerInstanceId;

    /**
     * 工作者执行代数
     */
    private Long ownerEpoch;

    /**
     * 执行租约截止时间
     */
    private LocalDateTime leaseUntil;

    /**
     * 下次调度时间
     */
    private LocalDateTime nextRunAt;

    /**
     * 节点状态版本
     */
    private Long revision;

    /**
     * 节点运行创建时间
     */
    private LocalDateTime createdAt;

    /**
     * 节点状态修改时间
     */
    private LocalDateTime updatedAt;
}
