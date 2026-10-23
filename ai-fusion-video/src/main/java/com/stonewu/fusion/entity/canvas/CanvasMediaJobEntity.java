package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_canvas_media_job")
public class CanvasMediaJobEntity {
    /**
     * 共享媒体任务标识
     */
    @TableId(type = IdType.INPUT)
    private String jobId;

    /**
     * 对应节点运行
     */
    private String nodeRunId;

    /**
     * 媒体处理能力类型
     */
    private String operation;

    /**
     * 音频供应商配置编号，本地处理为空
     */
    private Long apiConfigId;

    /**
     * 冻结的供应商连接摘要
     */
    private String apiConfigHash;

    /**
     * 远端尚未确认终态时持续占用并发
     */
    private Boolean permitHeld;

    /**
     * 连续查询错误次数
     */
    private Integer consecutiveErrors;

    /**
     * 供应商确认的结果检查点
     */
    private String resultJson;
}
