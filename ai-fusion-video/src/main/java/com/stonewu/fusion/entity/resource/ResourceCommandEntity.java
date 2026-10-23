package com.stonewu.fusion.entity.resource;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_resource_command")
public class ResourceCommandEntity {
    /**
     * 业务命令标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 经过认证的操作用户编号
     */
    private Long actorUserId;

    /**
     * 用户作用域内的幂等标识
     */
    private String requestId;

    /**
     * 命令内容与执行作用域摘要
     */
    private String requestHash;

    /**
     * 命令来源
     */
    private String origin;

    /**
     * 发起命令的画布标识
     */
    private String canvasId;

    /**
     * 可信智能体运行标识
     */
    private String agentRunId;

    /**
     * 可信工具调用标识
     */
    private String toolCallId;

    /**
     * 已验证的业务命令内容
     */
    private String requestJson;

    /**
     * 已提交的幂等回执
     */
    private String resultJson;

    /**
     * 授权恢复所需的变更前值
     */
    private String undoJson;

    /**
     * 命令创建时间
     */
    private LocalDateTime createTime;

    /**
     * 命令完成时间
     */
    private LocalDateTime committedAt;
}
