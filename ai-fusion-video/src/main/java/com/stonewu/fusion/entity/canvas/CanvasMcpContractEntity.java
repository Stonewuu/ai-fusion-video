package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.TableLogic;

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
@TableName("afv_canvas_mcp_contract")
public class CanvasMcpContractEntity {
    /**
     * MCP 工作流契约标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 拥有该工具授权的用户
     */
    private Long userId;

    /**
     * 供画布选择的操作名称
     */
    private String name;

    /**
     * 远端工具名称
     */
    private String toolName;

    /**
     * 远端服务名称
     */
    private String serverName;

    /**
     * 不含凭据的工具定义与执行约定
     */
    private String definitionJson;

    /**
     * 工具定义及执行约定摘要
     */
    private String definitionHash;

    /**
     * 契约版本，已确认计划固定该版本
     */
    private Long version;

    /**
     * 是否允许新调用
     */
    private Boolean enabled;

    /**
     * 是否已移除目录条目，历史契约仍保留
     */
    @TableLogic
    private Boolean deleted;

    /**
     * 创建时间
     */
    private LocalDateTime createdAt;

    /**
     * 修改时间
     */
    private LocalDateTime updatedAt;
}
