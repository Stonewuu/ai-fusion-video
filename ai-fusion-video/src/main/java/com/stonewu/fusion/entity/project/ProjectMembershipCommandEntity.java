package com.stonewu.fusion.entity.project;

import com.baomidou.mybatisplus.annotation.TableField;

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
@TableName("afv_project_membership_command")
public class ProjectMembershipCommandEntity {
    /**
     * 权限操作标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 项目标识
     */
    private Long projectId;

    /**
     * 提交后的成员权限版本
     */
    private Long revision;

    /**
     * 操作用户标识
     */
    private Long actorUserId;

    /**
     * 操作时团队标识
     */
    private Long actorTeamId;

    /**
     * 幂等请求标识
     */
    private String requestId;

    /**
     * 请求内容摘要
     */
    private String requestHash;

    /**
     * 被调整的成员标识
     */
    private Long userId;

    /**
     * 调整前的单独项目角色
     */
    private Integer previousRole;

    /**
     * 调整后的单独项目角色，空值表示移除
     */
    private Integer currentRole;

    /**
     * 随成员移除撤销的画布单独授权
     */
    private String removedCanvasGrants;

    /**
     * 是否已通知项目画布
     */
    private Boolean published;

    /**
     * 权限变更时间
     */
    private LocalDateTime createTime;
    @TableField(exist = false)
    private String actorUsername;
    @TableField(exist = false)
    private String actorName;
    @TableField(exist = false)
    private String memberUsername;
    @TableField(exist = false)
    private String memberName;
}
