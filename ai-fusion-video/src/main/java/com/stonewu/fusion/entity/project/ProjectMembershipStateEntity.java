package com.stonewu.fusion.entity.project;

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
@TableName("afv_project_membership_state")
public class ProjectMembershipStateEntity {
    /**
     * 项目标识
     */
    @TableId(type = IdType.INPUT)
    private Long projectId;

    /**
     * 成员权限版本
     */
    private Long revision;
}
