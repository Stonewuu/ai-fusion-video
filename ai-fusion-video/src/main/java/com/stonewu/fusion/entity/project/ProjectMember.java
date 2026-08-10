package com.stonewu.fusion.entity.project;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 项目成员实体
 * <p>
 * 对应数据库表：afv_project_member
 * 管理项目的参与人员及其角色。
 */
@TableName("afv_project_member")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ProjectMember extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属项目ID */
    private Long projectId;

    /** 成员用户ID */
    private Long userId;

    /** 成员角色：1-项目拥有者 2-管理员 3-普通成员 */
    @Builder.Default
    private Integer role = 3;
}
