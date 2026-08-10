package com.stonewu.fusion.entity.team;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

import java.time.LocalDateTime;

/**
 * 团队成员实体
 * <p>
 * 对应数据库表：afv_team_member
 * 管理团队的参与人员及其角色。
 */
@TableName("afv_team_member")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TeamMember extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属团队ID */
    private Long teamId;

    /** 成员用户ID */
    private Long userId;

    /** 角色：1-创建者 2-管理员 3-普通成员 */
    private Integer role;

    /** 状态：0-禁用 1-启用 */
    @Builder.Default
    private Integer status = 1;

    /** 加入时间 */
    private LocalDateTime joinTime;
}
