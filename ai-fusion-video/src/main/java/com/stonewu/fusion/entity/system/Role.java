package com.stonewu.fusion.entity.system;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 系统角色实体
 * <p>
 * 对应数据库表：sys_role
 * 定义系统中的权限角色，如管理员、普通用户等。
 */
@TableName("sys_role")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Role extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 角色名称 */
    private String name;

    /** 角色代码标识，如 admin、user */
    private String code;

    /** 排列顺序 */
    @Builder.Default
    private Integer sort = 0;

    /** 状态：0-禁用 1-启用 */
    @Builder.Default
    private Integer status = 1;

    /** 备注说明 */
    private String remark;
}
