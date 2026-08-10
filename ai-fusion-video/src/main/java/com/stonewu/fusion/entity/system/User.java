package com.stonewu.fusion.entity.system;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 用户实体
 * <p>
 * 对应数据库表：sys_user
 * 存储系统用户的基本信息和认证数据。
 */
@TableName("sys_user")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class User extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 登录用户名（唯一） */
    private String username;

    /** 登录密码（BCrypt 加密存储） */
    private String password;

    /** 用户昵称（显示名称） */
    private String nickname;

    /** 头像URL */
    private String avatar;

    /** 邮箱地址 */
    private String email;

    /** 手机号码 */
    private String phone;

    /** 状态：0-禁用 1-启用 */
    @Builder.Default
    private Integer status = 1;
}
