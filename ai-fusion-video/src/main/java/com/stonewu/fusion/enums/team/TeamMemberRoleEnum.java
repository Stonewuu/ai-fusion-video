package com.stonewu.fusion.enums.team;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 团队成员角色枚举
 */
@Getter
@AllArgsConstructor
public enum TeamMemberRoleEnum {

    OWNER(1, "创建者"),
    ADMIN(2, "管理员"),
    MEMBER(3, "普通成员");

    private final Integer role;
    private final String name;
}
