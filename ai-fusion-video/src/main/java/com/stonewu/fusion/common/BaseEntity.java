package com.stonewu.fusion.common;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableLogic;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * MyBatis-Plus 基础实体类
 * <p>
 * 所有业务实体的父类，提供统一的时间戳和逻辑删除字段。
 * 注意：id 字段由各子类自行定义，以兼容 Lombok @Builder。
 */
@Data
public abstract class BaseEntity {

    /** 创建时间，插入时自动填充 */
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    /** 更新时间，插入和更新时自动填充 */
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;

    /** 逻辑删除标志：false-未删除 true-已删除 */
    @TableLogic
    @TableField("deleted")
    private Boolean deleted;
}

