package com.stonewu.fusion.entity.storyboard;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.handler.JsonbTypeHandler;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 分镜脚本实体
 * <p>
 * 对应数据库表：afv_storyboard
 * 存储项目唯一的分镜容器，每个分镜下包含多个分镜分集、场次和条目（StoryboardItem）。
 */
@TableName(value = "afv_storyboard", autoResultMap = true)
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Storyboard extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属项目ID */
    private Long projectId;

    /** 关联剧本ID */
    private Long scriptId;

    /** 分镜标题 */
    private String title;

    /** 分镜描述 */
    private String description;

    /** 自定义列配置 JSON，用于扩展分镜表格的列定义 */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String customColumns;

    /** 可见范围：1-公开 2-私有 3-仅团队可见 */
    @Builder.Default
    private Integer scope = 3;

    /** 拥有者类型：1-个人 2-团队 */
    private Integer ownerType;

    /** 拥有者ID */
    private Long ownerId;

    /** 预估总时长（秒） */
    private Integer totalDuration;

    /** 状态：0-草稿 1-正常 */
    @Builder.Default
    private Integer status = 0;
}
