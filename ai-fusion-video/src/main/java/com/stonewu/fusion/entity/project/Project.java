package com.stonewu.fusion.entity.project;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.handler.JsonbTypeHandler;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 视频项目实体
 * <p>
 * 对应数据库表：afv_project
 * 项目是系统的核心组织单元，所有剧本、资产、分镜都隶属于某个项目。
 */
@TableName(value = "afv_project", autoResultMap = true)
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Project extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 项目名称 */
    private String name;

    /** 项目描述 */
    private String description;

    /** 项目封面图URL */
    private String coverUrl;

    /** 可见范围：1-公开 2-私有 3-仅团队可见 */
    @Builder.Default
    private Integer scope = 2;

    /** 拥有者类型：1-个人 2-团队 */
    private Integer ownerType;

    /** 拥有者ID，关联用户或团队 */
    private Long ownerId;

    /** 状态：0-筹备中 1-进行中 2-已完成 3-已归档 */
    @Builder.Default
    private Integer status = 0;

    /** 扩展配置 JSON */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String properties;

    /** 画风 key（预设 key 如 cartoon_3d 或 custom） */
    private String artStyle;

    /** 画风中文描述（自定义模式） */
    private String artStyleDescription;

    /** 画风英文提示词（自定义模式） */
    private String artStyleImagePrompt;

    /** 画风参考图路径（自定义模式，相对路径或完整 URL） */
    private String artStyleImageUrl;
}
