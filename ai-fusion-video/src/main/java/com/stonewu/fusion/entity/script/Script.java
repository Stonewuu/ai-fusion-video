package com.stonewu.fusion.entity.script;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.Version;
import com.stonewu.fusion.common.handler.JsonbTypeHandler;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 剧本实体
 * <p>
 * 对应数据库表：afv_script
 * 存储项目的总剧本信息，一个项目只能有一个总剧本。
 * 剧本可通过 AI 解析拆分为多集（ScriptEpisode）。
 */
@TableName(value = "afv_script", autoResultMap = true)
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Script extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属项目ID */
    private Long projectId;

    /** 剧本标题 */
    private String title;

    /** 剧本正文内容（格式化后） */
    private String content;

    /** 剧本原始内容（用户粘贴的原文） */
    private String rawContent;

    /** 总集数 */
    @Builder.Default
    private Integer totalEpisodes = 0;

    /** 故事梗概 */
    private String storySynopsis;

    /** 角色列表 JSON，存储从剧本中提取的角色信息 */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String charactersJson;

    /** 来源类型：0-手动创建 1-文件导入 2-AI生成 */
    @Builder.Default
    private Integer sourceType = 0;

    /** 解析状态：0-未解析 1-解析中 2-解析完成 3-解析失败 */
    @Builder.Default
    private Integer parsingStatus = 0;

    /** 解析进度描述 */
    private String parsingProgress;

    /** AI生成的剧本摘要 */
    private String summary;

    /** 剧本类型/题材 */
    private String genre;

    /** 目标受众 */
    private String targetAudience;

    /** 预估总时长（分钟） */
    private Integer durationEstimate;

    /** 可见范围：1-公开 2-私有 3-仅团队可见 */
    @Builder.Default
    private Integer scope = 3;

    /** 拥有者类型：1-个人 2-团队 */
    private Integer ownerType;

    /** 拥有者ID */
    private Long ownerId;

    /** 是否由AI生成 */
    @Builder.Default
    private Boolean aiGenerated = false;

    /** 乐观锁版本号 */
    @Version
    @Builder.Default
    private Integer version = 0;

    /** 状态：0-草稿 1-正常 */
    @Builder.Default
    private Integer status = 0;
}
