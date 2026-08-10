package com.stonewu.fusion.entity.storyboard;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 分镜场次实体
 * <p>
 * 对应数据库表：afv_storyboard_scene
 * 存储分镜集中的单个场次信息，每个场次下包含多个分镜条目（StoryboardItem）。
 */
@TableName("afv_storyboard_scene")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StoryboardScene extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属分镜集ID */
    private Long episodeId;

    /** 所属分镜ID（冗余字段，方便查询） */
    private Long storyboardId;

    /** 场次编号，如 "1-1" */
    private String sceneNumber;

    /** 场景标头，如 "内景 客厅 夜" */
    private String sceneHeading;

    /** 场景地点 */
    private String location;

    /** 时间段：日/夜/黄昏/清晨 等 */
    private String timeOfDay;

    /** 内外景标识：内景/外景/内外景 */
    private String intExt;

    /** 排列顺序 */
    @Builder.Default
    private Integer sortOrder = 0;

    /** 状态：0-草稿 1-正常 */
    @Builder.Default
    private Integer status = 0;
}
