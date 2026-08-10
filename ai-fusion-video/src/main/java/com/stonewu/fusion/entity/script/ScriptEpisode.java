package com.stonewu.fusion.entity.script;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.Version;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 分集剧本实体
 * <p>
 * 对应数据库表：afv_script_episode
 * 存储剧本按集拆分后的每一集信息，每集可继续拆分为多个场次（SceneItem）。
 */
@TableName("afv_script_episode")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ScriptEpisode extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属剧本ID */
    private Long scriptId;

    /** 集号（从1开始） */
    private Integer episodeNumber;

    /** 本集标题 */
    private String title;

    /** 本集剧情梗概 */
    private String synopsis;

    /** 本集原始剧本内容 */
    private String rawContent;

    /** 预估时长（分钟） */
    private Integer durationEstimate;

    /** 本集总场次数 */
    @Builder.Default
    private Integer totalScenes = 0;

    /** 来源类型：0-AI解析 1-手动添加 */
    @Builder.Default
    private Integer sourceType = 0;

    /** 排列顺序 */
    @Builder.Default
    private Integer sortOrder = 0;

    /** 解析状态：0-未解析 1-解析中 2-解析完成 3-解析失败 */
    @Builder.Default
    private Integer parsingStatus = 0;

    /** 状态：0-草稿 1-正常 */
    @Builder.Default
    private Integer status = 0;

    /** 乐观锁版本号 */
    @Version
    @Builder.Default
    private Integer version = 0;
}
