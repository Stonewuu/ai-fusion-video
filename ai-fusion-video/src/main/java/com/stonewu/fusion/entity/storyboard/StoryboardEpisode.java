package com.stonewu.fusion.entity.storyboard;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

import java.time.LocalDateTime;

/**
 * 分镜集实体
 * <p>
 * 对应数据库表：afv_storyboard_episode
 * 存储分镜脚本按集拆分后的每一集信息，每集可继续拆分为多个分镜场次。
 */
@TableName("afv_storyboard_episode")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StoryboardEpisode extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属分镜ID */
    private Long storyboardId;

    /** 关联的剧本分集ID */
    private Long scriptEpisodeId;

    /** 集号（从1开始） */
    private Integer episodeNumber;

    /** 本集标题 */
    private String title;

    /** 本集梗概 */
    private String synopsis;

    /** 排列顺序 */
    @Builder.Default
    private Integer sortOrder = 0;

    /** 状态：0-草稿 1-正常 */
    @Builder.Default
    private Integer status = 0;

    /**
     * 逻辑删除隔离标识。
     * 0 表示未删除；逻辑删除后写入当前记录 ID，用于避免唯一索引被已删除数据占用。
     */
    @Builder.Default
    private Long deletedId = 0L;

    /** 本集合成视频URL */
    private String composedVideoUrl;

    /** 合成状态: 0未开始 1合成中 2已完成 3失败 */
    @Builder.Default
    private Integer composeStatus = 0;

    /** 合成失败原因 */
    private String composeErrorMsg;

    /** 合成完成时间 */
    private LocalDateTime composedAt;

    /** 配乐音轨URL（持久化存档） */
    private String bgmAudioUrl;

    /** 配乐音轨授权凭证ID（商用留档） */
    private String bgmLicenseId;

    /** 音效音轨URL（持久化存档） */
    private String bgmSfxAudioUrl;

    /** 音效音轨授权凭证ID（商用留档） */
    private String bgmSfxLicenseId;

    /** 配乐失败原因（配乐失败不影响本集合成结果） */
    private String bgmErrorMsg;
}
