package com.stonewu.fusion.entity.storyboard;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.handler.JsonbTypeHandler;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

import java.math.BigDecimal;

/**
 * 分镜条目实体
 * <p>
 * 对应数据库表：afv_storyboard_item
 * 存储分镜脚本中的单个镜头信息，包括画面内容、景别、时长、对白、音效、镜头运动等。
 */
@TableName(value = "afv_storyboard_item", autoResultMap = true)
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StoryboardItem extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属分镜ID */
    private Long storyboardId;

    /** 所属分镜集ID */
    private Long storyboardEpisodeId;

    /** 所属分镜场次ID */
    private Long storyboardSceneId;

    /** 排列顺序 */
    @Builder.Default
    private Integer sortOrder = 0;

    /** 镜号 */
    private String shotNumber;

    /** 自动编号（系统生成） */
    private String autoShotNumber;

    /** 用户上传的参考图片URL */
    private String imageUrl;

    /** 外部参考图片URL */
    private String referenceImageUrl;

    /** 视频URL（最终成品） */
    private String videoUrl;

    /** AI生成的图片URL */
    private String generatedImageUrl;

    /** 首帧参考图片URL */
    private String firstFrameImageUrl;

    /** 尾帧参考图片URL */
    private String lastFrameImageUrl;

    /** AI生成首帧时使用的提示词 */
    private String firstFramePrompt;

    /** AI生成尾帧时使用的提示词 */
    private String lastFramePrompt;

    /** AI生成的视频URL */
    private String generatedVideoUrl;

    /** AI生成视频时使用的提示词（保存以便复用和手动调整） */
    private String videoPrompt;

    /** 景别：远景/全景/中景/近景/特写 */
    private String shotType;

    /** 预估时长（秒） */
    private BigDecimal duration;

    /** 画面内容描述 */
    private String content;

    /** 画面期望描述（用于AI生图的提示引导） */
    private String sceneExpectation;

    /** 声音描述 */
    private String sound;

    /** 台词/旁白 */
    private String dialogue;

    /** 音效 */
    private String soundEffect;

    /** 配乐建议 */
    private String music;

    /** 镜头运动：推/拉/摇/移/跟/升/降 等 */
    private String cameraMovement;

    /** 镜头角度：平视/俯视/仰视 等 */
    private String cameraAngle;

    /** 摄像机装备 */
    private String cameraEquipment;

    /** 镜头焦段 */
    private String focalLength;

    /** 转场效果：切/淡入/淡出/溶/划 等 */
    private String transition;

    /** 出场角色子资产ID列表 JSON (List<Long> of AssetItem.id) */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String characterIds;

    /** 场景子资产ID (AssetItem.id) */
    private Long sceneAssetItemId;

    /** 道具子资产ID列表 JSON (List<Long> of AssetItem.id) */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String propIds;

    /** 备注 */
    private String remark;

    /** 自定义扩展数据 JSON */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String customData;

    /** 是否由AI生成 */
    @Builder.Default
    private Boolean aiGenerated = false;

    /** 状态：0-草稿 1-正常 */
    @Builder.Default
    private Integer status = 0;
}
