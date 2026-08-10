package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.FieldStrategy;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.handler.JsonbTypeHandler;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 生视频任务实体
 * <p>
 * 对应数据库表：afv_video_task
 * 记录一次 AI 视频生成任务的完整信息，包括提示词、参考图、生成模式等。
 */
@TableName(value = "afv_video_task", autoResultMap = true)
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VideoTask extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 任务唯一标识 */
    private String taskId;

    /** 发起用户ID */
    private Long userId;

    /** 关联项目ID */
    private Long projectId;

    /** 生视频提示词 */
    private String prompt;

    /** 提示词模板ID */
    private Long promptTemplateId;

    /** 生成模式：text2video-文本生视频 / image2video-图片生视频 */
    private String generateMode;

    /** 首帧参考图片URL（图生视频模式） */
    @TableField(updateStrategy = FieldStrategy.NEVER)
    private String firstFrameImageUrl;

    /** 尾帧参考图片URL（图生视频模式） */
    @TableField(updateStrategy = FieldStrategy.NEVER)
    private String lastFrameImageUrl;

    /** 参考图片URL列表 JSON */
    @TableField(typeHandler = JsonbTypeHandler.class, updateStrategy = FieldStrategy.NEVER)
    private String referenceImageUrls;

    /** 参考视频URL列表 JSON（多模态参考 / 编辑视频 / 延长视频） */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String referenceVideoUrls;

    /** 参考音频URL列表 JSON（有声视频生成） */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String referenceAudioUrls;

    /** 画面比例，如 16:9 */
    private String ratio;

    /** 分辨率，如 1920x1080 */
    private String resolution;

    /** 视频时长（秒） */
    private Integer duration;

    /** 是否添加水印，null 表示未显式指定 */
    private Boolean watermark;

    /** 是否生成配音，null 表示未显式指定 */
    private Boolean generateAudio;

    /** 随机种子，用于复现相同结果 */
    private Long seed;

    /** 是否固定镜头（不做运动），null 表示未显式指定 */
    private Boolean cameraFixed;

    /** 生成数量 */
    @Builder.Default
    private Integer count = 1;

    /** 已成功生成的数量 */
    @Builder.Default
    private Integer successCount = 0;

    /** 任务状态：0-排队中 1-处理中 2-已完成 3-失败 */
    @Builder.Default
    private Integer status = 0;

    /** 失败时的错误信息 */
    private String errorMsg;

    /** 使用的AI模型ID */
    private Long modelId;

    /** 提交时固化的 ComfyUI 工作流版本ID */
    private Long workflowVersionId;

    /** 任务分类标签 */
    private String category;

    /** 拥有者类型：1-个人 2-团队 */
    private Integer ownerType;

    /** 拥有者ID */
    private Long ownerId;
}
