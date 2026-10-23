package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BaseEntity;
import org.apache.ibatis.type.StringTypeHandler;
import lombok.*;

import java.util.Map;

/**
 * 生图任务实体
 * <p>
 * 对应数据库表：afv_image_task
 * 记录一次 AI 图片生成任务的完整信息，包括提示词、模型配置和生成参数。
 */
@TableName(value = "afv_image_task", autoResultMap = true)
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ImageTask extends BaseEntity {
    /**
     * 仅在可靠执行输入中保存；不进入旧业务表，也不改变普通请求的空值摘要。
     */
    @TableField(exist = false)
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private ObjectNode workflowInputs;

    @TableField(exist = false)
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Map<String, Integer> inputMediaVersions;

    /**
     * 主键ID，自增
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 经认证的执行上下文，仅供服务端工作者读取。
     */
    @JsonIgnore
    @TableField(typeHandler = StringTypeHandler.class, updateStrategy = FieldStrategy.NEVER)
    private String executionContextJson;

    /**
     * 任务唯一标识
     */
    private String taskId;

    /**
     * 发起用户ID
     */
    private Long userId;

    /**
     * 关联项目ID
     */
    private Long projectId;

    /**
     * 生图提示词
     */
    private String prompt;

    /**
     * 提示词模板ID
     */
    private Long promptTemplateId;

    /**
     * 参考图片URL列表 JSON
     */
    @TableField(typeHandler = StringTypeHandler.class, updateStrategy = FieldStrategy.NEVER)
    private String refImageUrls;

    /**
     * 画面比例，如 16:9、1:1
     */
    private String ratio;

    /**
     * 分辨率，如 1920x1080
     */
    private String resolution;

    /**
     * 宽高比描述
     */
    private String aspectRatio;

    /**
     * 图片宽度（像素）
     */
    private Integer width;

    /**
     * 图片高度（像素）
     */
    private Integer height;

    /**
     * 生成数量
     */
    @Builder.Default
    private Integer count = 1;

    /**
     * 已成功生成的数量
     */
    @Builder.Default
    private Integer successCount = 0;

    /**
     * 任务状态：0-排队中 1-处理中 2-已完成 3-失败
     */
    @Builder.Default
    private Integer status = 0;

    /**
     * 失败时的错误信息
     */
    private String errorMsg;

    /**
     * 使用的AI模型ID
     */
    private Long modelId;

    /**
     * 提交时固化的 ComfyUI 工作流版本ID
     */
    private Long workflowVersionId;

    /**
     * 任务分类标签
     */
    private String category;

    /**
     * 拥有者类型：1-个人 2-团队
     */
    private Integer ownerType;

    /**
     * 拥有者ID
     */
    private Long ownerId;
}
