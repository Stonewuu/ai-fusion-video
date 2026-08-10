package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;

/** ComfyUI workflow identity and publication pointer. */
@TableName("afv_comfyui_workflow")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ComfyUiWorkflow extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long apiConfigId;

    private String name;

    private String code;

    private Integer modelType;

    private String description;

    private Long activeVersionId;

    @Builder.Default
    private Integer status = 1;

    @Builder.Default
    private Long deletedId = 0L;
}
