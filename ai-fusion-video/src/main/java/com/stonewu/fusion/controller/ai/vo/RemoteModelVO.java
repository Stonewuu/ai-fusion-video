package com.stonewu.fusion.controller.ai.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 远程 API 返回的可用模型信息
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RemoteModelVO {

    /** 模型 ID（即模型 code） */
    private String id;

    /** 建议展示名称 */
    private String displayName;

    /** 模型拥有者 */
    private String ownedBy;

    /** 远程来源平台 */
    private String providerPlatform;

    /** 模型类型：1-对话 2-图像生成 3-视频生成 */
    private Integer modelType;

    /** 可直接应用的模型能力预设代码 */
    private String capabilityPresetCode;

    /** 模型协议 */
    private String modelProtocol;

    /** 是否由系统推断补全元数据 */
    private Boolean inferredMetadata;
}
