package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * API 配置实体
 * <p>
 * 对应数据库表：afv_api_config
 * 管理 AI 服务的 API 接入配置，支持多种平台。
 */
@TableName("afv_api_config")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ApiConfig extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 配置名称 */
    private String name;

    /** 平台标识：deepseek / dashscope / openai_compatible / gemini / ollama / anthropic / vertex_ai / GoogleFlowReverseApi */
    private String platform;

    /** API 类型：1-文本对话 2-图片生成 3-视频生成 */
    private Integer apiType;

    /** API 接口地址 */
    private String apiUrl;

    /**
     * 是否为 OpenAI 兼容请求自动补充 /v1 前缀。
     * 仅对 openai_compatible 平台生效。
     */
    @Builder.Default
    private Boolean autoAppendV1Path = true;

    /** API 密钥 */
    private String apiKey;

    /** 应用ID（部分平台需要） */
    private String appId;

    /** 应用密钥（部分平台需要） */
    private String appSecret;

    /** 关联模型ID */
    private Long modelId;

    /** 状态：0-禁用 1-启用 */
    @Builder.Default
    private Integer status = 1;

    /** 备注说明 */
    private String remark;
}
