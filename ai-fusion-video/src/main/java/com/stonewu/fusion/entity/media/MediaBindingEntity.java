package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_media_binding")
public class MediaBindingEntity {
    /**
     * 引用媒体的业务对象类型
     */
    private String resourceType;

    /**
     * 引用媒体的业务对象编号
     */
    private Long resourceId;

    /**
     * 明确登记的媒体字段路径
     */
    private String fieldName;

    /**
     * 已就绪的托管媒体标识
     */
    private String mediaId;

    /**
     * 建立此引用时的业务版本
     */
    private Integer resourceVersion;

    /**
     * 经过认证的绑定操作用户编号
     */
    private Long boundBy;

    /**
     * 绑定提交时间
     */
    private LocalDateTime boundAt;
}
