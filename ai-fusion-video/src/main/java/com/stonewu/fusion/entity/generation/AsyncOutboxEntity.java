package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_async_outbox")
public class AsyncOutboxEntity {
    /**
     * 投递记录标识
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 事件幂等标识
     */
    private String eventKey;

    /**
     * 已登记投递类型
     */
    private String eventType;

    /**
     * 关联聚合标识
     */
    private String aggregateId;

    /**
     * 不含凭据的投递内容
     */
    private String payloadJson;

    /**
     * 待投递或已投递
     */
    private String state;

    /**
     * 投递尝试次数
     */
    private Integer attempts;

    /**
     * 下次投递时间（UTC）
     */
    private LocalDateTime nextAttemptAt;

    /**
     * 最近投递错误
     */
    private String lastError;

    /**
     * 与业务状态一起提交的创建时间（UTC）
     */
    private LocalDateTime createdAt;

    /**
     * 投递确认时间（UTC）
     */
    private LocalDateTime deliveredAt;
}
