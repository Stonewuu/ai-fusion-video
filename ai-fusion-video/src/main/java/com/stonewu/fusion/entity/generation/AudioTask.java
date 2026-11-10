package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = true)
@TableName("afv_audio_generation_task")
public class AudioTask extends BaseEntity {
    @TableId(type = IdType.INPUT)
    private String id;
    private String requestId;
    private Long userId;
    private Long teamId;
    private Long modelId;
    private String modelName;
    private String modelSnapshotJson;
    private Long apiConfigId;
    private String apiRouteJson;
    private String inputJson;
    private String inputHash;
    private String state;
    private String outputJson;
    private String resultJson;
    private String errorMessage;
    private String leaseOwner;
    private LocalDateTime leaseUntil;
    private LocalDateTime nextPollAt;
    private LocalDateTime deadlineAt;
}
