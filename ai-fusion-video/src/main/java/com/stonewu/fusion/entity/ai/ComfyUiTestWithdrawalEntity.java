package com.stonewu.fusion.entity.ai;

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
@TableName("afv_comfyui_test_withdrawal")
public class ComfyUiTestWithdrawalEntity {
    /**
     * 请求发起用户
     */
    private Long actorUserId;

    /**
     * 被撤回的试运行幂等标识
     */
    private String requestKey;

    /**
     * 撤回时核验的团队作用域
     */
    private Long actorTeamId;

    /**
     * 撤回时间（UTC）
     */
    private LocalDateTime createdAt;
}
