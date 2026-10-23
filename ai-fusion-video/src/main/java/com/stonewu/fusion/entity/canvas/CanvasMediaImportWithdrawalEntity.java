package com.stonewu.fusion.entity.canvas;

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
@TableName("afv_canvas_media_import_withdrawal")
public class CanvasMediaImportWithdrawalEntity {
    /**
     * 撤回请求的认证用户
     */
    private Long actorUserId;

    /**
     * 原导入请求标识
     */
    private String requestId;

    /**
     * 原请求所属画布
     */
    private String canvasId;

    /**
     * 撤回时固定的原团队作用域
     */
    private Long actorTeamId;

    /**
     * 撤回登记时间
     */
    private LocalDateTime createdAt;
}
