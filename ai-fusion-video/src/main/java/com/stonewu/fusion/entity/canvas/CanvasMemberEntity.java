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
@TableName("afv_canvas_member")
public class CanvasMemberEntity {
    /**
     * 画布标识
     */
    private String canvasId;

    /**
     * 成员用户编号
     */
    private Long userId;

    /**
     * 是否可编辑画布
     */
    private Boolean canEdit;

    /**
     * 是否可执行创作任务
     */
    private Boolean canExecute;

    /**
     * 是否可管理共享与归档
     */
    private Boolean canManage;

    /**
     * 授权用户编号
     */
    private Long grantedBy;

    /**
     * 授权时间
     */
    private LocalDateTime createTime;

    /**
     * 更新时间
     */
    private LocalDateTime updateTime;
}
