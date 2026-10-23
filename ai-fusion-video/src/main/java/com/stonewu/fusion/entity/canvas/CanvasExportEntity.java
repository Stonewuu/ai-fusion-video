package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
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
@TableName("afv_canvas_export")
public class CanvasExportEntity {
    /**
     * 交付任务编号
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 来源画布编号
     */
    private String canvasId;

    /**
     * 来源导出节点运行编号，手动导出为空
     */
    private String nodeRunId;

    /**
     * 发起用户编号
     */
    private Long actorUserId;

    /**
     * 发起团队作用域，零表示个人
     */
    private Long actorTeamId;

    /**
     * 发起作用域内幂等请求编号
     */
    private String requestId;

    /**
     * 请求内容摘要
     */
    private String requestHash;

    /**
     * 交付文件格式
     */
    private String format;

    /**
     * 交付任务状态
     */
    private String status;

    /**
     * 服务端确定的发起身份与运行来源
     */
    private String contextJson;

    /**
     * 不可变内容版本、图、媒体与来源清单
     */
    private String manifestJson;

    /**
     * 交付内容摘要
     */
    private String manifestHash;

    /**
     * 是否已请求取消
     */
    private Boolean cancelRequested;

    /**
     * 任务状态版本
     */
    private Long revision;

    /**
     * 实际构建尝试次数
     */
    private Integer attempts;

    /**
     * 当前工作者编号
     */
    private String workerId;

    /**
     * 工作者隔离代次
     */
    private Long ownerEpoch;

    /**
     * 当前工作者租约到期时间
     */
    private LocalDateTime leaseUntil;

    /**
     * 已发布加密文件编号
     */
    private String blobId;

    /**
     * 下载显示文件名
     */
    private String filename;

    /**
     * 下载内容类型
     */
    private String mimeType;

    /**
     * 明文文件字节数
     */
    private Long byteSize;

    /**
     * 明文文件内容摘要
     */
    private String contentHash;

    /**
     * 用户可读失败原因
     */
    private String lastError;

    /**
     * 创建时间
     */
    private LocalDateTime createTime;

    /**
     * 状态更新时间
     */
    private LocalDateTime updateTime;
}
