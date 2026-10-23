package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import com.baomidou.mybatisplus.annotation.TableLogic;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_canvas_node")
public class CanvasNodeEntity {
    /**
     * 节点唯一标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属画布标识
     */
    private String canvasId;

    /**
     * 节点类型
     */
    private String kind;

    /**
     * 节点协议版本
     */
    private Integer schemaVersion;

    /**
     * 节点名称
     */
    private String title;

    /**
     * 画布横坐标
     */
    private Double x;

    /**
     * 画布纵坐标
     */
    private Double y;

    /**
     * 指定展示宽度
     */
    private Double width;

    /**
     * 指定展示高度
     */
    private Double height;

    /**
     * 父分组节点标识
     */
    private String parentId;

    /**
     * 是否锁定位置
     */
    private Boolean locked;

    /**
     * 是否折叠
     */
    private Boolean collapsed;

    /**
     * 经过协议校验的节点配置
     */
    private String configJson;

    /**
     * 绑定的业务对象类型
     */
    private String resourceType;

    /**
     * 绑定的业务对象编号
     */
    private Long resourceId;

    /**
     * 最近确认的业务对象版本
     */
    private Long resourceVersion;

    /**
     * 配置与绑定版本
     */
    private Long configVersion;

    /**
     * 空间布局版本
     */
    private Long layoutVersion;

    /**
     * 是否已删除视图
     */
    @TableLogic
    private Boolean deleted;

    /**
     * 创建时间
     */
    private LocalDateTime createTime;

    /**
     * 更新时间
     */
    private LocalDateTime updateTime;
}
