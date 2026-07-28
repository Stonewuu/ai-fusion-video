package com.stonewu.fusion.entity.system;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 系统配置实体
 * <p>
 * 对应数据库表：afv_system_config
 * 用于存储全局系统配置，如站点和后端资源公网地址等。
 */
@TableName("afv_system_config")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SystemConfig extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    /** 配置键 */
    private String configKey;

    /** 配置值 */
    private String configValue;

    /** 备注 */
    private String remark;
}
