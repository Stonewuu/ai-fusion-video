package com.stonewu.fusion.controller.system.vo;

import lombok.Data;

/**
 * 当前运行版本响应
 */
@Data
public class RuntimeVersionInfoRespVO {

    /** 当前运行版本 */
    private String currentVersion;

    /** 当前版本展示文案 */
    private String currentVersionDisplay;

    /** 当前环境是否允许与线上发布版本比较 */
    private Boolean comparisonEnabled;

    /** 是否为本地开发/快照构建 */
    private Boolean developmentBuild;

    /** 当前运行环境配置 */
    private String buildProfile;

    /** 当前运行版本的说明 */
    private String message;
}