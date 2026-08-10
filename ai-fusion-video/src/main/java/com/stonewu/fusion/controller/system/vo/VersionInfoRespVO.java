package com.stonewu.fusion.controller.system.vo;

import lombok.Data;

/**
 * 版本信息响应
 */
@Data
public class VersionInfoRespVO {

    /** 当前运行版本 */
    private String currentVersion;

    /** 当前版本展示文案 */
    private String currentVersionDisplay;

    /** 最新发布版本 */
    private String latestVersion;

    /** 最新版本展示文案 */
    private String latestVersionDisplay;

    /** GitHub 最新 release 版本 */
    private String latestReleaseVersion;

    /** GitHub 最新 release 展示文案 */
    private String latestReleaseVersionDisplay;

    /** GitHub 最新 release 链接 */
    private String latestReleaseUrl;

    /** GitHub 最新 release 发布时间（ISO-8601） */
    private String latestReleasePublishedAt;

    /** GitHub 最新 release 的 Docker 镜像是否已就绪 */
    private Boolean latestReleaseDockerReady;

    /** 是否有新版本可升级 */
    private Boolean updateAvailable;

    /** 当前环境是否允许与线上发布版本比较 */
    private Boolean comparisonEnabled;

    /** 是否为本地开发/快照构建 */
    private Boolean developmentBuild;

    /** 当前运行环境配置 */
    private String buildProfile;

    /** 版本关系：behind/same/ahead/incomparable */
    private String versionRelation;

    /** 发布详情链接 */
    private String releaseUrl;

    /** 对应 tag 链接 */
    private String tagUrl;

    /** 发布时间（ISO-8601） */
    private String publishedAt;

    /** 最近检查时间（ISO-8601） */
    private String checkedAt;

    /** 版本来源 */
    private String source;

    /** 检查是否成功 */
    private Boolean checkSucceeded;

    /** 检查失败时的说明 */
    private String message;
}