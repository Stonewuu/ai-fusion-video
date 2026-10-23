package com.stonewu.fusion.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * 合成后配乐的调优参数（均为非敏感项）。
 * <p>
 * Sonilo 的接入地址、密钥与出站代理不在这里：在「设置 → AI 模型 → API 配置」中
 * 新建平台为 {@code sonilo} 的配置即开启合成后配乐，禁用或删除即关闭，无需重启。
 */
@Component
@ConfigurationProperties(prefix = "video.compose.bgm")
@Getter
@Setter
public class EpisodeBgmProperties {

    /**
     * 配乐风格提示（可留空，留空时按画面内容自动作曲）
     */
    private String musicPrompt = "";

    /**
     * 与成片原声 / 音效混音时配乐音轨的音量（0~1）
     */
    private double musicVolume = 0.7;

    /**
     * 是否同时生成音效音轨，默认关闭
     */
    private boolean sfxEnabled = false;

    /**
     * 任务轮询间隔（秒）
     */
    private int pollIntervalSeconds = 5;

    /**
     * 单个生成任务的等待上限（秒）
     */
    private int timeoutSeconds = 900;
}
