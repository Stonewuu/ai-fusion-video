package com.stonewu.fusion.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * 合成后配乐（Sonilo）配置。
 * <p>
 * 默认关闭；开启后按集合成完成时会把成片交给 Sonilo 生成配乐（可选音效），
 * 混入最终视频，并把每条音轨的 license_id 存档到分镜集。
 */
@Component
@ConfigurationProperties(prefix = "video.compose.bgm")
@Getter
@Setter
public class SoniloBgmProperties {

    /** 是否开启合成后配乐，默认关闭，不影响既有合成流程 */
    private boolean enabled = false;

    /** Sonilo API Key，通过环境变量 SONILO_API_KEY 注入 */
    private String apiKey = "";

    /** Sonilo API 地址 */
    private String baseUrl = "https://api.sonilo.com";

    /** 配乐风格提示（可留空，留空时按画面内容自动作曲） */
    private String musicPrompt = "";

    /** 与成片原声 / 音效混音时配乐音轨的音量（0~1） */
    private double musicVolume = 0.7;

    /** 是否同时生成音效音轨，默认关闭 */
    private boolean sfxEnabled = false;

    /** 任务轮询间隔（秒） */
    private int pollIntervalSeconds = 5;

    /** 单个生成任务的等待上限（秒） */
    private int timeoutSeconds = 900;
}
