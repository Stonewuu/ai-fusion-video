package com.stonewu.fusion.service.storyboard;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.EpisodeBgmProperties;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.sonilo.SoniloAudioClient;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * 配乐配置与纯混音命令；付费请求和媒体写入由持久化任务管理。
 */
@Service
@RequiredArgsConstructor
public class EpisodeBgmService {
    static final int MUSIC_MAX_SECONDS = 360;
    static final int SFX_MAX_SECONDS = 180;
    private final EpisodeBgmProperties properties;
    private final ApiConfigService apiConfigService;
    @Value("${video.compose.ffmpeg-path:ffmpeg}")
    private String ffmpegPath = "ffmpeg";

    @Cacheable(value = "episodeBgmConfig", key = "'enabled'", unless = "#result == null")
    public ApiConfig resolveApiConfig() {
        List<ApiConfig> configs = apiConfigService.getListByPlatform(SoniloAudioClient.PLATFORM);
        if (configs.isEmpty()) {
            return null;
        }
        if (configs.size() != 1) {
            throw new BusinessException("请只启用一个 Sonilo 配乐配置");
        }
        SoniloAudioClient.requireApiKey(configs.getFirst());
        return configs.getFirst();
    }

    public String musicSkipReason(double duration) {
        if (!Double.isFinite(duration) || duration <= 0) {
            return "无法确定成片视频时长";
        }
        return duration > MUSIC_MAX_SECONDS ? "本集时长超过配乐上限 6 分钟" : null;
    }

    public boolean sfxAllowed(double duration) {
        return duration > 0 && duration <= SFX_MAX_SECONDS;
    }

    public List<String> buildMixCommand(Path video, Path music, Path sfx, boolean original, double duration,
                                        Path output) {
        return buildMixCommand(video, music, sfx, original, duration, output, properties.getMusicVolume());
    }

    public List<String> buildMixCommand(Path video, Path music, Path sfx, boolean hasOriginalAudio,
                                        double durationSeconds, Path output, double volume) {
        if (!Double.isFinite(durationSeconds) || durationSeconds <= 0) {
            throw new BusinessException("混音必须提供有效的视频时长");
        }
        List<String> command = new ArrayList<>();
        command.add(getFfmpegExecutable());
        command.add("-y");
        command.add("-i");
        command.add(video.toString());
        command.add("-i");
        command.add(music.toString());
        if (sfx != null) {
            command.add("-i");
            command.add(sfx.toString());
        }

        if (!hasOriginalAudio && sfx == null) {
            command.add("-filter_complex");
            command.add("[1:a]volume=" + String.format(Locale.ROOT, "%.2f", volume)
                    + ",apad[aout]");
        } else {
            StringBuilder filter = new StringBuilder();
            filter.append("[1:a]volume=")
                    .append(String.format(Locale.ROOT, "%.2f", volume))
                    .append("[bgm];");
            int mixInputs = 1;
            if (hasOriginalAudio) {
                filter.append("[0:a]");
                mixInputs++;
            }
            filter.append("[bgm]");
            if (sfx != null) {
                filter.append("[2:a]");
                mixInputs++;
            }
            filter.append("amix=inputs=").append(mixInputs)
                    .append(":duration=longest:normalize=0,apad[aout]");
            command.add("-filter_complex");
            command.add(filter.toString());
        }

        command.add("-map");
        command.add("0:v:0");
        command.add("-map");
        command.add("[aout]");
        command.add("-c:v");
        command.add("copy");
        command.add("-c:a");
        command.add("aac");
        command.add("-t");
        command.add(String.format(Locale.ROOT, "%.3f", durationSeconds));
        command.add(output.toString());
        return command;
    }


    private String getFfmpegExecutable() {
        return ffmpegPath;
    }
}
