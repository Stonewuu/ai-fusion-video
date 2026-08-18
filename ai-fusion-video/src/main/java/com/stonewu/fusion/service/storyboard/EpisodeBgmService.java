package com.stonewu.fusion.service.storyboard;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.SoniloBgmProperties;
import com.stonewu.fusion.service.ai.sonilo.SoniloAudioClient;
import com.stonewu.fusion.service.ai.sonilo.SoniloAudioTrack;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storyboard.dto.EpisodeBgmOutcome;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * 合成后配乐服务（默认关闭）。
 * <p>
 * 在本集视频拼接完成后、持久化之前执行：把成片交给 Sonilo 按画面内容生成配乐
 * （时长与成片自动对齐，无提示词也可用；可选同时生成音效音轨），
 * 下载音轨并持久化留档，再用 ffmpeg 把音轨混入成片（视频流零转码）。
 * <p>
 * 配乐音乐自带授权、可商用（以 Sonilo 条款为准），每条音轨返回 license_id，
 * 随分镜集一并落库，便于商用留档与审计；单独的音效音轨为免版税。
 */
@Service
@Slf4j
public class EpisodeBgmService {

    /** 配乐接口的视频时长上限（秒） */
    public static final int MUSIC_MAX_SECONDS = 360;

    /** 音效接口的视频时长上限（秒） */
    public static final int SFX_MAX_SECONDS = 180;

    private static final int MUX_TIMEOUT_MINUTES = 15;
    private static final int PROBE_TIMEOUT_SECONDS = 30;

    private final SoniloBgmProperties properties;
    private final SoniloAudioClient soniloAudioClient;
    private final MediaStorageService mediaStorageService;

    @Value("${video.compose.ffmpeg-path:ffmpeg}")
    private String ffmpegPath;

    @Value("${video.compose.ffprobe-path:ffprobe}")
    private String ffprobePath;

    public EpisodeBgmService(SoniloBgmProperties properties,
                             SoniloAudioClient soniloAudioClient,
                             MediaStorageService mediaStorageService) {
        this.properties = properties;
        this.soniloAudioClient = soniloAudioClient;
        this.mediaStorageService = mediaStorageService;
    }

    public boolean isEnabled() {
        return properties.isEnabled();
    }

    /**
     * 对合成完成的成片执行配乐。
     *
     * @param composedVideo 拼接完成的本地成片文件
     * @param workDir       本次合成的临时目录（音轨与混音产物都写在这里，随合成流程统一清理）
     * @param progress      进度回调，透传到任务流
     * @return 配乐结果；配乐失败时抛出异常，由调用方决定是否降级
     */
    public EpisodeBgmOutcome apply(Path composedVideo, Path workDir, Consumer<String> progress) {
        if (StrUtil.isBlank(properties.getApiKey())) {
            throw new BusinessException("已开启合成后配乐但未配置 SONILO_API_KEY，"
                    + "请在 sonilo.com 获取 API Key 后通过环境变量注入");
        }

        double durationSeconds = probeDurationSeconds(composedVideo);
        String musicSkipReason = musicSkipReason(durationSeconds);
        if (musicSkipReason != null) {
            return EpisodeBgmOutcome.skipped(musicSkipReason);
        }

        progress.accept("配乐生成中：正在上传成片到 Sonilo…");
        String musicTaskId = soniloAudioClient.submitMusicTask(composedVideo, properties.getMusicPrompt());
        progress.accept("配乐任务已受理（task_id: " + musicTaskId + "），等待生成…");
        JSONObject musicTask = soniloAudioClient.awaitTask(musicTaskId);
        SoniloAudioTrack musicTrack = soniloAudioClient.extractPrimaryTrack(musicTask, musicTaskId);

        Path musicFile = workDir.resolve("bgm_music" + audioExtension(musicTrack.url()));
        soniloAudioClient.downloadAudio(musicTrack.url(), musicFile);
        String musicStoredUrl = mediaStorageService.storeFile(
                musicFile, "audio/bgm", trimDot(audioExtension(musicTrack.url())));

        Path sfxFile = null;
        String sfxStoredUrl = null;
        String sfxLicenseId = null;
        if (properties.isSfxEnabled()) {
            if (sfxAllowed(durationSeconds)) {
                // 音效是可选加分项：失败只记录原因，保留已生成的配乐
                try {
                    progress.accept("音效生成中：已提交 Sonilo 音效任务…");
                    String sfxTaskId = soniloAudioClient.submitSfxTask(composedVideo, null);
                    JSONObject sfxTask = soniloAudioClient.awaitTask(sfxTaskId);
                    SoniloAudioTrack sfxTrack = soniloAudioClient.extractPrimaryTrack(sfxTask, sfxTaskId);
                    sfxFile = workDir.resolve("bgm_sfx" + audioExtension(sfxTrack.url()));
                    soniloAudioClient.downloadAudio(sfxTrack.url(), sfxFile);
                    sfxStoredUrl = mediaStorageService.storeFile(
                            sfxFile, "audio/bgm", trimDot(audioExtension(sfxTrack.url())));
                    sfxLicenseId = sfxTrack.licenseId();
                } catch (Exception e) {
                    sfxFile = null;
                    sfxStoredUrl = null;
                    sfxLicenseId = null;
                    log.error("[EpisodeBgm] 音效生成失败，本次仅合入配乐", e);
                    progress.accept("音效生成失败：" + e.getMessage() + "，本次仅合入配乐");
                }
            } else {
                progress.accept("本集时长超过音效上限 " + SFX_MAX_SECONDS / 60 + " 分钟，本次仅生成配乐");
            }
        }

        progress.accept("配乐已生成，正在混入成片…");
        Path scoredVideo = workDir.resolve("output_bgm.mp4");
        boolean hasOriginalAudio = hasAudioStream(composedVideo);
        List<String> command = buildMixCommand(composedVideo, musicFile, sfxFile, hasOriginalAudio, scoredVideo);
        if (!runFfmpeg(command)) {
            throw new BusinessException("配乐混音失败（ffmpeg），本集视频保持无配乐版本");
        }

        return new EpisodeBgmOutcome(
                scoredVideo,
                musicStoredUrl,
                musicTrack.licenseId(),
                sfxStoredUrl,
                sfxLicenseId,
                null);
    }

    /**
     * 配乐前置校验。返回非 null 表示跳过配乐的原因。
     * 本地探测不到时长时交给后端强校验（超上限直接拒绝、不扣费）。
     */
    String musicSkipReason(double durationSeconds) {
        if (durationSeconds > MUSIC_MAX_SECONDS) {
            return "本集时长超过配乐上限 " + MUSIC_MAX_SECONDS / 60 + " 分钟，已跳过配乐";
        }
        return null;
    }

    boolean sfxAllowed(double durationSeconds) {
        return durationSeconds <= SFX_MAX_SECONDS;
    }

    /**
     * 构建混音命令。视频流一律 {@code -c:v copy} 零转码：
     * <ul>
     * <li>成片无原声、只有配乐：直接映射配乐音轨</li>
     * <li>其余情况：配乐先按配置降音量，再与原声 / 音效 amix 混合</li>
     * </ul>
     */
    List<String> buildMixCommand(Path video, Path music, Path sfx, boolean hasOriginalAudio, Path output) {
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
            command.add("-map");
            command.add("0:v");
            command.add("-map");
            command.add("1:a");
        } else {
            StringBuilder filter = new StringBuilder();
            filter.append("[1:a]volume=")
                    .append(String.format(Locale.ROOT, "%.2f", properties.getMusicVolume()))
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
                    .append(":duration=first:normalize=0[aout]");
            command.add("-filter_complex");
            command.add(filter.toString());
            command.add("-map");
            command.add("0:v");
            command.add("-map");
            command.add("[aout]");
        }

        command.add("-c:v");
        command.add("copy");
        command.add("-c:a");
        command.add("aac");
        command.add("-shortest");
        command.add(output.toString());
        return command;
    }

    /**
     * 尽力而为的本地时长探测。ffprobe 不可用或失败时返回 0，交给后端做最终校验。
     */
    private double probeDurationSeconds(Path video) {
        List<String> command = List.of(
                getFfprobeExecutable(), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video.toString());
        String output = runForOutput(command, PROBE_TIMEOUT_SECONDS);
        if (output == null) {
            return 0;
        }
        try {
            return Double.parseDouble(output.trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private boolean hasAudioStream(Path video) {
        List<String> command = List.of(
                getFfprobeExecutable(), "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                video.toString());
        String output = runForOutput(command, PROBE_TIMEOUT_SECONDS);
        return output != null && StrUtil.isNotBlank(output.trim());
    }

    private boolean runFfmpeg(List<String> command) {
        log.info("[EpisodeBgm] 执行混音: {}", String.join(" ", command));
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true);
        try {
            Process process = processBuilder.start();
            StringBuilder tail = new StringBuilder();
            Thread reader = new Thread(() -> drainOutput(process.getInputStream(), tail), "bgm-mux-output");
            reader.setDaemon(true);
            reader.start();
            boolean done = process.waitFor(MUX_TIMEOUT_MINUTES, TimeUnit.MINUTES);
            if (!done) {
                process.destroyForcibly();
                log.error("[EpisodeBgm] 混音超时，已强制终止");
                return false;
            }
            reader.join(TimeUnit.SECONDS.toMillis(5));
            if (process.exitValue() != 0) {
                log.error("[EpisodeBgm] 混音退出码非 0: {}。最近输出: {}", process.exitValue(), tail);
                return false;
            }
            return true;
        } catch (IOException e) {
            log.error("[EpisodeBgm] 混音进程启动失败", e);
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private String runForOutput(List<String> command, int timeoutSeconds) {
        try {
            Process process = new ProcessBuilder(command)
                    .redirectErrorStream(true)
                    .start();
            boolean done = process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            if (!done) {
                process.destroyForcibly();
                return null;
            }
            String output;
            try (InputStream in = process.getInputStream()) {
                output = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            }
            return process.exitValue() == 0 ? output : null;
        } catch (IOException e) {
            log.warn("[EpisodeBgm] ffprobe 不可用或执行失败: {}", e.getMessage());
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    private void drainOutput(InputStream inputStream, StringBuilder tail) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                synchronized (tail) {
                    if (!tail.isEmpty()) {
                        tail.append(System.lineSeparator());
                    }
                    tail.append(line);
                    int maxLength = 4000;
                    if (tail.length() > maxLength) {
                        tail.delete(0, tail.length() - maxLength);
                    }
                }
            }
        } catch (IOException e) {
            log.warn("[EpisodeBgm] 读取混音输出失败", e);
        }
    }

    private String audioExtension(String url) {
        String pathPart = url.split("\\?", 2)[0];
        int dot = pathPart.lastIndexOf('.');
        if (dot >= 0) {
            String ext = pathPart.substring(dot).toLowerCase(Locale.ROOT);
            if (List.of(".m4a", ".wav", ".mp3", ".aac", ".flac").contains(ext)) {
                return ext;
            }
        }
        return ".m4a";
    }

    private String trimDot(String extension) {
        return extension.startsWith(".") ? extension.substring(1) : extension;
    }

    private String getFfmpegExecutable() {
        return StrUtil.isNotBlank(ffmpegPath) ? ffmpegPath.trim() : "ffmpeg";
    }

    private String getFfprobeExecutable() {
        return StrUtil.isNotBlank(ffprobePath) ? ffprobePath.trim() : "ffprobe";
    }
}
