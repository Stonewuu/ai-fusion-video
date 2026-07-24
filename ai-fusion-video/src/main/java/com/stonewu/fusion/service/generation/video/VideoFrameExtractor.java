package com.stonewu.fusion.service.generation.video;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.util.FileSystemUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * 使用 ffmpeg 为生成视频补齐首尾帧。提帧属于增强步骤，失败时保留视频结果。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class VideoFrameExtractor {

    private static final String LOCAL_MEDIA_PREFIX = "/media/";
    private static final String DEFAULT_LOCAL_MEDIA_PATH = "./data/media";
    private static final String FRAME_STORAGE_DIR = "images/video-frames";

    private final MediaStorageService mediaStorageService;
    private final StorageConfigService storageConfigService;

    @Value("${video.compose.ffmpeg-path:ffmpeg}")
    private String ffmpegPath;

    @Value("${video.frame-extraction.timeout-seconds:60}")
    private long timeoutSeconds;

    public ExtractedFrames extract(String videoUrl, boolean extractFirst, boolean extractLast) {
        if (StrUtil.isBlank(videoUrl) || (!extractFirst && !extractLast)) {
            return ExtractedFrames.empty();
        }

        Path tempDirectory = null;
        try {
            tempDirectory = Files.createTempDirectory("fusion-video-frames-");
            String input = resolveInput(videoUrl);
            String firstFrameUrl = extractFirst
                    ? extractAndStore(input, tempDirectory.resolve("first.jpg"), false)
                    : null;
            String lastFrameUrl = extractLast
                    ? extractAndStore(input, tempDirectory.resolve("last.jpg"), true)
                    : null;
            return new ExtractedFrames(firstFrameUrl, lastFrameUrl);
        } catch (Exception e) {
            log.warn("[VideoFrameExtractor] 视频提帧失败，保留原视频: url={}, error={}", videoUrl, e.getMessage());
            return ExtractedFrames.empty();
        } finally {
            if (tempDirectory != null) {
                FileSystemUtils.deleteRecursively(tempDirectory.toFile());
            }
        }
    }

    private String extractAndStore(String input, Path output, boolean lastFrame) throws IOException {
        List<String> command = new ArrayList<>();
        command.add(StrUtil.blankToDefault(ffmpegPath, "ffmpeg").trim());
        command.add("-hide_banner");
        command.add("-loglevel");
        command.add("error");
        command.add("-y");
        if (lastFrame) {
            command.add("-sseof");
            command.add("-0.1");
        }
        command.add("-i");
        command.add(input);
        if (!lastFrame) {
            command.add("-vf");
            command.add("select=eq(n\\,0)");
        }
        command.add("-frames:v");
        command.add("1");
        command.add("-q:v");
        command.add("2");
        command.add(output.toString());

        Process process = new ProcessBuilder(command)
                .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                .redirectError(ProcessBuilder.Redirect.DISCARD)
                .start();
        boolean completed;
        try {
            completed = process.waitFor(Math.max(1, timeoutSeconds), TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            process.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("视频提帧被中断", e);
        }
        if (!completed) {
            process.destroyForcibly();
            throw new IOException("视频提帧超时");
        }
        if (process.exitValue() != 0 || !Files.isRegularFile(output) || Files.size(output) == 0) {
            throw new IOException((lastFrame ? "尾帧" : "首帧") + "提取失败，ffmpeg exit=" + process.exitValue());
        }
        return mediaStorageService.storeFile(output, FRAME_STORAGE_DIR, "jpg");
    }

    private String resolveInput(String videoUrl) throws IOException {
        if (!videoUrl.startsWith(LOCAL_MEDIA_PREFIX)) {
            return videoUrl;
        }

        StorageConfig config = storageConfigService.getDefaultConfig();
        String configuredBasePath = config != null ? config.getBasePath() : null;
        Path basePath = Paths.get(StrUtil.blankToDefault(configuredBasePath, DEFAULT_LOCAL_MEDIA_PATH))
                .toAbsolutePath()
                .normalize();
        Path localVideo = basePath
                .resolve(videoUrl.substring(LOCAL_MEDIA_PREFIX.length()))
                .normalize();
        if (!localVideo.startsWith(basePath) || !Files.isRegularFile(localVideo)) {
            throw new IOException("本地视频文件不存在: " + localVideo);
        }
        return localVideo.toString();
    }

    public record ExtractedFrames(String firstFrameUrl, String lastFrameUrl) {
        public static ExtractedFrames empty() {
            return new ExtractedFrames(null, null);
        }
    }
}
