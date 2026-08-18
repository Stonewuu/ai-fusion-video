package com.stonewu.fusion.service.storyboard;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.storyboard.dto.EpisodeBgmOutcome;
import com.stonewu.fusion.service.task.TaskStreamService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.FileSystemUtils;
import org.springframework.util.StringUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.IDN;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

/**
 * 按集合成视频服务。
 * <p>
 * 流程：取该集所有场次的所有镜头视频（按 sortOrder 排序），下载到临时目录，
 * 用 ffmpeg concat 拼接（先尝试 demuxer 零转码；失败则 fallback 到 filter_complex 重新编码），
 * 然后通过 MediaStorageService 持久化，更新 episode 的合成状态与URL。
 */
@Service
@Slf4j
public class VideoComposeService {

    private static final String LOCAL_MEDIA_PUBLIC_PREFIX = "/media/";
    private static final String TASK_TYPE = "storyboard_episode_compose";
    private static final String TASK_CONTEXT_TYPE = "storyboard_episode";
    private static final String TASK_INITIAL_MESSAGE = "已提交合成任务，正在拼接镜头视频…";

    public static final int STATUS_IDLE = 0;
    public static final int STATUS_RUNNING = 1;
    public static final int STATUS_DONE = 2;
    public static final int STATUS_FAILED = 3;

    private final StoryboardService storyboardService;
    private final StoryboardEpisodeMapper episodeMapper;
    private final MediaStorageService mediaStorageService;
    private final StorageConfigService storageConfigService;
    private final TaskStreamService taskStreamService;
    private final EpisodeBgmService episodeBgmService;
    private final Executor videoComposeExecutor;

    @Value("${app.storage.local-base-path:./data/media}")
    private String mediaLocalPath;

    @Value("${video.compose.allowed-hosts:}")
    private String allowedHostsConfig;

    @Value("${video.compose.max-redirects:3}")
    private int maxRedirects;

    @Value("${video.compose.ffmpeg-path:ffmpeg}")
    private String ffmpegPath;

    @Value("${video.compose.ffprobe-path:ffprobe}")
    private String ffprobePath;

    public VideoComposeService(StoryboardService storyboardService,
                               StoryboardEpisodeMapper episodeMapper,
                               MediaStorageService mediaStorageService,
                               StorageConfigService storageConfigService,
                               TaskStreamService taskStreamService,
                               EpisodeBgmService episodeBgmService,
                               @Qualifier("videoComposeExecutor") Executor videoComposeExecutor) {
        this.storyboardService = storyboardService;
        this.episodeMapper = episodeMapper;
        this.mediaStorageService = mediaStorageService;
        this.storageConfigService = storageConfigService;
        this.taskStreamService = taskStreamService;
        this.episodeBgmService = episodeBgmService;
        this.videoComposeExecutor = videoComposeExecutor;
    }

    /**
     * 提交合成任务。同步标记状态为 RUNNING，异步执行实际合成。
     * 若已在合成中则抛出异常。
     */
    public String submitCompose(Long episodeId, Long userId) {
        StoryboardEpisode episode = episodeMapper.selectById(episodeId);
        if (episode == null) {
            throw new BusinessException(404, "分镜集不存在: " + episodeId);
        }

        Storyboard storyboard = storyboardService.getById(episode.getStoryboardId());
        if (storyboard == null) {
            throw new BusinessException(404, "分镜不存在: " + episode.getStoryboardId());
        }

        String taskId = taskStreamService.createTask(
                userId,
                storyboard.getProjectId(),
                TASK_TYPE,
                buildTaskTitle(episode),
                TASK_CONTEXT_TYPE,
                episodeId,
                TASK_INITIAL_MESSAGE
        );

        List<String> videoUrls = collectVideoUrls(episodeId);
        if (videoUrls.isEmpty()) {
            String message = "本集没有可合成的视频，请先生成镜头视频";
            markFailed(episodeId, message);
            taskStreamService.fail(taskId, message);
            return taskId;
        }

        int updated = episodeMapper.update(null, new UpdateWrapper<StoryboardEpisode>()
            .eq("id", episodeId)
            .ne("compose_status", STATUS_RUNNING)
            .set("compose_status", STATUS_RUNNING)
            .set("compose_error_msg", null)
            .set("composed_video_url", null)
            .set("composed_at", null)
            .set("bgm_audio_url", null)
            .set("bgm_license_id", null)
            .set("bgm_sfx_audio_url", null)
            .set("bgm_sfx_license_id", null)
            .set("bgm_error_msg", null));
        if (updated == 0) {
            taskStreamService.fail(taskId, "本集已在合成中，请稍候");
            return taskId;
        }

        try {
            videoComposeExecutor.execute(() -> {
                try {
                    doCompose(episodeId, taskId, videoUrls);
                } catch (Throwable t) {
                    String errorMessage = resolveErrorMessage(t);
                    log.error("[VideoCompose] 合成失败: episodeId={}", episodeId, t);
                    markFailed(episodeId, errorMessage);
                    taskStreamService.fail(taskId, errorMessage);
                }
            });
        } catch (RejectedExecutionException e) {
            String message = "合成队列繁忙，请稍后重试";
            markFailed(episodeId, message);
            taskStreamService.fail(taskId, message);
        }
        return taskId;
    }

    private void doCompose(Long episodeId, String taskId, List<String> videoUrls) throws Exception {
        log.info("[VideoCompose] 开始合成 episodeId={}, taskId={}", episodeId, taskId);
        long startMs = System.currentTimeMillis();
        log.info("[VideoCompose] episodeId={}, 待合成视频数={}", episodeId, videoUrls.size());

        Path workDir = Files.createTempDirectory("compose_ep_" + episodeId + "_");
        try {
            List<Path> localFiles = new ArrayList<>();
            for (int i = 0; i < videoUrls.size(); i++) {
                Path local = workDir.resolve(String.format("v%04d.mp4", i));
                downloadToFile(videoUrls.get(i), local);
                localFiles.add(local);
            }

            Path listFile = workDir.resolve("list.txt");
            StringBuilder sb = new StringBuilder();
            for (Path f : localFiles) {
                String s = f.toAbsolutePath().toString().replace("'", "'\\''");
                sb.append("file '").append(s).append("'\n");
            }
            Files.writeString(listFile, sb.toString(), StandardCharsets.UTF_8);

            Path output = workDir.resolve("output.mp4");
            boolean ok = runFfmpegConcatDemuxer(listFile, output);
            if (!ok) {
                log.warn("[VideoCompose] concat demuxer 失败，回退到 filter_complex episodeId={}", episodeId);
                ok = runFfmpegFilterConcat(localFiles, output);
            }
            if (!ok || !Files.exists(output) || Files.size(output) == 0) {
                throw new RuntimeException("ffmpeg 合成失败（concat 与 filter 均失败）");
            }

            // 可选的合成后配乐：失败或跳过都不影响本集合成结果
            Path finalVideo = output;
            EpisodeBgmOutcome bgmOutcome = null;
            String bgmError = null;
            if (episodeBgmService.isEnabled()) {
                try {
                    bgmOutcome = episodeBgmService.apply(output, workDir,
                            message -> taskStreamService.publishContent(taskId, message));
                    if (bgmOutcome.scoredVideoFile() != null) {
                        finalVideo = bgmOutcome.scoredVideoFile();
                    }
                } catch (Exception e) {
                    bgmError = StringUtils.hasText(e.getMessage()) ? e.getMessage() : "配乐失败";
                    log.error("[VideoCompose] 配乐失败，保留无配乐成片: episodeId={}", episodeId, e);
                    taskStreamService.publishContent(taskId, "配乐失败：" + bgmError + "（本集视频不受影响）");
                }
            }

            String storedUrl = mediaStorageService.storeFile(finalVideo, "videos/composed", "mp4");
            log.info("[VideoCompose] 已保存到存储: {}", storedUrl);

            StoryboardEpisode update = new StoryboardEpisode();
            update.setId(episodeId);
            update.setComposedVideoUrl(storedUrl);
            update.setComposeStatus(STATUS_DONE);
            update.setComposedAt(LocalDateTime.now());
            update.setComposeErrorMsg(null);
            if (bgmOutcome != null) {
                update.setBgmAudioUrl(bgmOutcome.musicAudioUrl());
                update.setBgmLicenseId(bgmOutcome.musicLicenseId());
                update.setBgmSfxAudioUrl(bgmOutcome.sfxAudioUrl());
                update.setBgmSfxLicenseId(bgmOutcome.sfxLicenseId());
            }
            update.setBgmErrorMsg(bgmError);
            episodeMapper.updateById(update);

                taskStreamService.complete(taskId, buildCompleteMessage(storedUrl, bgmOutcome, bgmError));

            log.info("[VideoCompose] 完成 episodeId={}, 耗时={}ms, 视频数={}",
                    episodeId, System.currentTimeMillis() - startMs, videoUrls.size());
        } finally {
            try {
                FileSystemUtils.deleteRecursively(workDir.toFile());
            } catch (Exception e) {
                log.warn("[VideoCompose] 临时目录清理失败 {}", workDir, e);
            }
        }
    }

    private String buildCompleteMessage(String storedUrl, EpisodeBgmOutcome bgmOutcome, String bgmError) {
        StringBuilder message = new StringBuilder("✓ 合成完成 · 视频地址：").append(storedUrl);
        if (bgmOutcome != null && bgmOutcome.scoredVideoFile() != null) {
            message.append(" · 配乐已合入");
            if (StringUtils.hasText(bgmOutcome.musicLicenseId())) {
                message.append("（license_id: ").append(bgmOutcome.musicLicenseId()).append("）");
            }
        } else if (bgmOutcome != null && StringUtils.hasText(bgmOutcome.skippedReason())) {
            message.append(" · ").append(bgmOutcome.skippedReason());
        } else if (StringUtils.hasText(bgmError)) {
            message.append(" · 配乐失败：").append(bgmError);
        }
        return message.toString();
    }

    private String buildTaskTitle(StoryboardEpisode episode) {
        String episodeLabel = StringUtils.hasText(episode.getTitle())
                ? episode.getTitle().trim()
                : (episode.getEpisodeNumber() != null
                ? "第 " + episode.getEpisodeNumber() + " 集"
                : "集 " + episode.getId());
        return "合成本集视频：" + episodeLabel;
    }

    private String resolveErrorMessage(Throwable throwable) {
        if (throwable == null) {
            return "合成失败";
        }
        if (isExecutableMissing(throwable)) {
            return buildMissingExecutableMessage("ffmpeg", "video.compose.ffmpeg-path", getFfmpegExecutable());
        }
        if (StringUtils.hasText(throwable.getMessage())) {
            return throwable.getMessage();
        }
        return "合成失败";
    }

    private List<String> collectVideoUrls(Long episodeId) {
        List<StoryboardScene> scenes = new ArrayList<>(storyboardService.listScenesByEpisode(episodeId));
        scenes.sort(Comparator.comparing(s -> Optional.ofNullable(s.getSortOrder()).orElse(0)));

        List<String> urls = new ArrayList<>();
        for (StoryboardScene scene : scenes) {
            List<StoryboardItem> items = new ArrayList<>(storyboardService.listItemsByScene(scene.getId()));
            items.sort(Comparator.comparing(i -> Optional.ofNullable(i.getSortOrder()).orElse(0)));
            for (StoryboardItem item : items) {
                String url = StringUtils.hasText(item.getVideoUrl())
                        ? item.getVideoUrl()
                        : item.getGeneratedVideoUrl();
                if (StringUtils.hasText(url)) {
                    urls.add(url);
                }
            }
        }
        return urls;
    }

    private boolean runFfmpegConcatDemuxer(Path listFile, Path output) throws Exception {
        List<String> cmd = List.of(
                getFfmpegExecutable(), "-y",
                "-f", "concat", "-safe", "0",
                "-i", listFile.toString(),
                "-c", "copy",
                output.toString()
        );
        return runFfmpeg(cmd, "concat-demuxer", 15);
    }

    private boolean runFfmpegFilterConcat(List<Path> files, Path output) throws Exception {
        boolean includeAudio = allFilesHaveAudio(files);
        if (!includeAudio) {
            log.warn("[VideoCompose] 检测到至少一个输入无音轨，回退到仅视频重编码 concat");
        }

        List<String> cmd = new ArrayList<>();
        cmd.add(getFfmpegExecutable());
        cmd.add("-y");
        for (Path f : files) {
            cmd.add("-i");
            cmd.add(f.toString());
        }
        StringBuilder filter = new StringBuilder();
        for (int i = 0; i < files.size(); i++) {
            filter.append("[").append(i).append(":v]");
            if (includeAudio) {
                filter.append("[").append(i).append(":a]");
            }
        }
        filter.append("concat=n=").append(files.size())
                .append(":v=1:a=").append(includeAudio ? 1 : 0)
                .append(includeAudio ? "[outv][outa]" : "[outv]");
        cmd.add("-filter_complex");
        cmd.add(filter.toString());
        cmd.add("-map");
        cmd.add("[outv]");
        if (includeAudio) {
            cmd.add("-map");
            cmd.add("[outa]");
        }
        cmd.add("-c:v");
        cmd.add("libx264");
        cmd.add("-preset");
        cmd.add("veryfast");
        if (includeAudio) {
            cmd.add("-c:a");
            cmd.add("aac");
        }
        cmd.add("-pix_fmt");
        cmd.add("yuv420p");
        cmd.add(output.toString());
        return runFfmpeg(cmd, "filter-complex", 30);
    }

    private boolean runFfmpeg(List<String> cmd, String tag, int timeoutMinutes) throws Exception {
        log.info("[VideoCompose:{}] 执行: {}", tag, String.join(" ", cmd));
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p;
        try {
            p = pb.start();
        } catch (IOException e) {
            throw wrapExecutableStartException(e, "ffmpeg", "video.compose.ffmpeg-path", getFfmpegExecutable());
        }
        StringBuffer outputTail = new StringBuffer();
        Thread outputReader = new Thread(() -> drainProcessOutput(p.getInputStream(), tag, outputTail),
                "ffmpeg-" + tag + "-output");
        outputReader.setDaemon(true);
        outputReader.start();

        boolean done;
        try {
            done = p.waitFor(timeoutMinutes, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
            throw e;
        }
        if (!done) {
            p.destroy();
            if (!p.waitFor(5, TimeUnit.SECONDS)) {
                p.destroyForcibly();
            }
            outputReader.join(TimeUnit.SECONDS.toMillis(5));
            log.error("[VideoCompose:{}] 超时，已强制终止。最近输出: {}", tag, summarizeOutputTail(outputTail));
            return false;
        }

        outputReader.join(TimeUnit.SECONDS.toMillis(5));
        int exit = p.exitValue();
        if (exit != 0) {
            log.error("[VideoCompose:{}] 退出码非 0: {}。最近输出: {}", tag, exit, summarizeOutputTail(outputTail));
            return false;
        }
        return true;
    }

    private void downloadToFile(String url, Path dest) throws IOException {
        if (url.startsWith(LOCAL_MEDIA_PUBLIC_PREFIX)) {
            Path local = resolveManagedMediaPath(url);
            if (Files.exists(local)) {
                Files.copy(local, dest, StandardCopyOption.REPLACE_EXISTING);
                return;
            }
            if (!isAbsoluteHttpUrl(url)) {
                throw new IOException("本地媒体文件不存在: " + local);
            }
            log.warn("[VideoCompose] /media/ 路径文件不存在，回退 HTTP: {}", url);
        }

        URI current = URI.create(url);
        int redirects = 0;
        while (true) {
            validateRemoteUri(current);
            HttpURLConnection conn = openConnection(current);
            try {
                int code = conn.getResponseCode();
                if (isRedirect(code)) {
                    if (redirects >= maxRedirects) {
                        throw new IOException("下载重定向次数过多: " + url);
                    }
                    String location = conn.getHeaderField("Location");
                    if (!StringUtils.hasText(location)) {
                        throw new IOException("下载重定向缺少 Location: " + url);
                    }
                    current = current.resolve(location);
                    redirects++;
                    continue;
                }
                if (code < 200 || code >= 300) {
                    throw new IOException("下载失败 HTTP " + code + ": " + current);
                }
                try (InputStream in = conn.getInputStream()) {
                    Files.copy(in, dest, StandardCopyOption.REPLACE_EXISTING);
                    return;
                }
            } finally {
                conn.disconnect();
            }
        }
    }

    private void markFailed(Long episodeId, String msg) {
        try {
            String trimmed = msg == null ? "未知错误" : (msg.length() > 1000 ? msg.substring(0, 1000) : msg);
            StoryboardEpisode update = new StoryboardEpisode();
            update.setId(episodeId);
            update.setComposeStatus(STATUS_FAILED);
            update.setComposeErrorMsg(trimmed);
            update.setComposedVideoUrl(null);
            update.setComposedAt(null);
            episodeMapper.updateById(update);
        } catch (Exception e) {
            log.error("[VideoCompose] 更新失败状态异常", e);
        }
    }

    private boolean allFilesHaveAudio(List<Path> files) {
        for (Path file : files) {
            if (!hasAudioStream(file)) {
                return false;
            }
        }
        return true;
    }

    private boolean hasAudioStream(Path file) {
        List<String> cmd = List.of(
            getFfprobeExecutable(), "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                file.toString()
        );
        try {
            Process process = new ProcessBuilder(cmd)
                    .redirectErrorStream(true)
                    .start();
            boolean done = process.waitFor(15, TimeUnit.SECONDS);
            if (!done) {
                process.destroyForcibly();
                log.warn("[VideoCompose] ffprobe 检测音轨超时: {}", file);
                return false;
            }
            String output;
            try (InputStream in = process.getInputStream()) {
                output = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            }
            if (process.exitValue() != 0) {
                log.warn("[VideoCompose] ffprobe 检测音轨失败，按无音轨处理: {}", file);
                return false;
            }
            return StringUtils.hasText(output.trim());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("[VideoCompose] ffprobe 检测音轨被中断: {}", file, e);
            return false;
        } catch (IOException e) {
            IOException wrapped = wrapExecutableStartException(
                    e,
                    "ffprobe",
                    "video.compose.ffprobe-path",
                    getFfprobeExecutable()
            );
            log.warn("[VideoCompose] ffprobe 不可用或检测失败，按无音轨处理: {}", file, wrapped);
            return false;
        }
    }

    private String getFfmpegExecutable() {
        return StringUtils.hasText(ffmpegPath) ? ffmpegPath.trim() : "ffmpeg";
    }

    private String getFfprobeExecutable() {
        return StringUtils.hasText(ffprobePath) ? ffprobePath.trim() : "ffprobe";
    }

    private IOException wrapExecutableStartException(IOException exception,
                                                     String executableName,
                                                     String propertyName,
                                                     String configuredValue) {
        if (!isExecutableMissing(exception)) {
            return exception;
        }
        return new IOException(buildMissingExecutableMessage(executableName, propertyName, configuredValue), exception);
    }

    private boolean isExecutableMissing(Throwable throwable) {
        Throwable current = throwable;
        while (current != null) {
            String message = current.getMessage();
            if (StringUtils.hasText(message)) {
                String normalized = message.toLowerCase(Locale.ROOT);
                if (normalized.contains("createprocess error=2")
                        || normalized.contains("系统找不到指定的文件")
                        || normalized.contains("no such file or directory")
                        || normalized.contains("cannot run program")) {
                    return true;
                }
            }
            current = current.getCause();
        }
        return false;
    }

    private String buildMissingExecutableMessage(String executableName,
                                                String propertyName,
                                                String configuredValue) {
        return "未找到 " + executableName + " 可执行文件，请先安装 " + executableName
                + "，或在配置中设置 " + propertyName + "。当前值：" + configuredValue;
    }

    private void drainProcessOutput(InputStream inputStream, String tag, StringBuffer outputTail) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                appendOutputTail(outputTail, line);
                if (log.isDebugEnabled()) {
                    log.debug("[ffmpeg:{}] {}", tag, line);
                }
            }
        } catch (IOException e) {
            log.warn("[VideoCompose:{}] 读取 ffmpeg 输出失败", tag, e);
        }
    }

    private void appendOutputTail(StringBuffer outputTail, String line) {
        synchronized (outputTail) {
            if (outputTail.length() > 0) {
                outputTail.append(System.lineSeparator());
            }
            outputTail.append(line);
            int maxLength = 4000;
            if (outputTail.length() > maxLength) {
                outputTail.delete(0, outputTail.length() - maxLength);
            }
        }
    }

    private String summarizeOutputTail(StringBuffer outputTail) {
        synchronized (outputTail) {
            return outputTail.isEmpty() ? "<no output>" : outputTail.toString();
        }
    }

    private Path resolveManagedMediaPath(String url) throws IOException {
        String rel = url.substring(LOCAL_MEDIA_PUBLIC_PREFIX.length());
        Path relativePath = Paths.get(rel).normalize();
        if (relativePath.isAbsolute() || relativePath.startsWith("..")) {
            throw new IOException("非法媒体相对路径: " + rel);
        }

        Path basePath = resolveLocalMediaBasePath();
        Path resolved = basePath.resolve(relativePath).normalize();
        if (!resolved.startsWith(basePath)) {
            throw new IOException("媒体路径越界: " + rel);
        }
        return resolved;
    }

    private Path resolveLocalMediaBasePath() {
        String basePath = mediaLocalPath;
        try {
            StorageConfig config = storageConfigService.getDefaultConfig();
            if (config != null
                    && "local".equalsIgnoreCase(config.getType())
                    && StringUtils.hasText(config.getBasePath())) {
                basePath = config.getBasePath();
            }
        } catch (Exception e) {
            log.debug("[VideoCompose] 读取默认存储配置失败，回退到 application 配置路径: {}", basePath, e);
        }
        return Paths.get(basePath).toAbsolutePath().normalize();
    }

    private HttpURLConnection openConnection(URI uri) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) uri.toURL().openConnection();
        conn.setConnectTimeout(Math.toIntExact(TimeUnit.MINUTES.toMillis(1)));
        conn.setReadTimeout(Math.toIntExact(TimeUnit.MINUTES.toMillis(25)));
        conn.setInstanceFollowRedirects(false);
        return conn;
    }

    private void validateRemoteUri(URI uri) throws IOException {
        String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
        if (!"http".equals(scheme) && !"https".equals(scheme)) {
            throw new IOException("仅允许下载 http/https 资源: " + uri);
        }

        String host = uri.getHost();
        if (!StringUtils.hasText(host)) {
            throw new IOException("下载地址缺少 host: " + uri);
        }

        String normalizedHost = IDN.toASCII(host).toLowerCase(Locale.ROOT);
        Set<String> allowedHosts = parseAllowedHosts();
        if (!allowedHosts.isEmpty()) {
            if (!matchesAllowedHost(normalizedHost, allowedHosts)) {
                throw new IOException("下载地址 host 不在白名单: " + normalizedHost);
            }
            return;
        }

        InetAddress[] addresses = InetAddress.getAllByName(normalizedHost);
        for (InetAddress address : addresses) {
            if (address.isAnyLocalAddress()
                    || address.isLoopbackAddress()
                    || address.isLinkLocalAddress()
                    || address.isSiteLocalAddress()
                    || address.isMulticastAddress()) {
                throw new IOException("拒绝访问内网或本地地址: " + normalizedHost);
            }
        }
    }

    private Set<String> parseAllowedHosts() {
        Set<String> hosts = new LinkedHashSet<>();
        if (!StringUtils.hasText(allowedHostsConfig)) {
            return hosts;
        }
        for (String token : allowedHostsConfig.split(",")) {
            String trimmed = token.trim().toLowerCase(Locale.ROOT);
            if (!trimmed.isEmpty()) {
                hosts.add(trimmed);
            }
        }
        return hosts;
    }

    private boolean matchesAllowedHost(String host, Set<String> allowedHosts) {
        for (String pattern : allowedHosts) {
            if (pattern.startsWith("*.")) {
                String suffix = pattern.substring(1);
                if (host.endsWith(suffix)) {
                    return true;
                }
                continue;
            }
            if (host.equals(pattern) || host.endsWith("." + pattern)) {
                return true;
            }
        }
        return false;
    }

    private boolean isRedirect(int statusCode) {
        return statusCode == HttpURLConnection.HTTP_MOVED_PERM
                || statusCode == HttpURLConnection.HTTP_MOVED_TEMP
                || statusCode == HttpURLConnection.HTTP_SEE_OTHER
                || statusCode == 307
                || statusCode == 308;
    }

    private boolean isAbsoluteHttpUrl(String url) {
        String lower = url.toLowerCase(Locale.ROOT);
        return lower.startsWith("http://") || lower.startsWith("https://");
    }
}
