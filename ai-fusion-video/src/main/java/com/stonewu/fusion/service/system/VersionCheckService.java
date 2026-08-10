package com.stonewu.fusion.service.system;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.controller.system.vo.RuntimeVersionInfoRespVO;
import com.stonewu.fusion.controller.system.vo.VersionInfoRespVO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.springframework.boot.info.BuildProperties;
import org.springframework.core.env.Environment;
import org.springframework.lang.Nullable;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 版本检查服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class VersionCheckService {

    private static final String RELEASES_API_URL =
            "https://api.github.com/repos/Stonewuu/ai-fusion-video/releases?per_page=10";
    private static final String RELEASES_PAGE_URL =
            "https://github.com/Stonewuu/ai-fusion-video/releases";
    private static final String DOCKER_HUB_TAG_API_TEMPLATE =
            "https://hub.docker.com/v2/repositories/%s/tags/%s";
    private static final String BACKEND_IMAGE_REPOSITORY = "stonewuu/ai-fusion-video";
    private static final String FRONTEND_IMAGE_REPOSITORY = "stonewuu/ai-fusion-video-web";
    private static final Duration CACHE_TTL = Duration.ofHours(6);
    private static final Pattern SEMVER_PATTERN = Pattern.compile("(\\d+)(?:\\.(\\d+))?(?:\\.(\\d+))?");
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Nullable
    private final BuildProperties buildProperties;

    private final Environment environment;

    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(1, TimeUnit.MINUTES)
            .readTimeout(25, TimeUnit.MINUTES)
            .callTimeout(25, TimeUnit.MINUTES)
            .build();

    private volatile CachedVersionInfo cachedVersionInfo;

    public RuntimeVersionInfoRespVO getRuntimeVersionInfo() {
        return toRuntimeResp(resolveRuntimeVersionInfo());
    }

    public VersionInfoRespVO getVersionInfo(boolean forceRefresh) {
        RuntimeVersionInfo runtimeVersionInfo = resolveRuntimeVersionInfo();

        CachedVersionInfo snapshot = cachedVersionInfo;
        if (!forceRefresh && snapshot != null && !snapshot.isExpired()) {
            return toResp(runtimeVersionInfo, snapshot);
        }

        synchronized (this) {
            snapshot = cachedVersionInfo;
            if (!forceRefresh && snapshot != null && !snapshot.isExpired()) {
                return toResp(runtimeVersionInfo, snapshot);
            }

            CachedVersionInfo refreshed = fetchLatestVersionInfo();
            cachedVersionInfo = refreshed;
            return toResp(runtimeVersionInfo, refreshed);
        }
    }

    private VersionInfoRespVO toResp(RuntimeVersionInfo runtimeVersionInfo, CachedVersionInfo latest) {
        ReleaseInfo deployableRelease = latest.deployableRelease();
        ReleaseInfo latestRelease = latest.latestRelease();
        boolean comparisonEnabled = latest.checkSucceeded()
                && runtimeVersionInfo.comparisonEnabled()
            && deployableRelease != null
            && hasText(deployableRelease.version());
        String versionRelation = resolveVersionRelation(
                runtimeVersionInfo.currentVersion(),
            deployableRelease != null ? deployableRelease.version() : null,
                comparisonEnabled
        );

        VersionInfoRespVO respVO = new VersionInfoRespVO();
        respVO.setCurrentVersion(runtimeVersionInfo.currentVersion());
        respVO.setCurrentVersionDisplay(runtimeVersionInfo.currentVersionDisplay());
        respVO.setLatestVersion(deployableRelease != null ? deployableRelease.version() : null);
        respVO.setLatestVersionDisplay(displayVersion(deployableRelease != null ? deployableRelease.version() : null));
        respVO.setLatestReleaseVersion(latestRelease != null ? latestRelease.version() : null);
        respVO.setLatestReleaseVersionDisplay(displayVersion(latestRelease != null ? latestRelease.version() : null));
        respVO.setLatestReleaseUrl(latestRelease != null ? latestRelease.releaseUrl() : RELEASES_PAGE_URL);
        respVO.setLatestReleasePublishedAt(latestRelease != null ? latestRelease.publishedAt() : null);
        respVO.setLatestReleaseDockerReady(latestRelease != null && latestRelease.dockerReady());
        respVO.setUpdateAvailable("behind".equals(versionRelation));
        respVO.setComparisonEnabled(comparisonEnabled);
        respVO.setDevelopmentBuild(runtimeVersionInfo.developmentBuild());
        respVO.setBuildProfile(runtimeVersionInfo.buildProfile());
        respVO.setVersionRelation(versionRelation);
        respVO.setReleaseUrl(deployableRelease != null ? deployableRelease.releaseUrl() : RELEASES_PAGE_URL);
        respVO.setTagUrl(deployableRelease != null ? deployableRelease.tagUrl() : RELEASES_PAGE_URL);
        respVO.setPublishedAt(deployableRelease != null ? deployableRelease.publishedAt() : null);
        respVO.setCheckedAt(latest.checkedAt().toString());
        respVO.setSource("github-release+docker-hub");
        respVO.setCheckSucceeded(latest.checkSucceeded());
        respVO.setMessage(resolveMessage(runtimeVersionInfo, latest, versionRelation));
        return respVO;
    }

    private RuntimeVersionInfoRespVO toRuntimeResp(RuntimeVersionInfo runtimeVersionInfo) {
        RuntimeVersionInfoRespVO respVO = new RuntimeVersionInfoRespVO();
        respVO.setCurrentVersion(runtimeVersionInfo.currentVersion());
        respVO.setCurrentVersionDisplay(runtimeVersionInfo.currentVersionDisplay());
        respVO.setDevelopmentBuild(runtimeVersionInfo.developmentBuild());
        respVO.setBuildProfile(runtimeVersionInfo.buildProfile());
        respVO.setComparisonEnabled(runtimeVersionInfo.comparisonEnabled());
        respVO.setMessage(runtimeVersionInfo.message());
        return respVO;
    }

    private RuntimeVersionInfo resolveRuntimeVersionInfo() {
        String currentVersion = resolveCurrentVersion();
        String buildProfile = resolveBuildProfile();
        boolean localProfile = Arrays.stream(environment.getActiveProfiles())
                .anyMatch(profile -> "local".equalsIgnoreCase(profile));
        boolean snapshotVersion = hasText(currentVersion)
                && currentVersion.toUpperCase(Locale.ROOT).contains("SNAPSHOT");
        boolean developmentBuild = localProfile || snapshotVersion;
        boolean comparisonEnabled = hasText(currentVersion)
                && !"unknown".equalsIgnoreCase(currentVersion);

        String message = null;
        if (!hasText(currentVersion) || "unknown".equalsIgnoreCase(currentVersion)) {
            message = "当前运行实例未包含可比较的构建版本，线上版本仅供参考。";
        } else if (localProfile) {
            message = "当前运行在本地开发环境，仍会与线上可部署版本比较；如果本地版本更高，将不会提示升级。";
        } else if (snapshotVersion) {
            message = "当前为 SNAPSHOT 构建，仍会与线上可部署版本比较；如果本地版本更高，将不会提示升级。";
        }

        return new RuntimeVersionInfo(
                currentVersion,
                displayVersion(currentVersion),
                developmentBuild,
                buildProfile,
                comparisonEnabled,
                message
        );
    }

    private String resolveCurrentVersion() {
        if (buildProperties != null && hasText(buildProperties.getVersion())) {
            return sanitizeVersion(buildProperties.getVersion());
        }

        Package appPackage = VersionCheckService.class.getPackage();
        if (appPackage != null && hasText(appPackage.getImplementationVersion())) {
            return sanitizeVersion(appPackage.getImplementationVersion());
        }

        return "unknown";
    }

    private String resolveBuildProfile() {
        String[] activeProfiles = environment.getActiveProfiles();
        if (activeProfiles.length == 0) {
            return "default";
        }
        return String.join(",", activeProfiles);
    }

    private CachedVersionInfo fetchLatestVersionInfo() {
        Request request = new Request.Builder()
                .url(RELEASES_API_URL)
                .header("Accept", "application/vnd.github+json")
                .header("X-GitHub-Api-Version", "2022-11-28")
                .header("User-Agent", "ai-fusion-video-version-check")
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String message = "GitHub 版本检查失败: HTTP " + response.code();
                log.warn("[VersionCheck] {}", message);
                return CachedVersionInfo.failure(message);
            }

            if (response.body() == null) {
                String message = "GitHub 版本检查失败: 响应体为空";
                log.warn("[VersionCheck] {}", message);
                return CachedVersionInfo.failure(message);
            }

            String body = response.body().string();
            JsonNode root = OBJECT_MAPPER.readTree(body);
            if (!root.isArray() || root.isEmpty()) {
                String message = "GitHub 版本检查失败: 未找到可用 release";
                log.warn("[VersionCheck] {}", message);
                return CachedVersionInfo.failure(message);
            }

            ReleaseInfo latestRelease = null;
            ReleaseInfo deployableRelease = null;

            for (JsonNode releaseNode : root) {
                if (releaseNode.path("draft").asBoolean(false) || releaseNode.path("prerelease").asBoolean(false)) {
                    continue;
                }

                String tagName = sanitizeVersion(releaseNode.path("tag_name").asText(null));
                if (!hasText(tagName)) {
                    continue;
                }

                String releaseUrl = textOrNull(releaseNode, "html_url");
                String publishedAt = textOrNull(releaseNode, "published_at");
                boolean dockerReady = areDockerImagesReady(tagName);
                ReleaseInfo releaseInfo = new ReleaseInfo(
                        tagName,
                        hasText(releaseUrl) ? releaseUrl : buildTagUrl(tagName),
                        buildTagUrl(tagName),
                        publishedAt,
                        dockerReady
                );

                if (latestRelease == null) {
                    latestRelease = releaseInfo;
                }
                if (dockerReady) {
                    deployableRelease = releaseInfo;
                    break;
                }
            }

            if (latestRelease == null) {
                String message = "GitHub 版本检查失败: 未找到正式 release";
                log.warn("[VersionCheck] {}", message);
                return CachedVersionInfo.failure(message);
            }

            String message = null;
            if (!latestRelease.dockerReady()) {
                if (deployableRelease != null) {
                    message = String.format(
                            "最新 Release %s 已发布，但 Docker 镜像尚未就绪；当前可升级的最新版本为 %s。",
                            displayVersion(latestRelease.version()),
                            displayVersion(deployableRelease.version())
                    );
                } else {
                    message = String.format(
                            "最新 Release %s 已发布，但 Docker 镜像尚未就绪，暂不建议升级。",
                            displayVersion(latestRelease.version())
                    );
                }
            }

            return CachedVersionInfo.success(latestRelease, deployableRelease, message);
        } catch (IOException e) {
            String message = "GitHub 版本检查失败: " + e.getMessage();
            log.warn("[VersionCheck] {}", message);
            return CachedVersionInfo.failure(message);
        }
    }

    private boolean areDockerImagesReady(String version) {
        return dockerTagExists(BACKEND_IMAGE_REPOSITORY, version)
                && dockerTagExists(FRONTEND_IMAGE_REPOSITORY, version);
    }

    private boolean dockerTagExists(String repository, String version) {
        String encodedVersion = URLEncoder.encode(version, StandardCharsets.UTF_8);
        String url = String.format(DOCKER_HUB_TAG_API_TEMPLATE, repository, encodedVersion);
        Request request = new Request.Builder()
                .url(url)
                .header("Accept", "application/json")
                .header("User-Agent", "ai-fusion-video-version-check")
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (response.isSuccessful()) {
                return true;
            }
            if (response.code() != 404) {
                log.warn("[VersionCheck] Docker Hub 标签检测失败: repo={}, tag={}, code={}", repository, version,
                        response.code());
            }
            return false;
        } catch (IOException e) {
            log.warn("[VersionCheck] Docker Hub 标签检测异常: repo={}, tag={}, err={}", repository, version,
                    e.getMessage());
            return false;
        }
    }

    private String resolveVersionRelation(String currentVersion, String latestVersion, boolean comparisonEnabled) {
        if (!comparisonEnabled || !hasText(currentVersion) || !hasText(latestVersion)) {
            return "incomparable";
        }

        int result = compareSemver(currentVersion, latestVersion);
        if (result < 0) {
            return "behind";
        }
        if (result > 0) {
            return "ahead";
        }
        return "same";
    }

    private String resolveMessage(
            RuntimeVersionInfo runtimeVersionInfo,
            CachedVersionInfo latest,
            String versionRelation
    ) {
        if (!latest.checkSucceeded()) {
            return latest.message();
        }
        List<String> messages = new ArrayList<>();
        if (hasText(runtimeVersionInfo.message())) {
            messages.add(runtimeVersionInfo.message());
        }
        if (hasText(latest.message())) {
            messages.add(latest.message());
        }
        if ("ahead".equals(versionRelation)) {
            messages.add("当前运行版本高于线上最新可部署版本，可能来自本地开发代码或尚未发布的构建。");
        }
        return messages.isEmpty() ? null : String.join(" ", messages);
    }

    private int compareSemver(String left, String right) {
        int[] leftParts = parseSemver(left);
        int[] rightParts = parseSemver(right);
        for (int index = 0; index < 3; index++) {
            int result = Integer.compare(leftParts[index], rightParts[index]);
            if (result != 0) {
                return result;
            }
        }
        return 0;
    }

    private int[] parseSemver(String version) {
        Matcher matcher = SEMVER_PATTERN.matcher(version);
        if (!matcher.find()) {
            return new int[]{0, 0, 0};
        }
        return new int[]{
                parseIntSafe(matcher.group(1)),
                parseIntSafe(matcher.group(2)),
                parseIntSafe(matcher.group(3))
        };
    }

    private int parseIntSafe(String value) {
        if (!hasText(value)) {
            return 0;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }

    private String displayVersion(String version) {
        if (!hasText(version)) {
            return "未检测到";
        }
        if ("unknown".equalsIgnoreCase(version)) {
            return "未知版本";
        }
        return "v" + sanitizeVersion(version);
    }

    private String sanitizeVersion(String version) {
        if (!hasText(version)) {
            return null;
        }
        String normalized = version.trim();
        if (normalized.toLowerCase(Locale.ROOT).startsWith("v")) {
            normalized = normalized.substring(1);
        }
        return normalized;
    }

    private String buildTagUrl(String tagName) {
        if (!hasText(tagName)) {
            return RELEASES_PAGE_URL;
        }
        return "https://github.com/Stonewuu/ai-fusion-video/releases/tag/v" + sanitizeVersion(tagName);
    }

    private String textOrNull(JsonNode node, String fieldName) {
        JsonNode value = node.path(fieldName);
        return value.isMissingNode() || value.isNull() ? null : value.asText();
    }

    private boolean hasText(String value) {
        return value != null && !value.isBlank();
    }

    private record CachedVersionInfo(
            ReleaseInfo latestRelease,
            ReleaseInfo deployableRelease,
            Instant checkedAt,
            boolean checkSucceeded,
            String message
    ) {
        private boolean isExpired() {
            return checkedAt.plus(CACHE_TTL).isBefore(Instant.now());
        }

        private static CachedVersionInfo success(ReleaseInfo latestRelease, ReleaseInfo deployableRelease, String message) {
            return new CachedVersionInfo(
                latestRelease,
                deployableRelease,
                    Instant.now(),
                    true,
                message
            );
        }

        private static CachedVersionInfo failure(String message) {
            return new CachedVersionInfo(
                    null,
                    null,
                    Instant.now(),
                    false,
                    message
            );
        }

        private static boolean hasTextStatic(String value) {
            return value != null && !value.isBlank();
        }
    }

    private record RuntimeVersionInfo(
            String currentVersion,
            String currentVersionDisplay,
            boolean developmentBuild,
            String buildProfile,
            boolean comparisonEnabled,
            String message
    ) {
    }

            private record ReleaseInfo(
                String version,
                String releaseUrl,
                String tagUrl,
                String publishedAt,
                boolean dockerReady
            ) {
            }
}
