"use client";

import { useEffect, useState } from "react";
import { Download, ExternalLink, Loader2, RefreshCw } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { getSystemRuntimeVersion, getSystemVersion, type SystemRuntimeVersionInfo, type SystemVersionInfo } from "@/lib/api/system";
import { clearIgnoredVersion, getIgnoredVersion, setIgnoredVersion } from "@/lib/version-update";
import { itemVariants, settingsTypography } from "../../_shared";

function formatDateTime(value?: string | null): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function VersionSettings() {
  const [runtimeVersionInfo, setRuntimeVersionInfo] = useState<SystemRuntimeVersionInfo | null>(null);
  const [versionInfo, setVersionInfo] = useState<SystemVersionInfo | null>(null);
  const [runtimeVersionError, setRuntimeVersionError] = useState<string | null>(null);
  const [versionError, setVersionError] = useState<string | null>(null);
  const [loadingRuntimeVersion, setLoadingRuntimeVersion] = useState(true);
  const [loadingVersion, setLoadingVersion] = useState(true);
  const [checkingVersion, setCheckingVersion] = useState(false);
  const [ignoredVersion, setIgnoredVersionState] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const runtimeVersion = await getSystemRuntimeVersion();
        if (cancelled) {
          return;
        }
        setRuntimeVersionInfo(runtimeVersion);
        setRuntimeVersionError(null);
      } catch (err) {
        if (!cancelled) {
          setRuntimeVersionError(getApiErrorMessage(err));
        }
      } finally {
        if (!cancelled) {
          setLoadingRuntimeVersion(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const latest = await getSystemVersion();
        if (cancelled) {
          return;
        }
        setVersionInfo(latest);
        setVersionError(null);
        setIgnoredVersionState(getIgnoredVersion());
      } catch (err) {
        if (!cancelled) {
          setVersionError(getApiErrorMessage(err));
        }
      } finally {
        if (!cancelled) {
          setLoadingVersion(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleCheckVersion = async () => {
    setCheckingVersion(true);
    setRuntimeVersionError(null);
    setVersionError(null);
    const [runtimeResult, versionResult] = await Promise.allSettled([
      getSystemRuntimeVersion(),
      getSystemVersion(true),
    ]);

    if (runtimeResult.status === "fulfilled") {
      setRuntimeVersionInfo(runtimeResult.value);
    } else {
      setRuntimeVersionError(getApiErrorMessage(runtimeResult.reason));
    }

    if (versionResult.status === "fulfilled") {
      setVersionInfo(versionResult.value);
      setIgnoredVersionState(getIgnoredVersion());
    } else {
      setVersionError(getApiErrorMessage(versionResult.reason));
    }
    setCheckingVersion(false);
  };

  const handleIgnoreVersion = () => {
    if (!versionInfo?.latestVersion) return;
    setIgnoredVersion(versionInfo.latestVersion);
    setIgnoredVersionState(versionInfo.latestVersion);
  };

  const handleRestoreVersionReminder = () => {
    clearIgnoredVersion();
    setIgnoredVersionState(null);
  };

  const currentVersionDisplay = runtimeVersionInfo?.currentVersionDisplay
    || (loadingRuntimeVersion ? "读取中..." : "未知版本");
  const latestVersionDisplay = versionInfo?.latestVersionDisplay || "未检测到";
  const hasUpdate = Boolean(versionInfo?.updateAvailable);
  const latestReleaseDisplay = versionInfo?.latestReleaseVersionDisplay || "未检测到";
  const versionBusy = loadingVersion || checkingVersion;
  const ignoredCurrentVersion = Boolean(
    versionInfo?.latestVersion && ignoredVersion === versionInfo.latestVersion
  );
  const versionFailed = Boolean(versionError) || versionInfo?.checkSucceeded === false;
  const versionStatus = loadingVersion
    ? {
        label: "检查中",
        className: "border-sky-500/30 bg-sky-500/10 text-sky-600",
        summary: "页面已加载，正在后台检查最新版本。",
      }
    : versionError
      ? {
          label: "加载失败",
          className: "border-rose-500/30 bg-rose-500/10 text-rose-600",
          summary: "版本信息加载失败，可稍后重试。",
        }
    : !versionInfo
    ? {
        label: "未加载",
        className: "border-border/30 bg-muted/20 text-muted-foreground",
        summary: "尚未获取版本信息。",
      }
    : !versionInfo.checkSucceeded
      ? {
          label: "检查失败",
          className: "border-rose-500/30 bg-rose-500/10 text-rose-600",
          summary: "当前版本已读取，但最新版本检查失败，可稍后重试。",
        }
      : versionInfo.versionRelation === "ahead"
        ? {
            label: "本地较新",
            className: "border-sky-500/30 bg-sky-500/10 text-sky-600",
            summary: "当前运行版本高于线上最新可部署版本，不需要升级。",
          }
      : hasUpdate
        ? {
            label: "可升级",
            className: "border-amber-500/30 bg-amber-500/10 text-amber-600",
            summary: `检测到新版本 ${latestVersionDisplay}，建议安排维护窗口后升级。`,
          }
        : versionInfo.latestReleaseDockerReady === false
          ? {
              label: "镜像待发布",
              className: "border-cyan-500/30 bg-cyan-500/10 text-cyan-600",
              summary: `最新 Release ${latestReleaseDisplay} 已发布，但 Docker 镜像尚未就绪。`,
            }
        : {
            label: "已最新",
            className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-600",
            summary: "当前已是最新发布版本。",
          };

  return (
          <motion.div
            variants={itemVariants}
            className="mt-6 rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-6"
          >
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Download className="h-4 w-4 text-primary" />
                <h3 className={settingsTypography.sectionTitle}>版本更新</h3>
                <span
                  className={cn(
                    "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium border",
                    versionStatus.className
                  )}
                >
                  {versionStatus.label}
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                通过 GitHub Release 检查当前运行版本与最新发布版本，便于管理员决定是否升级部署。
              </p>
            </div>

            <Button variant="outline"
              type="button"
              onClick={handleCheckVersion}
              disabled={versionBusy}
            >
              {versionBusy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              {checkingVersion ? "检查中…" : loadingVersion ? "后台检查中…" : "检查更新"}
            </Button>
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-border/20 bg-muted/10 p-4">
              <p className="text-xs text-muted-foreground mb-1">当前运行版本</p>
              <p className="text-lg font-semibold tracking-tight">{currentVersionDisplay}</p>
              <p className="mt-2 text-xs text-muted-foreground">
                最近检查：{formatDateTime(versionInfo?.checkedAt)}
              </p>
              {runtimeVersionError ? (
                <p className="mt-2 text-xs leading-relaxed text-rose-600" role="alert">
                  {runtimeVersionError}
                </p>
              ) : null}
              {runtimeVersionInfo?.buildProfile ? (
                <p className="mt-1 text-xs text-muted-foreground">
                  运行环境：{runtimeVersionInfo.buildProfile}
                </p>
              ) : null}
            </div>

            <div className="rounded-xl border border-border/20 bg-muted/10 p-4">
              <p className="text-xs text-muted-foreground mb-1">最新可部署版本</p>
              <p className="text-lg font-semibold tracking-tight">{latestVersionDisplay}</p>
              <p className="mt-2 text-xs text-muted-foreground">
                发布时间：{formatDateTime(versionInfo?.publishedAt)}
              </p>
            </div>
          </div>

          {versionInfo?.latestReleaseVersion ? (
            <div className="mt-4 rounded-xl border border-border/20 bg-muted/10 p-4">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-xs text-muted-foreground">GitHub 最新 Release</p>
                <span
                  className={cn(
                    "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium border",
                    versionInfo.latestReleaseDockerReady
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600"
                      : "border-cyan-500/30 bg-cyan-500/10 text-cyan-600"
                  )}
                >
                  {versionInfo.latestReleaseDockerReady ? "镜像已就绪" : "镜像未就绪"}
                </span>
              </div>
              <p className="mt-2 text-lg font-semibold tracking-tight">{latestReleaseDisplay}</p>
              <p className="mt-2 text-xs text-muted-foreground">
                发布时间：{formatDateTime(versionInfo.latestReleasePublishedAt)}
              </p>
            </div>
          ) : null}

          <div
            className={cn(
              "mt-4 rounded-xl border p-4",
              versionFailed
                ? "border-rose-500/20 bg-rose-500/5"
                : hasUpdate
                ? "border-amber-500/20 bg-amber-500/5"
                : "border-emerald-500/20 bg-emerald-500/5"
            )}
          >
            <p className="text-sm font-medium">{versionStatus.summary}</p>

            {versionError || versionInfo?.message ? (
              <p
                className={cn(
                  "mt-2 text-xs leading-relaxed",
                  versionError ? "text-rose-600" : "text-muted-foreground",
                )}
                role={versionError ? "alert" : undefined}
              >
                {versionError || versionInfo?.message}
              </p>
            ) : null}

            {ignoredCurrentVersion ? (
              <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                当前浏览器已忽略 {latestVersionDisplay} 的全局升级提醒。
              </p>
            ) : null}

            <div className="mt-4 flex flex-wrap gap-2">
              {hasUpdate ? (
                ignoredCurrentVersion ? (
                  <Button variant="outline"
                    type="button"
                    onClick={handleRestoreVersionReminder}
                  >
                    恢复提醒
                  </Button>
                ) : (
                  <Button variant="outline"
                    type="button"
                    onClick={handleIgnoreVersion}
                  >
                    忽略此版本提醒
                  </Button>
                )
              ) : null}

              <a
                href={versionInfo?.latestReleaseUrl || "https://github.com/Stonewuu/ai-fusion-video/releases"}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-xl border border-border/30 bg-background/80 px-4 py-2 text-sm font-medium transition-colors hover:bg-muted/20"
              >
                <ExternalLink className="h-4 w-4" />
                查看最新 Release
              </a>

              {versionInfo?.latestVersion ? (
                <a
                  href={versionInfo?.tagUrl || "https://github.com/Stonewuu/ai-fusion-video/releases"}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 rounded-xl border border-border/30 bg-background/80 px-4 py-2 text-sm font-medium transition-colors hover:bg-muted/20"
                >
                  <ExternalLink className="h-4 w-4" />
                  查看可部署版本
                </a>
              ) : null}
            </div>

            {hasUpdate ? (
              <div className="mt-4 grid gap-3 lg:grid-cols-2">
                <div className="rounded-lg border border-border/20 bg-background/70 p-3">
                  <p className="mb-2 text-xs font-medium text-foreground/80">Docker 部署</p>
                  <pre className="overflow-x-auto whitespace-pre-wrap break-all text-xs text-muted-foreground">
docker compose pull
docker compose up -d
                  </pre>
                </div>

                <div className="rounded-lg border border-border/20 bg-background/70 p-3">
                  <p className="mb-2 text-xs font-medium text-foreground/80">源码部署</p>
                  <pre className="overflow-x-auto whitespace-pre-wrap break-all text-xs text-muted-foreground">
git pull
cd ai-fusion-video
./mvnw clean package
                  </pre>
                </div>
              </div>
            ) : null}

            <p className="mt-4 text-xs text-muted-foreground leading-relaxed">
              版本源：GitHub Release。当前接口会缓存检查结果，点击“检查更新”可立即强制刷新。
            </p>
          </div>
          </motion.div>
  );
}
