"use client";

import { useState, useEffect } from "react";
import {
  Globe,
  Save,
  Loader2,
  RefreshCw,
  ExternalLink,
  Download,
  Mail,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { API_BASE_URL, http } from "@/lib/api/client";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import {
  getSystemRuntimeVersion,
  getSystemVersion,
  type SystemRuntimeVersionInfo,
  type SystemVersionInfo,
} from "@/lib/api/system";
import {
  clearIgnoredVersion,
  getIgnoredVersion,
  setIgnoredVersion,
} from "@/lib/version-update";
import { containerVariants, itemVariants, settingsTypography } from "../_shared";

interface SystemConfigs {
  site_base_url: string;
  resource_base_url: string;
  allow_register: boolean;
  mail_smtp_host?: string;
  mail_smtp_port?: string;
  mail_username?: string;
  mail_password?: string;
  mail_ssl?: boolean;
  mail_from?: string;
}

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

function normalizePublicUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

function isLocalOrPrivateUrl(value: string): boolean {
  if (!value.trim()) return false;
  try {
    const hostname = new URL(value).hostname.toLowerCase();
    if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1") {
      return true;
    }
    if (/^10\./.test(hostname) || /^192\.168\./.test(hostname)) {
      return true;
    }
    const match = hostname.match(/^172\.(\d+)\./);
    return Boolean(match && Number(match[1]) >= 16 && Number(match[1]) <= 31);
  } catch {
    return false;
  }
}

export default function GeneralSettingsPage() {
  const currentUser = useAuthStore((state) => state.user);
  const isAdmin = currentUser?.roles?.includes("admin") ?? false;
  const [configs, setConfigs] = useState<SystemConfigs>({
    site_base_url: "",
    resource_base_url: "",
    allow_register: false,
    mail_smtp_host: "",
    mail_smtp_port: "",
    mail_username: "",
    mail_password: "",
    mail_ssl: false,
    mail_from: "",
  });
  const [original, setOriginal] = useState<SystemConfigs>({
    site_base_url: "",
    resource_base_url: "",
    allow_register: false,
    mail_smtp_host: "",
    mail_smtp_port: "",
    mail_username: "",
    mail_password: "",
    mail_ssl: false,
    mail_from: "",
  });
  const [runtimeVersionInfo, setRuntimeVersionInfo] = useState<SystemRuntimeVersionInfo | null>(null);
  const [versionInfo, setVersionInfo] = useState<SystemVersionInfo | null>(null);
  const [runtimeVersionError, setRuntimeVersionError] = useState<string | null>(null);
  const [versionError, setVersionError] = useState<string | null>(null);
  const [loadingConfigs, setLoadingConfigs] = useState(true);
  const [loadingRuntimeVersion, setLoadingRuntimeVersion] = useState(true);
  const [loadingVersion, setLoadingVersion] = useState(true);
  const [saving, setSaving] = useState(false);
  const [checkingVersion, setCheckingVersion] = useState(false);
  const [ignoredVersion, setIgnoredVersionState] = useState<string | null>(null);
  const [separatePublicOrigins, setSeparatePublicOrigins] = useState(false);

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
        const configResult = await http.get<never, { configKey: string; configValue: string }[]>(
          "/api/system/config"
        );
        if (cancelled) {
          return;
        }

        const map: Record<string, string> = {};
        configResult.forEach((c) => {
          map[c.configKey] = c.configValue || "";
        });
        const stored = {
          site_base_url: map.site_base_url || "",
          resource_base_url: map.resource_base_url || "",
          allow_register: map.allow_register === "true",
          mail_smtp_host: map.mail_smtp_host || "",
          mail_smtp_port: map.mail_smtp_port || "",
          mail_username: map.mail_username || "",
          mail_password: map.mail_password || "",
          mail_ssl: map.mail_ssl === "true",
          mail_from: map.mail_from || "",
        };
        const detectedSiteUrl = normalizePublicUrl(window.location.origin);
        const detectedResourceUrl = normalizePublicUrl(
          API_BASE_URL ? API_BASE_URL : detectedSiteUrl,
        );
        const loaded = {
          ...stored,
          site_base_url: stored.site_base_url
            ? stored.site_base_url
            : detectedSiteUrl,
          resource_base_url: stored.resource_base_url
            ? stored.resource_base_url
            : detectedResourceUrl,
        };
        setSeparatePublicOrigins(
          normalizePublicUrl(loaded.site_base_url) !==
            normalizePublicUrl(loaded.resource_base_url),
        );
        setConfigs(loaded);
        setOriginal(stored);
      } catch (err) {
        if (!cancelled) {
          console.error("加载系统配置失败:", err);
          toastApiError(err, "加载系统配置失败");
        }
      } finally {
        if (!cancelled) {
          setLoadingConfigs(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (loadingConfigs) {
      return;
    }

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
  }, [loadingConfigs]);

  const hasChanges =
    configs.site_base_url !== original.site_base_url ||
    configs.resource_base_url !== original.resource_base_url ||
    configs.allow_register !== original.allow_register ||
    configs.mail_smtp_host !== original.mail_smtp_host ||
    configs.mail_smtp_port !== original.mail_smtp_port ||
    configs.mail_username !== original.mail_username ||
    configs.mail_password !== original.mail_password ||
    configs.mail_ssl !== original.mail_ssl ||
    configs.mail_from !== original.mail_from;

  const handleSave = async () => {
    setSaving(true);
    try {
      const normalizedConfigs = {
        ...configs,
        site_base_url: normalizePublicUrl(configs.site_base_url),
        resource_base_url: normalizePublicUrl(configs.resource_base_url),
      };
      await http.put("/api/system/config", {
        site_base_url: normalizedConfigs.site_base_url,
        resource_base_url: normalizedConfigs.resource_base_url,
        allow_register: String(configs.allow_register),
        mail_smtp_host: configs.mail_smtp_host || "",
        mail_smtp_port: configs.mail_smtp_port || "",
        mail_username: configs.mail_username || "",
        mail_password: configs.mail_password || "",
        mail_ssl: String(!!configs.mail_ssl),
        mail_from: configs.mail_from || "",
      });
      setConfigs(normalizedConfigs);
      setOriginal(normalizedConfigs);
    } catch (err) {
      console.error("保存系统配置失败:", err);
      toastApiError(err, "保存系统配置失败");
    } finally {
      setSaving(false);
    }
  };

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
      className="w-full"
      variants={containerVariants}
      initial={false}
      animate="visible"
    >
      {/* 标题 */}
      <motion.div variants={itemVariants} className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className={settingsTypography.pageTitle}>通用设置</h1>
            <p className={settingsTypography.pageDescription}>
              管理系统全局参数
            </p>
            {!isAdmin ? (
              <p className="text-xs text-amber-600 mt-2">
                当前账号只能查看系统设置，只有管理员可以修改。
              </p>
            ) : null}
          </div>
          <button
            onClick={handleSave}
            disabled={!isAdmin || !hasChanges || saving}
            className={cn(
              "flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-medium transition-all duration-200",
              isAdmin && hasChanges
                ? "bg-primary text-primary-foreground shadow-sm hover:opacity-90"
                : "bg-muted/50 text-muted-foreground cursor-not-allowed border border-border/30"
            )}
          >
            {saving ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            {saving ? "保存中…" : "保存"}
          </button>
        </div>
      </motion.div>

      {loadingConfigs ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <>
          <motion.div
            variants={itemVariants}
            className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-6"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <Globe className="h-4 w-4 text-primary" />
                  <h3 className={settingsTypography.sectionTitle}>公网访问地址</h3>
                </div>
                <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                  已根据当前访问环境自动识别，通常无需修改。
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">前后端不同域名</span>
                <button
                  type="button"
                  disabled={!isAdmin}
                  onClick={() => {
                    if (!isAdmin) return;
                    setSeparatePublicOrigins((current) => {
                      const next = !current;
                      if (!next) {
                        setConfigs((previous) => ({
                          ...previous,
                          resource_base_url: previous.site_base_url,
                        }));
                      }
                      return next;
                    });
                  }}
                  className={cn(
                    "relative inline-flex h-7 w-12 shrink-0 rounded-full border transition-colors",
                    separatePublicOrigins
                      ? "border-primary/40 bg-primary/20"
                      : "border-border/40 bg-muted/30",
                    !isAdmin ? "cursor-not-allowed opacity-60" : "cursor-pointer"
                  )}
                  aria-label="切换前后端不同域名"
                  aria-pressed={separatePublicOrigins}
                >
                  <span
                    className={cn(
                      "absolute top-0.5 h-5.5 w-5.5 rounded-full bg-white shadow transition-transform",
                      separatePublicOrigins ? "translate-x-6" : "translate-x-0.5"
                    )}
                  />
                </button>
              </div>
            </div>

            {separatePublicOrigins ? (
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div>
                  <label className="block text-xs font-medium mb-1.5">前端站点地址</label>
                  <input
                    type="url"
                    disabled={!isAdmin}
                    value={configs.site_base_url}
                    onChange={(e) =>
                      setConfigs((prev) => ({ ...prev, site_base_url: e.target.value }))
                    }
                    placeholder="https://app.example.com"
                    className={cn(
                      "w-full px-4 py-2.5 rounded-xl text-sm",
                      "bg-muted/30 border border-border/30",
                      "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                      "placeholder:text-muted-foreground/40 disabled:opacity-60"
                    )}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1.5">后端资源地址</label>
                  <input
                    type="url"
                    disabled={!isAdmin}
                    value={configs.resource_base_url}
                    onChange={(e) =>
                      setConfigs((prev) => ({ ...prev, resource_base_url: e.target.value }))
                    }
                    placeholder="https://api.example.com"
                    className={cn(
                      "w-full px-4 py-2.5 rounded-xl text-sm",
                      "bg-muted/30 border border-border/30",
                      "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                      "placeholder:text-muted-foreground/40 disabled:opacity-60"
                    )}
                  />
                </div>
              </div>
            ) : (
              <div className="mt-4">
                <input
                  type="url"
                  aria-label="公网访问地址"
                  disabled={!isAdmin}
                  value={configs.site_base_url}
                  onChange={(e) =>
                    setConfigs((prev) => ({
                      ...prev,
                      site_base_url: e.target.value,
                      resource_base_url: e.target.value,
                    }))
                  }
                  placeholder="https://fusion.example.com"
                  className={cn(
                    "w-full px-4 py-2.5 rounded-xl text-sm",
                    "bg-muted/30 border border-border/30",
                    "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                    "placeholder:text-muted-foreground/40 disabled:opacity-60"
                  )}
                />
              </div>
            )}

            {isLocalOrPrivateUrl(configs.resource_base_url) ? (
              <p className="mt-3 text-xs text-amber-600">
                当前是本地或内网地址，云端 AI 无法直接读取本地资源；需要 URL 传图时请使用对象存储或公网隧道。
              </p>
            ) : null}
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="mt-6 rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-6"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-2">
                <h3 className={settingsTypography.sectionTitle}>公开注册</h3>
                <p className="text-xs text-muted-foreground leading-relaxed max-w-[520px]">
                  仅在系统完成管理员初始化后生效。开启后，访客可以通过用户名和密码注册账号。
                </p>
              </div>

              <button
                type="button"
                disabled={!isAdmin}
                onClick={() => {
                  if (!isAdmin) return;
                  setConfigs((prev) => ({ ...prev, allow_register: !prev.allow_register }));
                }}
                className={cn(
                  "relative inline-flex h-7 w-12 shrink-0 rounded-full border transition-colors",
                  configs.allow_register
                    ? "border-emerald-500/40 bg-emerald-500/20"
                    : "border-border/40 bg-muted/30",
                  !isAdmin ? "cursor-not-allowed opacity-60" : "cursor-pointer"
                )}
                aria-label="切换公开注册"
                aria-pressed={configs.allow_register}
              >
                <span
                  className={cn(
                    "absolute top-0.5 h-5.5 w-5.5 rounded-full bg-white shadow transition-transform",
                    configs.allow_register ? "translate-x-6" : "translate-x-0.5"
                  )}
                />
              </button>
            </div>

            <div className="mt-4 rounded-lg border border-border/20 bg-muted/10 p-3 text-xs text-muted-foreground">
              当前状态：
              <span
                className={cn(
                  "ml-2 font-medium",
                  configs.allow_register ? "text-emerald-600" : "text-foreground/80"
                )}
              >
                {configs.allow_register ? "已开启" : "未开启"}
              </span>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="mt-6 rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <Mail className="h-4 w-4 text-primary" />
              <h3 className={settingsTypography.sectionTitle}>邮箱 SMTP 配置</h3>
            </div>

            <p className="text-xs text-muted-foreground mb-4 leading-relaxed">
              配置系统邮件发送服务，用于发送系统通知、验证码等邮件。
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <label className="block text-xs text-muted-foreground mb-1.5 font-medium">SMTP 服务器地址</label>
                <input
                  type="text"
                  disabled={!isAdmin}
                  value={configs.mail_smtp_host || ""}
                  onChange={(e) =>
                    setConfigs((prev) => ({ ...prev, mail_smtp_host: e.target.value }))
                  }
                  placeholder="smtp.qq.com"
                  className={cn(
                    "w-full px-4 py-2.5 rounded-xl text-sm",
                    "bg-muted/30 border border-border/30",
                    "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                    "placeholder:text-muted-foreground/40 disabled:opacity-60"
                  )}
                />
              </div>

              <div>
                <label className="block text-xs text-muted-foreground mb-1.5 font-medium">SMTP 端口</label>
                <input
                  type="text"
                  disabled={!isAdmin}
                  value={configs.mail_smtp_port || ""}
                  onChange={(e) =>
                    setConfigs((prev) => ({ ...prev, mail_smtp_port: e.target.value }))
                  }
                  placeholder="465"
                  className={cn(
                    "w-full px-4 py-2.5 rounded-xl text-sm",
                    "bg-muted/30 border border-border/30",
                    "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                    "placeholder:text-muted-foreground/40 disabled:opacity-60"
                  )}
                />
              </div>

              <div>
                <label className="block text-xs text-muted-foreground mb-1.5 font-medium">发件人邮箱账户</label>
                <input
                  type="email"
                  disabled={!isAdmin}
                  value={configs.mail_username || ""}
                  onChange={(e) =>
                    setConfigs((prev) => ({ ...prev, mail_username: e.target.value }))
                  }
                  placeholder="your-email@qq.com"
                  className={cn(
                    "w-full px-4 py-2.5 rounded-xl text-sm",
                    "bg-muted/30 border border-border/30",
                    "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                    "placeholder:text-muted-foreground/40 disabled:opacity-60"
                  )}
                />
              </div>

              <div>
                <label className="block text-xs text-muted-foreground mb-1.5 font-medium">发件箱密码/授权码</label>
                <input
                  type="password"
                  disabled={!isAdmin}
                  value={configs.mail_password || ""}
                  onChange={(e) =>
                    setConfigs((prev) => ({ ...prev, mail_password: e.target.value }))
                  }
                  placeholder="••••••••••••••••"
                  className={cn(
                    "w-full px-4 py-2.5 rounded-xl text-sm",
                    "bg-muted/30 border border-border/30",
                    "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                    "placeholder:text-muted-foreground/40 disabled:opacity-60"
                  )}
                />
              </div>

              <div>
                <label className="block text-xs text-muted-foreground mb-1.5 font-medium">自定义发件人名称</label>
                <input
                  type="text"
                  disabled={!isAdmin}
                  value={configs.mail_from || ""}
                  onChange={(e) =>
                    setConfigs((prev) => ({ ...prev, mail_from: e.target.value }))
                  }
                  placeholder="融光视频平台 &lt;your-email@qq.com&gt;"
                  className={cn(
                    "w-full px-4 py-2.5 rounded-xl text-sm",
                    "bg-muted/30 border border-border/30",
                    "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                    "placeholder:text-muted-foreground/40 disabled:opacity-60"
                  )}
                />
              </div>

              <div className="flex items-center justify-between mt-6 md:mt-8">
                <div className="space-y-0.5">
                  <label className="block text-xs text-muted-foreground font-medium">启用 SSL/TLS</label>
                  <p className="text-[10px] text-muted-foreground/75 leading-none">常用 SSL 端口如 465 需要开启此项</p>
                </div>
                <button
                  type="button"
                  disabled={!isAdmin}
                  onClick={() => {
                    if (!isAdmin) return;
                    setConfigs((prev) => ({ ...prev, mail_ssl: !prev.mail_ssl }));
                  }}
                  className={cn(
                    "relative inline-flex h-7 w-12 shrink-0 rounded-full border transition-colors",
                    configs.mail_ssl
                      ? "border-emerald-500/40 bg-emerald-500/20"
                      : "border-border/40 bg-muted/30",
                    !isAdmin ? "cursor-not-allowed opacity-60" : "cursor-pointer"
                  )}
                  aria-label="切换邮件SSL"
                  aria-pressed={configs.mail_ssl}
                >
                  <span
                    className={cn(
                      "absolute top-0.5 h-5.5 w-5.5 rounded-full bg-white shadow transition-transform",
                      configs.mail_ssl ? "translate-x-6" : "translate-x-0.5"
                    )}
                  />
                </button>
              </div>
            </div>
          </motion.div>

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

            <button
              type="button"
              onClick={handleCheckVersion}
              disabled={versionBusy}
              className={cn(
                "inline-flex items-center justify-center gap-2 rounded-xl border px-4 py-2 text-sm font-medium transition-colors",
                "border-border/30 bg-muted/20 hover:bg-muted/40 disabled:cursor-not-allowed disabled:opacity-60"
              )}
            >
              {versionBusy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              {checkingVersion ? "检查中…" : loadingVersion ? "后台检查中…" : "检查更新"}
            </button>
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

            <div className="mt-4 flex flex-wrap gap-3">
              {hasUpdate ? (
                ignoredCurrentVersion ? (
                  <button
                    type="button"
                    onClick={handleRestoreVersionReminder}
                    className="inline-flex items-center gap-2 rounded-xl border border-border/30 bg-background/80 px-4 py-2 text-sm font-medium transition-colors hover:bg-muted/20"
                  >
                    恢复提醒
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleIgnoreVersion}
                    className="inline-flex items-center gap-2 rounded-xl border border-border/30 bg-background/80 px-4 py-2 text-sm font-medium transition-colors hover:bg-muted/20"
                  >
                    忽略此版本提醒
                  </button>
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
        </>
      )}
    </motion.div>
  );
}
