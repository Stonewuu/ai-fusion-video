"use client";

import { useState, useEffect } from "react";
import {
  Globe,
  Save,
  Loader2,
  AlertTriangle,
  RefreshCw,
  ExternalLink,
  Download,
  Mail,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { http } from "@/lib/api/client";
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
import { containerVariants, itemVariants } from "../_shared";

interface SystemConfigs {
  site_base_url: string;
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

export default function GeneralSettingsPage() {
  const currentUser = useAuthStore((state) => state.user);
  const isAdmin = currentUser?.roles?.includes("admin") ?? false;
  const [configs, setConfigs] = useState<SystemConfigs>({
    site_base_url: "",
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
  const [loadingConfigs, setLoadingConfigs] = useState(true);
  const [loadingRuntimeVersion, setLoadingRuntimeVersion] = useState(true);
  const [loadingVersion, setLoadingVersion] = useState(true);
  const [saving, setSaving] = useState(false);
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
      } catch (err) {
        if (!cancelled) {
          console.error("加载当前运行版本失败:", err);
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
        const loaded = {
          site_base_url: map.site_base_url || "",
          allow_register: map.allow_register === "true",
          mail_smtp_host: map.mail_smtp_host || "",
          mail_smtp_port: map.mail_smtp_port || "",
          mail_username: map.mail_username || "",
          mail_password: map.mail_password || "",
          mail_ssl: map.mail_ssl === "true",
          mail_from: map.mail_from || "",
        };
        setConfigs(loaded);
        setOriginal(loaded);
      } catch (err) {
        if (!cancelled) {
          console.error("加载系统配置失败:", err);
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
        setIgnoredVersionState(getIgnoredVersion());
      } catch (err) {
        if (!cancelled) {
          console.error("加载版本信息失败:", err);
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
      await http.put("/api/system/config", {
        site_base_url: configs.site_base_url,
        allow_register: String(configs.allow_register),
        mail_smtp_host: configs.mail_smtp_host || "",
        mail_smtp_port: configs.mail_smtp_port || "",
        mail_username: configs.mail_username || "",
        mail_password: configs.mail_password || "",
        mail_ssl: String(!!configs.mail_ssl),
        mail_from: configs.mail_from || "",
      });
      setOriginal({ ...configs });
    } catch (err) {
      console.error("保存系统配置失败:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleCheckVersion = async () => {
    setCheckingVersion(true);
    try {
      const latest = await getSystemVersion(true);
      setVersionInfo(latest);
      setIgnoredVersionState(getIgnoredVersion());
    } catch (err) {
      console.error("检查更新失败:", err);
    } finally {
      setCheckingVersion(false);
    }
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
  const versionStatus = loadingVersion
    ? {
        label: "检查中",
        className: "border-sky-500/30 bg-sky-500/10 text-sky-600",
        summary: "页面已加载，正在后台检查最新版本。",
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
            <h1 className="text-2xl font-bold tracking-tight">通用设置</h1>
            <p className="text-muted-foreground mt-1 text-sm">
              配置站点访问域名等全局参数
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
          <div className="flex items-center gap-2 mb-4">
            <Globe className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">项目访问域名</h3>
          </div>

          <p className="text-xs text-muted-foreground mb-4 leading-relaxed">
            配置本项目部署后的完整访问域名（不含末尾斜杠）。
            系统将使用该域名拼接内部资源的公网访问地址，供外部服务和 API 调用。
          </p>

          <input
            type="url"
            value={configs.site_base_url}
            onChange={(e) =>
              setConfigs((prev) => ({ ...prev, site_base_url: e.target.value }))
            }
            placeholder="https://fusion.example.com"
            className={cn(
              "w-full px-4 py-2.5 rounded-xl text-sm",
              "bg-muted/30 border border-border/30",
              "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
              "placeholder:text-muted-foreground/40"
            )}
          />

          <div className="mt-4 space-y-3">
            <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/10 border border-border/20">
              <div className="text-xs text-muted-foreground leading-relaxed">
                <p className="mb-1">
                  <strong>示例：</strong>
                </p>
                <ul className="list-disc list-inside space-y-0.5">
                  <li>本地开发：<code className="text-foreground/80">http://localhost:8080</code></li>
                  <li>内网部署：<code className="text-foreground/80">http://192.168.1.100:8080</code></li>
                  <li>公网部署：<code className="text-foreground/80">https://fusion.example.com</code></li>
                </ul>
              </div>
            </div>

            <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
              <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
              <p className="text-xs text-muted-foreground leading-relaxed">
                <strong>备注：</strong>画风参考图生图功能依赖此配置或对象存储。
                使用本地存储时，需要配置此域名才能将参考图传递给 AI API；
                若已配置对象存储，上传的图片会自动获得公网 URL，此项可不填。
              </p>
            </div>
          </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="mt-6 rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-6"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-2">
                <h3 className="text-sm font-semibold">公开注册</h3>
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
              <h3 className="text-sm font-semibold">邮箱 SMTP 配置</h3>
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
                <h3 className="text-sm font-semibold">版本更新</h3>
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
              hasUpdate
                ? "border-amber-500/20 bg-amber-500/5"
                : "border-emerald-500/20 bg-emerald-500/5"
            )}
          >
            <p className="text-sm font-medium">{versionStatus.summary}</p>

            {versionInfo?.message ? (
              <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                {versionInfo.message}
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
