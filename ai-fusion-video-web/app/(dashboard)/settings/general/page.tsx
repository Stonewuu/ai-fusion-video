"use client";

import { useState, useEffect } from "react";
import { Save, Loader2, RefreshCw } from "lucide-react";
import { motion } from "framer-motion";
import { getApiBaseUrl, http } from "@/lib/api/client";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { httpAddressIssue } from "@/lib/settings/form-validation";
import { useAuthStore } from "@/lib/store/auth-store";
import { containerVariants, itemVariants, settingsTypography } from "../_shared";
import { normalizePublicUrl, type SystemConfigs } from "./_components/general-settings-types";
import { GeneralConfigurationForm } from "./_components/general-configuration-form";
import { VersionSettings } from "./_components/version-settings";

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
  const [loadingConfigs, setLoadingConfigs] = useState(true);
  const [saving, setSaving] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [configRetry, setConfigRetry] = useState(0);
  const [separatePublicOrigins, setSeparatePublicOrigins] = useState(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        setLoadingConfigs(true);
        setConfigError(null);
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
          getApiBaseUrl() || detectedSiteUrl,
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
          setConfigError(getApiErrorMessage(err));
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
  }, [configRetry]);

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
    if (!isAdmin || loadingConfigs || configError || saving) return;
    for (const [label, value] of [["公网访问地址", configs.site_base_url], ["后端资源地址", configs.resource_base_url]]) {
      const error = value.trim() ? httpAddressIssue(value, label) : null;
      if (error) { setSaveError(error); return; }
    }
    if (configs.mail_smtp_port?.trim() && (!Number.isInteger(Number(configs.mail_smtp_port)) || Number(configs.mail_smtp_port) < 1 || Number(configs.mail_smtp_port) > 65535)) {
      setSaveError("SMTP 端口需为 1–65535 的整数。");
      document.getElementById("mail_smtp_port")?.focus();
      return;
    }
    setSaveError(null);
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
      toast.success("通用设置已保存");
    } catch (err) {
      setSaveError(getApiErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <motion.div
      className="w-full"
      onChange={() => setSaveError(null)}
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
          <div className="flex items-center gap-3">
          {isAdmin && !loadingConfigs && !configError && <span className="text-xs text-muted-foreground" role="status">{hasChanges ? "有未保存的配置" : "已保存"}</span>}
          <Button
            onClick={handleSave}
            disabled={!isAdmin || !hasChanges || saving || loadingConfigs || !!configError}
          >
            {saving ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            {saving ? "保存中…" : "保存"}
          </Button>
          </div>
        </div>
        {saveError && <p className="mt-3 text-sm text-destructive" role="alert">{saveError}</p>}
      </motion.div>

      {loadingConfigs ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : configError ? (
        <div className="space-y-3 rounded-xl border border-border/30 bg-card p-6">
          <p role="alert" className="text-sm text-destructive">配置加载失败：{configError}</p>
          <Button variant="outline" onClick={() => setConfigRetry(value => value + 1)}><RefreshCw />重试</Button>
        </div>
      ) : (
        <>
          <GeneralConfigurationForm configs={configs} setConfigs={setConfigs} isAdmin={isAdmin} saving={saving} separatePublicOrigins={separatePublicOrigins} setSeparatePublicOrigins={setSeparatePublicOrigins} />
          <VersionSettings />

        </>
      )}
    </motion.div>
  );
}
