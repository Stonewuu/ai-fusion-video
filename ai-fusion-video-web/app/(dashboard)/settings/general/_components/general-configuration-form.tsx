"use client";

import type { Dispatch, SetStateAction } from "react";
import { Globe, Mail } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { itemVariants, settingsTypography } from "../../_shared";
import { isLocalOrPrivateUrl, type SystemConfigs } from "./general-settings-types";

interface GeneralConfigurationFormProps {
  configs: SystemConfigs;
  setConfigs: Dispatch<SetStateAction<SystemConfigs>>;
  isAdmin: boolean;
  saving: boolean;
  separatePublicOrigins: boolean;
  setSeparatePublicOrigins: Dispatch<SetStateAction<boolean>>;
}

export function GeneralConfigurationForm({ configs, setConfigs, isAdmin, saving, separatePublicOrigins, setSeparatePublicOrigins }: GeneralConfigurationFormProps) {
  return (<>
          <motion.div
            variants={itemVariants}
            className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-6"
          >
            <div className="flex flex-col items-start gap-4 sm:flex-row sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <Globe className="h-4 w-4 text-primary" />
                  <h3 className={settingsTypography.sectionTitle}>公网访问地址</h3>
                </div>
                <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                  已根据当前访问入口预填。云端模型读取资源时，需要可公网访问的资源地址。
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <span className="text-xs text-muted-foreground">前后端不同域名</span>
                <button
                  type="button"
                  disabled={!isAdmin || saving}
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
                  <label htmlFor="site_base_url" className="block text-xs font-medium mb-1.5">前端站点地址</label>
                  <input id="site_base_url"
                    type="url"
                    disabled={!isAdmin || saving}
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
                  <label htmlFor="resource_base_url" className="block text-xs font-medium mb-1.5">后端资源地址</label>
                  <input id="resource_base_url"
                    type="url"
                    disabled={!isAdmin || saving}
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
                  disabled={!isAdmin || saving}
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
                disabled={!isAdmin || saving}
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
                <label htmlFor="mail_smtp_host" className="block text-xs text-muted-foreground mb-1.5 font-medium">SMTP 服务器地址</label>
                <input id="mail_smtp_host"
                  type="text"
                  disabled={!isAdmin || saving}
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
                <label htmlFor="mail_smtp_port" className="block text-xs text-muted-foreground mb-1.5 font-medium">SMTP 端口</label>
                <input id="mail_smtp_port"
                  type="text"
                  disabled={!isAdmin || saving}
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
                <label htmlFor="mail_username" className="block text-xs text-muted-foreground mb-1.5 font-medium">发件人邮箱账户</label>
                <input id="mail_username"
                  type="email"
                  disabled={!isAdmin || saving}
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
                <label htmlFor="mail_password" className="block text-xs text-muted-foreground mb-1.5 font-medium">发件箱密码/授权码</label>
                <input id="mail_password"
                  type="password"
                  disabled={!isAdmin || saving}
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
                <label htmlFor="mail_from" className="block text-xs text-muted-foreground mb-1.5 font-medium">自定义发件人名称</label>
                <input id="mail_from"
                  type="text"
                  disabled={!isAdmin || saving}
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
                  disabled={!isAdmin || saving}
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

  </>);
}
