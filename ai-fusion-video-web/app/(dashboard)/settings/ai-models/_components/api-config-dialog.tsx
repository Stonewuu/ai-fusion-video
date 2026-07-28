"use client";

import { useEffect, useState } from "react";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  apiConfigApi,
  PLATFORM_OPTIONS,
  type ApiConfig,
  type ApiConfigSaveReq,
} from "@/lib/api/ai-model";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { getPlatformFields } from "../../_shared";
import { getProtocolOptionsForModelType } from "./model-config-support";

export interface ApiConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingConfig: ApiConfig | null;
  onSaved: () => void;
}

export const PROXY_TYPE_OPTIONS = [
  { value: "none", label: "不使用代理", description: "直连模型服务" },
  { value: "socks5", label: "SOCKS5", description: "常见本地代理，如 127.0.0.1:7890" },
  { value: "http", label: "HTTP", description: "HTTP/HTTPS 出站代理" },
] as const;

export const UNSET_PROTOCOL_VALUE = "__unset_protocol__";

export const API_PROVIDER_PRESETS = [
  { id: "deepseek", platform: "deepseek", label: "DeepSeek", url: "https://api.deepseek.com", textProtocol: "deepseek", imageProtocol: "", videoProtocol: "" },
  { id: "dashscope", platform: "dashscope", label: "通义千问", url: "https://dashscope.aliyuncs.com", textProtocol: "dashscope", imageProtocol: "dashscope", videoProtocol: "dashscope" },
  { id: "openai", platform: "openai_compatible", label: "OpenAI", url: "https://api.openai.com", textProtocol: "openai_compatible", imageProtocol: "openai", videoProtocol: "openai" },
  { id: "agnes", platform: "openai_compatible", label: "Agnes AI", url: "https://apihub.agnes-ai.com", textProtocol: "openai_compatible", imageProtocol: "agnes", videoProtocol: "agnes" },
] as const;

export function ApiConfigDialog({ open, onOpenChange, editingConfig, onSaved }: ApiConfigDialogProps) {
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<ApiConfigSaveReq>({ name: "" });
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (open) {
      if (editingConfig) {
        setForm({
          id: editingConfig.id,
          name: editingConfig.name,
          platform: editingConfig.platform || "",
          textProtocol: editingConfig.textProtocol || "",
          imageProtocol: editingConfig.imageProtocol || "",
          videoProtocol: editingConfig.videoProtocol || "",
          apiUrl: editingConfig.apiUrl || "",
          autoAppendV1Path: editingConfig.autoAppendV1Path ?? true,
          proxyType: editingConfig.proxyType || "none",
          proxyHost: editingConfig.proxyHost || "",
          proxyPort: editingConfig.proxyPort ?? undefined,
          proxyUsername: editingConfig.proxyUsername || "",
          proxyPassword: editingConfig.proxyPassword || "",
          apiKey: editingConfig.apiKey || "",
          appId: editingConfig.appId || "",
          appSecret: editingConfig.appSecret || "",
          status: editingConfig.status,
          remark: editingConfig.remark || "",
        });
      } else {
        setForm({ name: "", platform: "openai_compatible", textProtocol: "openai_compatible", imageProtocol: "openai", videoProtocol: "openai", apiUrl: "", autoAppendV1Path: true, proxyType: "none", proxyHost: "", proxyPort: undefined, proxyUsername: "", proxyPassword: "", apiKey: "", appId: "", appSecret: "", status: 1 });
      }
      setShowSecrets({});
    }
  }, [open, editingConfig]);

  const updateField = <K extends keyof ApiConfigSaveReq>(key: K, value: ApiConfigSaveReq[K]) => {
    setForm(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    if (!form.name.trim()) return;
    setSaving(true);
    try {
      const proxyEnabled = form.proxyType && form.proxyType !== "none";
      const payload: ApiConfigSaveReq = {
        ...form,
        proxyType: proxyEnabled ? form.proxyType : "none",
        proxyHost: proxyEnabled ? form.proxyHost?.trim() : "",
        proxyPort: proxyEnabled ? form.proxyPort : undefined,
        proxyUsername: proxyEnabled ? form.proxyUsername?.trim() : "",
        proxyPassword: proxyEnabled && form.proxyUsername?.trim() ? form.proxyPassword : "",
      };
      if (editingConfig) {
        await apiConfigApi.update(payload);
      } else {
        await apiConfigApi.create(payload);
      }
      onSaved();
      onOpenChange(false);
    } catch (err) {
      console.error("保存 API 配置失败:", err);
      toastApiError(err, "保存 API 配置失败");
    } finally {
      setSaving(false);
    }
  };

  const fields = getPlatformFields(form.platform || "");
  const proxyEnabled = form.proxyType && form.proxyType !== "none";
  const proxyInvalid = Boolean(proxyEnabled && (
    !form.proxyHost?.trim() || !form.proxyPort || form.proxyPort < 1 || form.proxyPort > 65535
    || (!!form.proxyPassword && !form.proxyUsername?.trim())
  ));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>{editingConfig ? "编辑 API 配置" : "新建 API 配置"}</DialogTitle>
          <DialogDescription>配置连接与鉴权，并为文本、图片、视频分别设置默认请求协议；模型可按需覆盖。</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 overflow-y-auto min-h-0 px-1 -mx-1">
          {/* 配置名称 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">配置名称</Label>
            <Input
              placeholder="例如：DeepSeek / Gemini"
              value={form.name}
              onChange={e => updateField("name", e.target.value)}
              className="text-sm"
            />
          </div>

          {/* 接入与鉴权类型 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">接入与鉴权类型</Label>
            <Select
              value={form.platform || "openai_compatible"}
              onValueChange={v => {
                updateField("platform", v as string);
                if (v === "openai_compatible") {
                  updateField("autoAppendV1Path", true);
                }
              }}
              items={PLATFORM_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
            >
              <SelectTrigger className="w-full text-sm">
                <SelectValue placeholder="选择接入类型" />
              </SelectTrigger>
              <SelectContent className="text-sm">
                <SelectGroup>
                  {PLATFORM_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      <div>
                        <div>{opt.label}</div>
                        <div className="text-[10px] text-muted-foreground">{opt.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            <p className="text-[10px] leading-4 text-muted-foreground">
              决定认证字段、地址规则和远程模型发现方式；具体接口格式由下方各能力协议决定。
            </p>
          </div>

          {/* 动态平台字段 */}
          <div className="rounded-2xl border border-border/40 bg-muted/15 p-3">
            <div className="mb-2 flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-medium">常用提供商</p>
                <p className="mt-0.5 text-[10px] text-muted-foreground">快速填入连接、地址和各能力默认协议。</p>
              </div>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {API_PROVIDER_PRESETS.map(provider => {
                const normalizedCurrentUrl = (form.apiUrl || "").replace(/\/+$/, "");
                const selected = form.platform === provider.platform && normalizedCurrentUrl === provider.url;
                return (
                  <button
                    key={provider.id}
                    type="button"
                    onClick={() => {
                      updateField("platform", provider.platform);
                      updateField("apiUrl", provider.url);
                      updateField("name", form.name || provider.label);
                      updateField("textProtocol", provider.textProtocol);
                      updateField("imageProtocol", provider.imageProtocol);
                      updateField("videoProtocol", provider.videoProtocol);
                      if (provider.platform === "openai_compatible" || provider.platform === "deepseek") {
                        updateField("autoAppendV1Path", true);
                      }
                    }}
                    className={cn(
                      "rounded-xl border px-3 py-2 text-left transition-colors",
                      selected
                        ? "border-primary/50 bg-primary/8"
                        : "border-border/40 hover:border-primary/35 hover:bg-background/50"
                    )}
                  >
                    <span className="text-xs font-medium">{provider.label}</span>
                    <span className="mt-0.5 block truncate text-[10px] text-muted-foreground">{provider.url}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="rounded-2xl border border-border/40 bg-muted/15 p-3 space-y-3">
            <div>
              <p className="text-xs font-medium">能力默认协议</p>
              <p className="mt-0.5 text-[10px] leading-4 text-muted-foreground">
                同一站点可以按能力使用不同请求格式。例如 Agnes：文本使用 OpenAI 兼容协议，图片和视频使用 Agnes 协议。
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              {([
                { key: "textProtocol", label: "文本默认", modelType: 1 },
                { key: "imageProtocol", label: "图片默认", modelType: 2 },
                { key: "videoProtocol", label: "视频默认", modelType: 3 },
              ] as const).map(item => {
                const options = getProtocolOptionsForModelType(item.modelType);
                return (
                  <div key={item.key} className="space-y-1.5">
                    <Label className="text-[11px] text-muted-foreground">{item.label}</Label>
                    <Select
                      value={form[item.key] || UNSET_PROTOCOL_VALUE}
                      onValueChange={value => updateField(item.key, value === UNSET_PROTOCOL_VALUE ? "" : String(value))}
                      items={[
                        { value: UNSET_PROTOCOL_VALUE, label: "未配置" },
                        ...options.map(option => ({ value: option.value, label: option.label })),
                      ]}
                    >
                      <SelectTrigger className="w-full text-xs">
                        <SelectValue placeholder="未配置" />
                      </SelectTrigger>
                      <SelectContent className="text-xs">
                        <SelectGroup>
                          <SelectItem value={UNSET_PROTOCOL_VALUE} className="text-xs">未配置</SelectItem>
                          {options.map(option => (
                            <SelectItem key={option.value} value={option.value} className="text-xs">
                              {option.label}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>
                );
              })}
            </div>
          </div>

          {fields.map(field => (
            <div key={field.key} className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">
                {field.label}
                {field.required && <span className="text-destructive ml-0.5">*</span>}
              </Label>
              <div className="relative">
                {field.multiline ? (
                  <Textarea
                    placeholder={field.placeholder}
                    value={(form as unknown as Record<string, string>)[field.key] || ""}
                    onChange={e => updateField(field.key as keyof ApiConfigSaveReq, e.target.value)}
                    spellCheck={false}
                    className="min-h-28 max-h-60 overflow-auto resize-none font-mono text-xs leading-relaxed break-all whitespace-pre-wrap"
                  />
                ) : (
                  <Input
                    type={field.type === "password" && !showSecrets[field.key] ? "password" : "text"}
                    placeholder={field.placeholder}
                    value={(form as unknown as Record<string, string>)[field.key] || ""}
                    onChange={e => updateField(field.key as keyof ApiConfigSaveReq, e.target.value)}
                    className="text-sm pr-9"
                  />
                )}
                {field.type === "password" && (
                  <button
                    type="button"
                    onClick={() => setShowSecrets(prev => ({ ...prev, [field.key]: !prev[field.key] }))}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showSecrets[field.key] ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  </button>
                )}
              </div>
              {field.helperText && (
                <p className="text-[10px] text-muted-foreground/70">{field.helperText}</p>
              )}
            </div>
          ))}

          {form.platform === "openai_compatible" && (
            <div className="rounded-lg border border-border/40 bg-muted/20 px-3 py-2.5">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => updateField("autoAppendV1Path", !form.autoAppendV1Path)}
                  className={cn(
                    "relative w-9 h-5 rounded-full transition-colors duration-200",
                    form.autoAppendV1Path ? "bg-primary" : "bg-muted-foreground/30"
                  )}
                >
                  <span
                    className={cn(
                      "absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200",
                      form.autoAppendV1Path && "translate-x-4"
                    )}
                  />
                </button>
                <div className="min-w-0">
                  <Label
                    className="text-xs text-muted-foreground cursor-pointer"
                    onClick={() => updateField("autoAppendV1Path", !form.autoAppendV1Path)}
                  >
                    自动补充 /v1 路径
                  </Label>
                  <p className="text-[10px] text-muted-foreground/70 mt-1">
                    开启后系统会在服务根地址后补充 /v1，再根据模型协议拼接 images、videos、chat 等接口。
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="rounded-lg border border-border/40 bg-muted/20 px-3 py-2.5 space-y-3">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">出站代理</Label>
              <Select
                value={form.proxyType || "none"}
                onValueChange={v => {
                  updateField("proxyType", v as string);
                  if (v === "none") {
                    updateField("proxyHost", "");
                    updateField("proxyPort", undefined);
                    updateField("proxyUsername", "");
                    updateField("proxyPassword", "");
                  }
                }}
                items={PROXY_TYPE_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
              >
                <SelectTrigger className="w-full text-sm">
                  <SelectValue placeholder="选择代理类型" />
                </SelectTrigger>
                <SelectContent className="text-sm">
                  <SelectGroup>
                    {PROXY_TYPE_OPTIONS.map(opt => (
                      <SelectItem key={opt.value} value={opt.value} className="text-sm">
                        <div>
                          <div>{opt.label}</div>
                          <div className="text-[10px] text-muted-foreground">{opt.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
              <p className="text-[10px] text-muted-foreground/70">
                仅影响后端访问当前 API 配置绑定的模型服务，适合 Vertex AI、OpenAI、Anthropic 等国外服务。支持可选代理账号密码。
              </p>
            </div>

            {form.proxyType && form.proxyType !== "none" && (
              <div className="space-y-3">
                <div className="grid grid-cols-[1fr_110px] gap-2">
                  <div className="space-y-1.5">
                    <Label className="text-xs text-muted-foreground">代理主机</Label>
                    <Input
                      placeholder="127.0.0.1"
                      value={form.proxyHost || ""}
                      onChange={e => updateField("proxyHost", e.target.value)}
                      className="text-sm font-mono"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs text-muted-foreground">端口</Label>
                    <Input
                      type="number"
                      min={1}
                      max={65535}
                      placeholder="7890"
                      value={form.proxyPort ?? ""}
                      onChange={e => updateField("proxyPort", e.target.value ? Number(e.target.value) : undefined)}
                      className="text-sm font-mono"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1.5">
                    <Label className="text-xs text-muted-foreground">代理用户名</Label>
                    <Input
                      placeholder="可选"
                      value={form.proxyUsername || ""}
                      onChange={e => updateField("proxyUsername", e.target.value)}
                      className="text-sm"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs text-muted-foreground">代理密码</Label>
                    <div className="relative">
                      <Input
                        type={showSecrets.proxyPassword ? "text" : "password"}
                        placeholder="可选"
                        value={form.proxyPassword || ""}
                        onChange={e => updateField("proxyPassword", e.target.value)}
                        className="text-sm pr-9"
                      />
                      <button
                        type="button"
                        onClick={() => setShowSecrets(prev => ({ ...prev, proxyPassword: !prev.proxyPassword }))}
                        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {showSecrets.proxyPassword ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* 备注 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">备注</Label>
            <Input
              placeholder="可选备注信息"
              value={form.remark || ""}
              onChange={e => updateField("remark", e.target.value)}
              className="text-sm"
            />
          </div>
        </div>

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button size="sm" onClick={handleSave} disabled={saving || !form.name.trim() || proxyInvalid}>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            {editingConfig ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
