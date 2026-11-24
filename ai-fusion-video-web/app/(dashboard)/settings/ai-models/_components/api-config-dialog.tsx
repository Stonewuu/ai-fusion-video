"use client";

import { useEffect, useState } from "react";
import { ChevronRight, Eye, EyeOff, Loader2, PlugZap } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useUnsavedDialog } from "@/components/settings/use-unsaved-dialog";
import {
  apiConfigApi,
  PLATFORM_OPTIONS,
  type ApiConfig,
  type ApiConfigSaveReq,
} from "@/lib/api/ai-model";
import { comfyUiWorkflowApi } from "@/lib/api/comfyui-workflow";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";
import {
  apiConfigIssue,
  type ConfigIssue,
} from "@/lib/settings/form-validation";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogClose,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { ProviderVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { getPlatformFields } from "../../_shared";
import { getProtocolOptionsForModelType } from "./model-config-support";
import { ProviderPresetPicker } from "./provider-preset-picker";
import { ApiProxySettings } from "./api-proxy-settings";

export interface ApiConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingConfig: ApiConfig | null;
  providers: ProviderDefinition[];
  providersLoading: boolean;
  providersError: string | null;
  onReloadProviders: () => void;
  onQuickSetup: (
    provider: ProviderDefinition,
    configuration: ApiConfigSaveReq,
  ) => void;
  initialValues?: ApiConfigSaveReq;
  onSaved: () => Promise<void>;
  onCreateModel: (configId: number) => void;
}

export const UNSET_PROTOCOL_VALUE = "__unset_protocol__";

export function ApiConfigDialog({
  open,
  onOpenChange,
  editingConfig,
  providers,
  providersLoading,
  providersError,
  onReloadProviders,
  onQuickSetup,
  initialValues,
  onSaved,
  onCreateModel,
}: ApiConfigDialogProps) {
  const [saving, setSaving] = useState(false);
  const [issue, setIssue] = useState<ConfigIssue | null>(null);
  const [noKey, setNoKey] = useState(false);
  const [originalNoKey, setOriginalNoKey] = useState(false);
  const [protocolsOpen, setProtocolsOpen] = useState(false);
  const [proxyOpen, setProxyOpen] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [form, setForm] = useState<ApiConfigSaveReq>({ name: "" });
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [original, setOriginal] = useState<ApiConfigSaveReq | null>(null);

  const selectedProvider = providers.find(
    (provider) => provider.id === form.providerId,
  );
  const editingRequiresKey =
    providers.find((provider) => provider.id === editingConfig?.providerId)
      ?.quickSetup === true;

  const fields = getPlatformFields(form.platform || "").map((field) =>
    form.videoProtocol === "dashscope_wan3" && field.key === "apiUrl"
      ? {
          ...field,
          required: true,
          placeholder: "https://业务空间ID.cn-beijing.maas.aliyuncs.com",
          helperText: "填写业务空间根地址，无需追加 /api/v1。",
        }
      : field,
  );

  const changeOpen = useUnsavedDialog(
    open,
    !!original
      && (JSON.stringify(form) !== JSON.stringify(original) || noKey !== originalNoKey),
    saving,
    onOpenChange,
  );

  useEffect(() => {
    if (!open) {
      return;
    }
    const initializeForm = (value: ApiConfigSaveReq) => {
      setForm(value);
      setOriginal(value);
    };
    if (editingConfig) {
      initializeForm({
        id: editingConfig.id,
        name: editingConfig.name,
        providerId: editingConfig.providerId ?? "",
        platform:
          editingConfig.platform === "deepseek"
            ? "openai_compatible"
            : editingConfig.platform || "",
        textProtocol:
          editingConfig.textProtocol === "deepseek"
            ? "openai_compatible"
            : editingConfig.textProtocol || "",
        imageProtocol: editingConfig.imageProtocol || "",
        videoProtocol: editingConfig.videoProtocol || "",
        apiUrl:
          editingConfig.apiUrl
          || (editingConfig.platform === "vertex_ai" ? "us-central1" : ""),
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
      setProxyOpen(Boolean(editingConfig.proxyType && editingConfig.proxyType !== "none"));
    } else {
      initializeForm({
        name: "",
        platform: "openai_compatible",
        textProtocol: "openai_compatible",
        imageProtocol: "openai",
        videoProtocol: "openai",
        apiUrl: "",
        autoAppendV1Path: true,
        proxyType: "none",
        proxyHost: "",
        proxyPort: undefined,
        proxyUsername: "",
        proxyPassword: "",
        apiKey: "",
        appId: "",
        appSecret: "",
        status: 1,
        ...initialValues,
      });
      setProxyOpen(Boolean(initialValues?.proxyType && initialValues.proxyType !== "none"));
    }
    setProtocolsOpen(false);
    setShowSecrets({});
    setIssue(null);
    const keylessInitial =
      !!editingConfig
      && !editingRequiresKey
      && editingConfig.platform === "openai_compatible"
      && !editingConfig.apiKey;
    setNoKey(keylessInitial);
    setOriginalNoKey(keylessInitial);
    setTestingConnection(false);
  }, [open, editingConfig, initialValues, editingRequiresKey]);

  const updateField = <K extends keyof ApiConfigSaveReq>(
    key: K,
    value: ApiConfigSaveReq[K],
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    setIssue(null);
  };

  const handleSave = async () => {
    const invalid = apiConfigIssue(form, fields, noKey);
    if (invalid) {
      setIssue(invalid);
      if (invalid.field.startsWith("api-proxy-")) {
        setProxyOpen(true);
      }
      requestAnimationFrame(() => {
        document.getElementById(invalid.field)?.focus();
      });
      return;
    }
    setIssue(null);
    setSaving(true);
    try {
      const proxyEnabled = Boolean(form.proxyType && form.proxyType !== "none");
      const payload: ApiConfigSaveReq = {
        ...form,
        apiKey:
          noKey && form.platform === "openai_compatible" ? "" : form.apiKey,
        proxyType: proxyEnabled ? form.proxyType : "none",
        proxyHost: proxyEnabled ? form.proxyHost?.trim() : "",
        proxyPort: proxyEnabled ? form.proxyPort : undefined,
        proxyUsername: proxyEnabled ? form.proxyUsername?.trim() : "",
        proxyPassword:
          proxyEnabled && form.proxyUsername?.trim() ? form.proxyPassword : "",
      };
      const configId = editingConfig
        ? (await apiConfigApi.update(payload), editingConfig.id)
        : await apiConfigApi.create(payload);
      await onSaved();
      onOpenChange(false);
      const createModelNext =
        !editingConfig && !["comfyui", "sonilo"].includes(form.platform || "");
      if (createModelNext) {
        onCreateModel(configId);
      }
      toast.success(editingConfig ? "API 配置已保存" : "API 服务已创建", {
        description: editingConfig
          ? undefined
          : form.platform === "comfyui"
            ? "接下来管理工作流并添加模型。"
            : form.platform === "sonilo"
              ? "音频服务已可使用。"
              : "服务已保存，继续填写模型信息即可使用。",
      });
    } catch (err) {
      setIssue({
        field: "",
        message: err instanceof Error ? err.message : "保存 API 配置失败",
      });
    } finally {
      setSaving(false);
    }
  };

  const selectProvider = (provider: ProviderDefinition) => {
    const replaceName =
      !form.name
      || (!editingConfig && providers.some((item) => item.label === form.name));
    const configuration: ApiConfigSaveReq =
      form.providerId === provider.id
        ? form
        : {
            ...form,
            name: replaceName ? provider.label : form.name,
            providerId: provider.id,
            platform: provider.platform,
            apiUrl: provider.url,
            textProtocol: provider.textProtocol,
            imageProtocol: provider.imageProtocol,
            videoProtocol: provider.videoProtocol,
            autoAppendV1Path: provider.appendV1,
          };
    setNoKey(false);
    setIssue(null);
    if (!editingConfig && provider.quickSetup) {
      onOpenChange(false);
      onQuickSetup(provider, configuration);
      return;
    }
    setForm(configuration);
  };

  const handleTestConnection = async () => {
    if (!editingConfig) {
      return;
    }
    setTestingConnection(true);
    try {
      const result = await comfyUiWorkflowApi.testConnection(editingConfig.id);
      toast.success("ComfyUI 连接成功", {
        description: `版本 ${result.version || "未知"}，Jobs API ${result.jobsApiSupported ? "可用" : "不可用"}`,
      });
    } catch (err) {
      console.error("检测 ComfyUI 连接失败:", err);
      toastApiError(err, "检测 ComfyUI 连接失败");
    } finally {
      setTestingConnection(false);
    }
  };

  const protocolsLocked =
    form.platform === "comfyui" || form.platform === "sonilo";

  const protocolFields = (
    [
      { key: "textProtocol", label: "文本默认", modelType: 1 },
      { key: "imageProtocol", label: "图片默认", modelType: 2 },
      { key: "videoProtocol", label: "视频默认", modelType: 3 },
    ] as const
  ).filter(
    (item) =>
      !selectedProvider
      || selectedProvider.supportedModelTypes.includes(item.modelType),
  );
  const configuredProtocolCount = protocolFields.filter((item) =>
    Boolean(form[item.key]),
  ).length;

  const presetPicker = (
    <ProviderPresetPicker
      providers={providers}
      selectedId={form.providerId}
      loading={providersLoading}
      error={providersError}
      disabled={saving}
      allowQuickSetup={!editingConfig}
      onReload={onReloadProviders}
      onSelect={selectProvider}
    />
  );

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] w-[calc(100%-2rem)] flex-col overflow-hidden sm:max-w-3xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            {editingConfig ? "编辑 API 配置" : "新建 API 配置"}
          </DialogTitle>
          <DialogDescription>
            {editingConfig
              ? "更新服务连接地址、密钥、默认协议与代理设置。"
              : "先选择常用提供商自动填充参数，再核对地址与密钥即可完成创建。"}
          </DialogDescription>
        </DialogHeader>

        <DialogScrollBody className="p-2">
          <fieldset disabled={saving} className="flex min-w-0 flex-col gap-4">
            {/* 新建时将常用提供商前置，贴合“先选平台再填密钥”的自然动线 */}
            {!editingConfig && presetPicker}

            {/* 基础信息：配置名称 + 接入类型 */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="flex min-w-0 flex-col gap-2">
                <Label htmlFor="api-config-name" className="text-xs text-muted-foreground">
                  配置名称 <span className="text-destructive">*</span>
                </Label>
                <Input
                  placeholder="例如：DeepSeek / Gemini"
                  id="api-config-name"
                  aria-invalid={issue?.field === "api-config-name"}
                  value={form.name}
                  onChange={(event) => updateField("name", event.target.value)}
                  className="text-sm"
                />
              </div>

              <div className="flex min-w-0 flex-col gap-2">
                <Label className="text-xs text-muted-foreground">
                  接入与鉴权类型
                </Label>
                <Select
                  value={form.platform || "openai_compatible"}
                  onValueChange={(value) => {
                    const platform = String(value);
                    setNoKey(false);
                    updateField("platform", platform);
                    updateField("providerId", "");
                    setForm((previous) => ({
                      ...previous,
                      textProtocol: getProtocolOptionsForModelType(1, null, providers).some(
                        (option) => option.value === previous.textProtocol,
                      )
                        ? previous.textProtocol
                        : "",
                      imageProtocol: getProtocolOptionsForModelType(2, null, providers).some(
                        (option) => option.value === previous.imageProtocol,
                      )
                        ? previous.imageProtocol
                        : "",
                      videoProtocol: getProtocolOptionsForModelType(3, null, providers).some(
                        (option) => option.value === previous.videoProtocol,
                      )
                        ? previous.videoProtocol
                        : "",
                    }));
                    if (platform === "vertex_ai" || platform === "vertex_ai_api_key") {
                      setForm((previous) => ({
                        ...previous,
                        textProtocol: "vertex_ai",
                        imageProtocol: "vertex_ai",
                        videoProtocol: "vertex_ai",
                        apiUrl: form.platform?.startsWith("vertex_ai")
                          ? previous.apiUrl || "global"
                          : "global",
                        autoAppendV1Path: false,
                      }));
                    }
                    if (platform === "gemini") {
                      setForm((previous) => ({
                        ...previous,
                        textProtocol: "gemini",
                        imageProtocol: "gemini",
                        videoProtocol: "gemini",
                        apiUrl: "",
                        autoAppendV1Path: false,
                      }));
                    }
                    if (platform === "openai_compatible") {
                      updateField("autoAppendV1Path", true);
                    }
                    if (platform === "comfyui") {
                      setForm((previous) => ({
                        ...previous,
                        platform,
                        textProtocol: "",
                        imageProtocol: "comfyui",
                        videoProtocol: "comfyui",
                        apiUrl: "http://localhost:8188",
                        autoAppendV1Path: false,
                      }));
                    }
                    if (platform === "sonilo") {
                      setForm((previous) => ({
                        ...previous,
                        platform,
                        textProtocol: "",
                        imageProtocol: "",
                        videoProtocol: "",
                        apiUrl: "",
                        autoAppendV1Path: false,
                      }));
                    }
                  }}
                  items={PLATFORM_OPTIONS.map((option) => ({
                    value: option.value,
                    label: option.label,
                  }))}
                >
                  <SelectTrigger aria-label="接入与鉴权类型" className="w-full text-sm">
                    <SelectValue placeholder="选择接入类型" />
                  </SelectTrigger>
                  <SelectContent className="text-sm">
                    <SelectGroup>
                      {PLATFORM_OPTIONS.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value} className="text-sm">
                          <ProviderVendorIcon provider={opt.value} className="size-4 self-center" />
                          <div className="min-w-0">
                            <div>{opt.label}</div>
                            <div className="text-xs text-muted-foreground">
                              {opt.description}
                            </div>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {form.videoProtocol === "dashscope_wan3" && (
              <p className="rounded-lg border border-border/20 bg-background/70 p-3 text-xs leading-relaxed text-muted-foreground">
                万相 3.0 需填写所属业务空间的地域地址，例如
                https://业务空间ID.cn-beijing.maas.aliyuncs.com。模型、地址和 API Key 必须属于同一地域。{" "}
                <a
                  className="text-primary underline underline-offset-2"
                  href="https://www.alibabacloud.com/help/zh/model-studio/wan3-video-generation-api-reference"
                  target="_blank"
                  rel="noreferrer"
                >
                  查看官方地址说明
                </a>
              </p>
            )}

            {form.platform === "openai_compatible" && !selectedProvider?.quickSetup && (
              <label className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
                <Checkbox
                  checked={noKey}
                  disabled={saving}
                  onCheckedChange={(checked) => {
                    setNoKey(checked);
                    setIssue(null);
                  }}
                />
                此服务无需 API 密钥（如本地模型服务）
              </label>
            )}

            {fields.map((field) => (
              <div key={field.key} className="flex min-w-0 flex-col gap-2">
                <Label
                  htmlFor={`api-field-${field.key}`}
                  className="text-xs text-muted-foreground"
                >
                  {field.label}
                  {field.required
                    && !(
                      field.key === "apiKey"
                      && noKey
                      && form.platform === "openai_compatible"
                    ) && <span className="ml-0.5 text-destructive">*</span>}
                </Label>
                <div className="relative">
                  {field.multiline ? (
                    <Textarea
                      id={`api-field-${field.key}`}
                      aria-invalid={issue?.field === `api-field-${field.key}`}
                      aria-describedby={
                        issue?.field === `api-field-${field.key}`
                          ? "api-form-error"
                          : undefined
                      }
                      placeholder={field.placeholder}
                      value={(form as unknown as Record<string, string>)[field.key] || ""}
                      onChange={(event) =>
                        updateField(
                          field.key as keyof ApiConfigSaveReq,
                          event.target.value,
                        )
                      }
                      spellCheck={false}
                      className="max-h-60 min-h-28 resize-none overflow-auto break-all whitespace-pre-wrap font-mono text-xs leading-relaxed"
                    />
                  ) : (
                    <Input
                      disabled={
                        saving
                        || (field.key === "apiKey"
                          && noKey
                          && form.platform === "openai_compatible")
                      }
                      type={
                        field.type === "password" && !showSecrets[field.key]
                          ? "password"
                          : "text"
                      }
                      id={`api-field-${field.key}`}
                      aria-invalid={issue?.field === `api-field-${field.key}`}
                      aria-describedby={
                        issue?.field === `api-field-${field.key}`
                          ? "api-form-error"
                          : undefined
                      }
                      placeholder={field.placeholder}
                      value={(form as unknown as Record<string, string>)[field.key] || ""}
                      onChange={(event) =>
                        updateField(
                          field.key as keyof ApiConfigSaveReq,
                          event.target.value,
                        )
                      }
                      className={cn("text-sm", field.type === "password" && "pr-9")}
                    />
                  )}
                  {field.type === "password" && (
                    <button
                      type="button"
                      aria-label={`${showSecrets[field.key] ? "隐藏" : "显示"}${field.label}`}
                      aria-pressed={!!showSecrets[field.key]}
                      onClick={() =>
                        setShowSecrets((prev) => ({
                          ...prev,
                          [field.key]: !prev[field.key],
                        }))
                      }
                      className="absolute right-2.5 top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      {showSecrets[field.key] ? (
                        <EyeOff className="size-3.5" />
                      ) : (
                        <Eye className="size-3.5" />
                      )}
                    </button>
                  )}
                </div>
                {field.helperText && (
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {field.helperText}
                  </p>
                )}
              </div>
            ))}

            {/* 启用此配置 */}
            <div className="flex items-center justify-between gap-4 rounded-xl border border-border/30 bg-background/70 px-3.5 py-3">
              <div className="flex min-w-0 flex-col gap-1">
                <Label htmlFor="api-config-enabled" className="cursor-pointer text-sm font-medium">
                  启用此配置
                </Label>
                <p className="text-xs text-muted-foreground">
                  {editingRequiresKey && !editingConfig?.apiKey
                    ? "填写密钥并启用后，即可使用已导入的模型。"
                    : "停用后保留所有模型设置，仅暂停调用此连接。"}
                </p>
              </div>
              <Switch
                id="api-config-enabled"
                checked={form.status === 1}
                disabled={saving}
                onCheckedChange={(enabled) => updateField("status", enabled ? 1 : 0)}
              />
            </div>

            {form.platform === "openai_compatible" && (
              <div className="flex items-center justify-between gap-4 rounded-xl border border-border/30 bg-background/70 px-3.5 py-3">
                <div className="flex min-w-0 flex-col gap-1">
                  <Label htmlFor="api-auto-v1" className="cursor-pointer text-sm font-medium">
                    自动补充 /v1 路径
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    开启后自动在根地址后追加 /v1，再按模型协议拼接对应接口路径。
                  </p>
                </div>
                <Switch
                  id="api-auto-v1"
                  checked={!!form.autoAppendV1Path}
                  disabled={saving}
                  onCheckedChange={(checked) => updateField("autoAppendV1Path", checked)}
                />
              </div>
            )}

            {/* 高级协议设置 */}
            {!protocolsLocked && (
              <Collapsible
                open={protocolsOpen}
                onOpenChange={setProtocolsOpen}
                className="rounded-xl border border-border/30 bg-background/70"
              >
                <CollapsibleTrigger className="flex w-full items-center justify-between gap-3 rounded-xl px-3.5 py-2.5 text-left outline-none transition-colors hover:bg-muted/35 focus-visible:ring-2 focus-visible:ring-ring">
                  <span className="flex min-w-0 items-center gap-2">
                    <span className="text-sm font-medium text-foreground">
                      高级协议设置
                    </span>
                    <Badge variant="outline" className="text-xs">
                      已配 {configuredProtocolCount}/{protocolFields.length} 项默认协议
                    </Badge>
                  </span>
                  <ChevronRight
                    className={cn(
                      "size-4 shrink-0 text-muted-foreground transition-transform duration-200 motion-reduce:transition-none",
                      protocolsOpen && "rotate-90",
                    )}
                  />
                </CollapsibleTrigger>

                <CollapsibleContent>
                  <div className="space-y-3 border-t border-border/20 p-3.5">
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      同一服务可按能力使用不同请求协议，模型未单独覆盖时将自动继承此处设置。
                    </p>
                    <div className="grid gap-3 sm:grid-cols-3">
                      {protocolFields.map((item) => {
                        const options = getProtocolOptionsForModelType(
                          item.modelType,
                          form.providerId,
                          providers,
                        );
                        return (
                          <div key={item.key} className="flex min-w-0 flex-col gap-2">
                            <Label className="text-xs text-muted-foreground">
                              {item.label}
                            </Label>
                            <Select
                              value={form[item.key] || UNSET_PROTOCOL_VALUE}
                              onValueChange={(value) =>
                                updateField(
                                  item.key,
                                  value === UNSET_PROTOCOL_VALUE ? "" : String(value),
                                )
                              }
                              disabled={providersLoading || !!providersError}
                              items={[
                                { value: UNSET_PROTOCOL_VALUE, label: "未配置" },
                                ...options.map((option) => ({
                                  value: option.value,
                                  label: option.label,
                                })),
                              ]}
                            >
                              <SelectTrigger aria-label={item.label} className="w-full text-xs">
                                <SelectValue placeholder="未配置" />
                              </SelectTrigger>
                              <SelectContent className="text-xs">
                                <SelectGroup>
                                  <SelectItem value={UNSET_PROTOCOL_VALUE} className="text-xs">
                                    未配置
                                  </SelectItem>
                                  {options.map((option) => (
                                    <SelectItem
                                      key={option.value}
                                      value={option.value}
                                      className="text-xs"
                                    >
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
                </CollapsibleContent>
              </Collapsible>
            )}

            <ApiProxySettings
              form={form}
              issue={issue}
              open={proxyOpen}
              onOpenChange={setProxyOpen}
              updateField={updateField}
              showSecrets={showSecrets}
              setShowSecrets={setShowSecrets}
            />

            {/* 编辑模式仍保留快捷切换提供商入口，但不抢占主表单动线 */}
            {editingConfig && presetPicker}

            {/* 备注 */}
            <div className="flex min-w-0 flex-col gap-2">
              <Label htmlFor="api-remark" className="text-xs text-muted-foreground">
                备注（可选）
              </Label>
              <Input
                id="api-remark"
                placeholder="例如：账号用途或计费说明"
                value={form.remark || ""}
                onChange={(event) => updateField("remark", event.target.value)}
                className="text-sm"
              />
            </div>
          </fieldset>
        </DialogScrollBody>

        {issue && (
          <p id="api-form-error" role="alert" className="shrink-0 text-sm text-destructive">
            {issue.message}
          </p>
        )}

        <DialogFooter className="shrink-0">
          {editingConfig?.platform === "comfyui" && (
            <Button
              type="button"
              variant="outline"
              onClick={handleTestConnection}
              disabled={testingConnection || saving}
            >
              {testingConnection ? <Loader2 className="animate-spin" /> : <PlugZap />}
              测试连接
            </Button>
          )}
          <DialogClose render={<Button variant="outline" />}>取消</DialogClose>
          <Button type="button" onClick={handleSave} disabled={saving}>
            {saving && <Loader2 className="animate-spin" />}
            {editingConfig ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
