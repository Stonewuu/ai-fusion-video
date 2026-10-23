"use client";

import { useEffect, useRef, useState } from "react";
import { useUnsavedDialog } from "@/components/settings/use-unsaved-dialog";
import { Eye, EyeOff, Loader2, PlugZap } from "lucide-react";
import { toast } from "sonner";
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
import { Checkbox } from "@/components/ui/checkbox";
import {
  apiConfigIssue,
  type ConfigIssue,
} from "@/lib/settings/form-validation";
import { Textarea } from "@/components/ui/textarea";
import { ProviderVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { getPlatformFields } from "../../_shared";
import { getProtocolOptionsForModelType } from "./model-config-support";
import { comfyUiWorkflowApi } from "@/lib/api/comfyui-workflow";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";
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
  const proxyDetails = useRef<HTMLDetailsElement>(null);
  const [testingConnection, setTestingConnection] = useState(false);
  const [form, setForm] = useState<ApiConfigSaveReq>({ name: "" });
  const selectedProvider = providers.find(
    (provider) => provider.id === form.providerId,
  );
  const editingRequiresKey =
    providers.find((provider) => provider.id === editingConfig?.providerId)
      ?.quickSetup === true;
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
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

  const [original, setOriginal] = useState<ApiConfigSaveReq | null>(null);
  const changeOpen = useUnsavedDialog(
    open,
    !!original &&
      (JSON.stringify(form) !== JSON.stringify(original) ||
        noKey !== originalNoKey),
    saving,
    onOpenChange,
  );

  useEffect(() => {
    if (open) {
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
      }
      setShowSecrets({});
      setIssue(null);
      setNoKey(
        !!editingConfig &&
          !editingRequiresKey &&
          editingConfig.platform === "openai_compatible" &&
          !editingConfig.apiKey,
      );
      setOriginalNoKey(
        !!editingConfig &&
          !editingRequiresKey &&
          editingConfig.platform === "openai_compatible" &&
          !editingConfig.apiKey,
      );
      setTestingConnection(false);
    }
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
      if (invalid.field.startsWith("api-proxy-") && proxyDetails.current) {
        proxyDetails.current.open = true;
      }
      requestAnimationFrame(() =>
        document.getElementById(invalid.field)?.focus(),
      );
      return;
    }
    setIssue(null);
    setSaving(true);
    try {
      const proxyEnabled = form.proxyType && form.proxyType !== "none";
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
      !form.name ||
      (!editingConfig && providers.some((item) => item.label === form.name));
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

  // ComfyUI / Sonilo 不走文本、图片、视频协议路由，协议选择器整体锁定
  const protocolsLocked =
    form.platform === "comfyui" || form.platform === "sonilo";

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogContent className="w-[calc(100%-2rem)] sm:max-w-3xl max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            {editingConfig ? "编辑 API 配置" : "新建 API 配置"}
          </DialogTitle>
          <DialogDescription>
            选择提供商，填写地址与密钥。创建后添加模型，即可在助手和画布中使用。
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 overflow-y-auto p-2">
          <fieldset disabled={saving} className="min-w-0 space-y-4">
            {/* 配置名称 */}
            <div className="flex min-w-0 flex-col gap-2">
              <Label
                htmlFor="api-config-name"
                className="text-xs text-muted-foreground"
              >
                配置名称
              </Label>
              <Input
                placeholder="例如：DeepSeek / Gemini"
                id="api-config-name"
                aria-invalid={issue?.field === "api-config-name"}
                value={form.name}
                onChange={(e) => updateField("name", e.target.value)}
                className="text-sm"
              />
            </div>

            {/* 接入与鉴权类型 */}
            <div className="flex min-w-0 flex-col gap-2">
              <Label className="text-xs text-muted-foreground">
                接入与鉴权类型
              </Label>
              <Select
                value={form.platform || "openai_compatible"}
                onValueChange={(v) => {
                  const platform = String(v);
                  setNoKey(false);
                  updateField("platform", platform);
                  updateField("providerId", "");
                  if (v === "openai_compatible") {
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
                items={PLATFORM_OPTIONS.map((o) => ({
                  value: o.value,
                  label: o.label,
                }))}
              >
                <SelectTrigger
                  aria-label="接入与鉴权类型"
                  className="w-full text-sm"
                >
                  <SelectValue placeholder="选择接入类型" />
                </SelectTrigger>
                <SelectContent className="text-sm">
                  <SelectGroup>
                    {PLATFORM_OPTIONS.map((opt) => (
                      <SelectItem
                        key={opt.value}
                        value={opt.value}
                        className="text-sm"
                      >
                        <ProviderVendorIcon
                          provider={opt.value}
                          className="size-4 self-center"
                        />
                        <div className="min-w-0">
                          <div>{opt.label}</div>
                          <div className="text-[10px] text-muted-foreground">
                            {opt.description}
                          </div>
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

            {form.videoProtocol === "dashscope_wan3" && (
              <p className="text-xs leading-relaxed text-muted-foreground">
                万相 3.0 需填写所属业务空间的地域地址，例如
                https://业务空间ID.cn-beijing.maas.aliyuncs.com。模型、地址和
                API Key 必须属于同一地域。
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
            {form.platform === "openai_compatible" &&
              !selectedProvider?.quickSetup && (
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={noKey}
                    disabled={saving}
                    onCheckedChange={(checked) => {
                      setNoKey(checked);
                      setIssue(null);
                    }}
                  />
                  此服务无需 API 密钥（如本地模型）
                </label>
              )}
            {fields.map((field) => (
              <div key={field.key} className="flex min-w-0 flex-col gap-2">
                <Label
                  htmlFor={`api-field-${field.key}`}
                  className="text-xs text-muted-foreground"
                >
                  {field.label}
                  {field.required &&
                    !(
                      field.key === "apiKey" &&
                      noKey &&
                      form.platform === "openai_compatible"
                    ) && <span className="text-destructive ml-0.5">*</span>}
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
                      value={
                        (form as unknown as Record<string, string>)[
                          field.key
                        ] || ""
                      }
                      onChange={(e) =>
                        updateField(
                          field.key as keyof ApiConfigSaveReq,
                          e.target.value,
                        )
                      }
                      spellCheck={false}
                      className="min-h-28 max-h-60 overflow-auto resize-none font-mono text-xs leading-relaxed break-all whitespace-pre-wrap"
                    />
                  ) : (
                    <Input
                      disabled={
                        saving ||
                        (field.key === "apiKey" &&
                          noKey &&
                          form.platform === "openai_compatible")
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
                      value={
                        (form as unknown as Record<string, string>)[
                          field.key
                        ] || ""
                      }
                      onChange={(e) =>
                        updateField(
                          field.key as keyof ApiConfigSaveReq,
                          e.target.value,
                        )
                      }
                      className="text-sm pr-9"
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
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {showSecrets[field.key] ? (
                        <EyeOff className="h-3.5 w-3.5" />
                      ) : (
                        <Eye className="h-3.5 w-3.5" />
                      )}
                    </button>
                  )}
                </div>
                {field.helperText && (
                  <p className="text-[10px] text-muted-foreground/70">
                    {field.helperText}
                  </p>
                )}
              </div>
            ))}

            <div className="flex items-start gap-3 rounded-lg border border-border/20 bg-background/70 p-3">
              <Checkbox
                id="api-config-enabled"
                checked={form.status === 1}
                disabled={saving}
                onCheckedChange={(enabled) =>
                  updateField("status", enabled ? 1 : 0)
                }
              />
              <div className="flex min-w-0 flex-col gap-1">
                <Label htmlFor="api-config-enabled">启用此配置</Label>
                <p className="text-xs text-muted-foreground">
                  {editingRequiresKey && !editingConfig?.apiKey
                    ? "填写密钥并启用后，即可使用已导入的模型。"
                    : "停用后保留模型设置，暂停使用此连接。"}
                </p>
              </div>
            </div>

            <details className="rounded-lg border border-border/20 bg-background/70 p-3">
              <summary className="cursor-pointer rounded-lg text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-ring">
                高级协议设置
              </summary>
              <div className="mt-3 space-y-3">
                <div>
                  <p className="text-xs font-medium">能力默认协议</p>
                  <p className="mt-0.5 text-[10px] leading-4 text-muted-foreground">
                    同一站点可以按能力使用不同请求格式。例如 Agnes：文本使用
                    OpenAI 兼容协议，图片和视频使用 Agnes 协议。
                  </p>
                </div>
                <div className="grid gap-3 sm:grid-cols-3">
                  {(
                    [
                      { key: "textProtocol", label: "文本默认", modelType: 1 },
                      { key: "imageProtocol", label: "图片默认", modelType: 2 },
                      { key: "videoProtocol", label: "视频默认", modelType: 3 },
                    ] as const
                  )
                    .filter(
                      (item) =>
                        !selectedProvider ||
                        selectedProvider.supportedModelTypes.includes(
                          item.modelType,
                        ),
                    )
                    .map((item) => {
                      const options = getProtocolOptionsForModelType(
                        item.modelType,
                      );
                      return (
                        <div
                          key={item.key}
                          className="flex min-w-0 flex-col gap-2"
                        >
                          <Label className="text-[11px] text-muted-foreground">
                            {item.label}
                          </Label>
                          <Select
                            value={form[item.key] || UNSET_PROTOCOL_VALUE}
                            onValueChange={(value) =>
                              updateField(
                                item.key,
                                value === UNSET_PROTOCOL_VALUE
                                  ? ""
                                  : String(value),
                              )
                            }
                            disabled={protocolsLocked}
                            items={[
                              { value: UNSET_PROTOCOL_VALUE, label: "未配置" },
                              ...options.map((option) => ({
                                value: option.value,
                                label: option.label,
                              })),
                            ]}
                          >
                            <SelectTrigger
                              aria-label={item.label}
                              className="w-full text-xs"
                            >
                              <SelectValue placeholder="未配置" />
                            </SelectTrigger>
                            <SelectContent className="text-xs">
                              <SelectGroup>
                                <SelectItem
                                  value={UNSET_PROTOCOL_VALUE}
                                  className="text-xs"
                                >
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
            </details>

            {form.platform === "openai_compatible" && (
              <div className="rounded-lg border border-border/40 bg-muted/20 px-3 py-2.5">
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    role="switch"
                    aria-label="自动补充 /v1 路径"
                    aria-checked={!!form.autoAppendV1Path}
                    onClick={() =>
                      updateField("autoAppendV1Path", !form.autoAppendV1Path)
                    }
                    className={cn(
                      "relative w-9 h-5 rounded-full transition-colors duration-200",
                      form.autoAppendV1Path
                        ? "bg-primary"
                        : "bg-muted-foreground/30",
                    )}
                  >
                    <span
                      className={cn(
                        "absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200",
                        form.autoAppendV1Path && "translate-x-4",
                      )}
                    />
                  </button>
                  <div className="min-w-0">
                    <Label
                      className="text-xs text-muted-foreground cursor-pointer"
                      onClick={() =>
                        updateField("autoAppendV1Path", !form.autoAppendV1Path)
                      }
                    >
                      自动补充 /v1 路径
                    </Label>
                    <p className="text-[10px] text-muted-foreground/70 mt-1">
                      开启后系统会在服务根地址后补充 /v1，再根据模型协议拼接
                      images、videos、chat 等接口。
                    </p>
                  </div>
                </div>
              </div>
            )}

            <ApiProxySettings
              form={form}
              issue={issue}
              detailsRef={proxyDetails}
              updateField={updateField}
              showSecrets={showSecrets}
              setShowSecrets={setShowSecrets}
            />

            {/* 备注 */}
            <div className="flex min-w-0 flex-col gap-2">
              <Label
                htmlFor="api-remark"
                className="text-xs text-muted-foreground"
              >
                备注
              </Label>
              <Input
                id="api-remark"
                placeholder="可选备注信息"
                value={form.remark || ""}
                onChange={(e) => updateField("remark", e.target.value)}
                className="text-sm"
              />
            </div>
          </fieldset>
        </div>

        {issue && (
          <p
            id="api-form-error"
            role="alert"
            className="shrink-0 text-sm text-destructive"
          >
            {issue.message}
          </p>
        )}
        <DialogFooter className="shrink-0">
          {editingConfig?.platform === "comfyui" && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleTestConnection}
              disabled={testingConnection || saving}
            >
              {testingConnection ? (
                <Loader2 className="animate-spin" />
              ) : (
                <PlugZap />
              )}
              测试连接
            </Button>
          )}
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button size="sm" onClick={handleSave} disabled={saving}>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            {editingConfig ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
