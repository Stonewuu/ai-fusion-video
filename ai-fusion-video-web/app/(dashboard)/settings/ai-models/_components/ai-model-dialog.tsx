"use client";

import { useEffect, useState } from "react";
import { Check, ChevronRight, Loader2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  aiModelApi,
  PLATFORM_LABELS,
  MODEL_TYPE_OPTIONS,
  MODEL_TYPE_LABELS,
  type AiModel,
  type ApiConfig,
  type AiModelCreateReq,
  type AiModelUpdateReq,
  type ModelPreset,
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
import { CapabilityPresetSelect } from "./capability-preset-select";
import { ModelConfigForm } from "./model-config-form";
import {
  parseConfigJson,
  mergeConfigObjects,
  stringifyConfigObject,
  formatSemanticLabel,
  OPENAI_RESPONSE_MODE_CONFIG_KEYS,
  MODEL_PROTOCOL_LABELS,
  INHERIT_PROVIDER_PROTOCOL_VALUE,
  getProtocolOptionsForModelType,
  getApiConfigDefaultProtocol,
  supportsReasoningConfig,
  supportsOpenAiResponsesMode,
  stripReasoningConfig,
  sanitizeModelConfigJson,
  getPositiveNumberValue,
  getFirstConfigString,
  isCapabilityPresetCompatible,
  ToggleSettingCard,
} from "./model-config-support";

export interface AiModelDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingModel: AiModel | null;
  apiConfigs: ApiConfig[];
  defaultApiConfigId?: number;
  onSaved: () => void;
}

type AiModelFormState = AiModelCreateReq & { id?: number; status?: number };

export function AiModelDialog({ open, onOpenChange, editingModel, apiConfigs, defaultApiConfigId, onSaved }: AiModelDialogProps) {
  const [saving, setSaving] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [form, setForm] = useState<AiModelFormState>({
    name: "",
    code: "",
    capabilityPresetCode: "",
    modelProtocol: "",
    modelType: 1,
    maxConcurrency: 5,
    supportVision: false,
    supportReasoning: false,
    contextWindow: 0,
  });
  const [presets, setPresets] = useState<ModelPreset[]>([]);

  useEffect(() => {
    if (open) {
      aiModelApi.presets().then(setPresets).catch(console.error);
      if (editingModel) {
        setForm({
          id: editingModel.id,
          name: editingModel.name,
          code: editingModel.code,
          capabilityPresetCode: editingModel.capabilityPresetCode || "",
          modelProtocol: editingModel.modelProtocol || "",
          modelType: editingModel.modelType,
          description: editingModel.description || "",
          config: editingModel.config || "",
          maxConcurrency: editingModel.maxConcurrency ?? 5,
          defaultModel: editingModel.defaultModel,
          supportVision: editingModel.supportVision,
          supportReasoning: editingModel.supportReasoning,
          contextWindow: editingModel.contextWindow ?? 0,
          apiConfigId: editingModel.apiConfigId ?? undefined,
          status: editingModel.status,
        });
      } else {
        setForm({
          name: "",
          code: "",
          capabilityPresetCode: "",
          modelProtocol: "",
          modelType: 1,
          maxConcurrency: 5,
          defaultModel: false,
          supportVision: false,
          supportReasoning: false,
          contextWindow: 0,
          apiConfigId: defaultApiConfigId,
        });
      }
      setShowAdvanced(false);
    }
  }, [open, editingModel, defaultApiConfigId]);

  const updateField = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) => {
    setForm(prev => ({ ...prev, [key]: value }));
  };

  const updateMetaNumberField = (key: "maxConcurrency" | "contextWindow", rawValue: string) => {
    if (!rawValue.trim()) {
      updateField(key, 0);
      return;
    }
    const parsed = Number.parseInt(rawValue, 10);
    updateField(key, Number.isNaN(parsed) ? 0 : parsed);
  };

  const selectedApiConfig = apiConfigs.find(c => c.id === form.apiConfigId);
  const selectedPlatform = selectedApiConfig?.platform;
  const selectedCapabilityPreset = form.capabilityPresetCode
    ? presets.find(p => p.code === form.capabilityPresetCode) || null
    : null;
  const visiblePresets = presets.filter(preset => preset.modelType === form.modelType);
  const selectedPresetConfig = selectedCapabilityPreset?.config as Record<string, unknown> | undefined;
  const effectiveModelConfig = mergeConfigObjects(selectedPresetConfig || {}, parseConfigJson(form.config));
  const showOpenAiResponsesMode = form.modelType === 1 && supportsOpenAiResponsesMode(selectedPlatform);
  const apiModeValue = getFirstConfigString(effectiveModelConfig, OPENAI_RESPONSE_MODE_CONFIG_KEYS) ?? "__unset__";
  const hasRequestProtocol = form.modelType >= 1 && form.modelType <= 3;
  const apiConfigDefaultProtocol = getApiConfigDefaultProtocol(selectedApiConfig, form.modelType);
  const protocolOptions = getProtocolOptionsForModelType(form.modelType);
  const modelProtocolOverride = form.modelProtocol?.trim() || "";
  const effectiveModelProtocol = modelProtocolOverride || apiConfigDefaultProtocol;
  const modelProtocolSelectOptions = [
    {
      value: INHERIT_PROVIDER_PROTOCOL_VALUE,
      label: apiConfigDefaultProtocol
        ? `继承 API 配置默认（${formatSemanticLabel(apiConfigDefaultProtocol, MODEL_PROTOCOL_LABELS)}）`
        : "继承 API 配置默认（当前未配置）",
    },
    ...protocolOptions,
  ];
  const protocolConfigured = !hasRequestProtocol || Boolean(effectiveModelProtocol);
  const compatibleCapabilityPresets = presets.filter(preset => isCapabilityPresetCompatible({
    preset,
    modelType: form.modelType,
    platform: selectedPlatform,
    effectiveProtocol: effectiveModelProtocol,
  }));
  const hasCustomCapabilityConfig = Boolean(form.config?.trim() && form.config.trim() !== "{}");

  const materializeEffectiveConfig = (source: AiModelFormState): string => {
    const preset = source.capabilityPresetCode
      ? presets.find(item => item.code === source.capabilityPresetCode)
      : null;
    return stringifyConfigObject(mergeConfigObjects(preset?.config || {}, parseConfigJson(source.config)));
  };

  const detachIncompatibleCapabilityPreset = (
    previous: AiModelFormState,
    next: AiModelFormState,
    nextApiConfig: ApiConfig | undefined
  ): AiModelFormState => {
    if (!previous.capabilityPresetCode) return next;
    const preset = presets.find(item => item.code === previous.capabilityPresetCode);
    if (!preset) return next;
    const nextProtocol = next.modelProtocol?.trim()
      || getApiConfigDefaultProtocol(nextApiConfig, next.modelType);
    if (isCapabilityPresetCompatible({
      preset,
      modelType: next.modelType,
      platform: nextApiConfig?.platform,
      effectiveProtocol: nextProtocol,
    })) {
      return next;
    }
    return {
      ...next,
      capabilityPresetCode: "",
      config: materializeEffectiveConfig(previous),
    };
  };

  const selectCustomCapabilities = () => {
    setForm(previous => previous.capabilityPresetCode
      ? {
          ...previous,
          capabilityPresetCode: "",
          config: materializeEffectiveConfig(previous),
        }
      : previous);
  };

  const applyCapabilityPreset = (preset: ModelPreset) => {
    setForm(previous => ({
      ...previous,
      capabilityPresetCode: preset.code,
      config: "",
    }));
  };

  const updateCustomCapabilityConfig = (config: string) => {
    setForm(previous => ({
      ...previous,
      capabilityPresetCode: "",
      config,
    }));
  };

  const updateOpenAiResponsesMode = (rawValue: string) => {
    const next = { ...effectiveModelConfig };
    OPENAI_RESPONSE_MODE_CONFIG_KEYS.forEach((key) => {
      delete next[key];
    });
    if (rawValue !== "__unset__") {
      next[OPENAI_RESPONSE_MODE_CONFIG_KEYS[0]] = rawValue;
    }

    const cleaned = Object.fromEntries(
      Object.entries(next).filter(([, value]) => value !== "" && value !== undefined && value !== null)
    );
    updateCustomCapabilityConfig(stringifyConfigObject(cleaned));
  };

  const handleSave = async () => {
    if (!form.name.trim() || !form.code.trim() || !form.apiConfigId || !protocolConfigured) return;
    setSaving(true);
    try {
      const normalizedConfig = sanitizeModelConfigJson({
        configJson: form.config,
        modelType: form.modelType,
        platform: selectedPlatform,
        supportReasoning: !!form.supportReasoning,
      });
      const normalizedCapabilityPresetCode = form.capabilityPresetCode?.trim() || "";
      const normalizedModelProtocol = hasRequestProtocol
        ? form.modelProtocol?.trim() || ""
        : undefined;

      if (editingModel) {
        const updateReq: AiModelUpdateReq = {
          id: editingModel.id,
          name: form.name,
          code: form.code,
          capabilityPresetCode: normalizedCapabilityPresetCode,
          modelProtocol: normalizedModelProtocol,
          modelType: form.modelType,
          description: form.description,
          config: normalizedConfig,
          maxConcurrency: form.maxConcurrency,
          defaultModel: form.defaultModel,
          supportVision: form.supportVision,
          supportReasoning: form.supportReasoning,
          contextWindow: form.contextWindow,
          apiConfigId: form.apiConfigId,
          status: form.status,
        };
        await aiModelApi.update(updateReq);
      } else {
        await aiModelApi.create({
          ...form,
          capabilityPresetCode: normalizedCapabilityPresetCode || undefined,
          modelProtocol: normalizedModelProtocol,
          config: normalizedConfig,
        });
      }
      onSaved();
      onOpenChange(false);
    } catch (err) {
      console.error("保存 AI 模型失败:", err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-4xl max-h-[calc(100vh-2rem)] sm:max-h-[88vh] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>{editingModel ? "编辑 AI 模型" : "新建 AI 模型"}</DialogTitle>
          <DialogDescription>填写模型标识并关联 API 配置；通常继承对应能力的默认协议，特殊模型可单独覆盖。</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 overflow-y-auto min-h-0 px-2 -mx-2">
          {/* 预设快速选择 */}
          {!editingModel && visiblePresets.length > 0 && (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground flex items-center gap-1">
                <Sparkles className="h-3 w-3" />
                从预设导入
              </Label>
              <div className="flex flex-wrap gap-1.5">
                {visiblePresets.map(preset => (
                  <button
                    key={preset.code}
                    type="button"
                    onClick={() => {
                      setForm(prev => ({
                        ...prev,
                        name: preset.name,
                        code: preset.modelCode || preset.code,
                        capabilityPresetCode: preset.code,
                        modelProtocol: preset.modelProtocol || "",
                        modelType: preset.modelType,
                        description: preset.description,
                        config: "",
                      }));
                    }}
                    className={cn(
                      "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs transition-all duration-200",
                      "border",
                      form.capabilityPresetCode === preset.code
                        ? "border-primary/50 bg-primary/10 text-primary font-medium"
                        : "border-border/30 text-muted-foreground hover:border-primary/30 hover:text-foreground hover:bg-white/5"
                    )}
                  >
                    {form.capabilityPresetCode === preset.code && <Check className="h-3 w-3" />}
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="grid gap-3 sm:grid-cols-2">
          {/* 模型名称 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">模型名称 <span className="text-destructive">*</span></Label>
            <Input
              placeholder="例如：claude-sonnet-4.5"
              value={form.name}
              onChange={e => updateField("name", e.target.value)}
              className="text-sm"
            />
          </div>

          {/* 模型标识 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">模型标识 (code) <span className="text-destructive">*</span></Label>
            <Input
              placeholder="例如：claude-sonnet-4.5"
              value={form.code}
              onChange={e => updateField("code", e.target.value)}
              className="text-sm font-mono"
            />
          </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
          {/* 模型类型 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">模型类型</Label>
            <Select
              value={form.modelType}
              onValueChange={v => {
                const nextType = v as number;
                setForm(prev => {
                  let next: AiModelFormState = { ...prev, modelType: nextType };
                  if (nextType !== 1) {
                    next.supportVision = false;
                    next.supportReasoning = false;
                    next.contextWindow = 0;
                  }
                  if (nextType !== prev.modelType) {
                    next.modelProtocol = "";
                  }
                  next = detachIncompatibleCapabilityPreset(prev, next, selectedApiConfig);
                  next.config = sanitizeModelConfigJson({
                    configJson: next.config,
                    modelType: nextType,
                    platform: selectedPlatform,
                    supportReasoning: nextType === 1 ? !!prev.supportReasoning : false,
                  });
                  return next;
                });
              }}
              items={MODEL_TYPE_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
            >
              <SelectTrigger className="w-full text-sm">
                <SelectValue placeholder="选择类型" />
              </SelectTrigger>
              <SelectContent className="text-sm">
                <SelectGroup>
                  {MODEL_TYPE_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          {/* API 配置 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">API 配置 <span className="text-destructive">*</span></Label>
            <Select
              value={form.apiConfigId}
              onValueChange={v => {
                const nextApiConfigId = v as number;
                const nextApiConfig = apiConfigs.find(c => c.id === nextApiConfigId);
                const nextPlatform = nextApiConfig?.platform;
                setForm(prev => {
                  let next: AiModelFormState = { ...prev, apiConfigId: nextApiConfigId };
                  if (!supportsReasoningConfig(nextPlatform)) {
                    next.supportReasoning = false;
                  }
                  next = detachIncompatibleCapabilityPreset(prev, next, nextApiConfig);
                  next.config = sanitizeModelConfigJson({
                    configJson: next.config,
                    modelType: prev.modelType,
                    platform: nextPlatform,
                    supportReasoning: prev.modelType === 1 && supportsReasoningConfig(nextPlatform)
                      ? !!prev.supportReasoning
                      : false,
                  });
                  return next;
                });
              }}
              items={apiConfigs.map(c => ({ value: c.id, label: `${c.name}${c.platform ? ` (${PLATFORM_LABELS[c.platform] || c.platform})` : ""}` }))}
            >
              <SelectTrigger className="w-full text-sm">
                <SelectValue placeholder="选择 API 配置" />
              </SelectTrigger>
              <SelectContent className="text-sm">
                <SelectGroup>
                  {apiConfigs.map(c => (
                    <SelectItem key={c.id} value={c.id} className="text-sm">
                      {c.name}
                      {c.platform && <span className="text-muted-foreground ml-1">({PLATFORM_LABELS[c.platform] || c.platform})</span>}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
          </div>

          {hasRequestProtocol && (
            <div className="rounded-xl border border-primary/20 bg-primary/[0.035] p-3 space-y-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <Label className="text-xs font-medium text-foreground">请求协议</Label>
                  <p className="mt-1 text-[11px] leading-4 text-muted-foreground">
                    优先使用模型覆盖；未覆盖时使用当前 API 配置为该能力设置的默认协议。
                  </p>
                </div>
                <span className="rounded-md border border-border/50 bg-background/70 px-2 py-1 text-[10px] text-muted-foreground">
                  模型覆盖 → API 配置默认
                </span>
              </div>

              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">协议覆盖（可选）</Label>
                <Select
                  value={form.modelProtocol || INHERIT_PROVIDER_PROTOCOL_VALUE}
                  onValueChange={value => {
                    const modelProtocol = value === INHERIT_PROVIDER_PROTOCOL_VALUE ? "" : String(value);
                    setForm(previous => detachIncompatibleCapabilityPreset(
                      previous,
                      { ...previous, modelProtocol },
                      selectedApiConfig
                    ));
                  }}
                  items={modelProtocolSelectOptions.map(option => ({ value: option.value, label: option.label }))}
                >
                  <SelectTrigger className="w-full text-sm">
                    <SelectValue placeholder={modelProtocolSelectOptions[0]?.label || "继承 API 配置默认"} />
                  </SelectTrigger>
                  <SelectContent className="text-sm">
                    <SelectGroup>
                      {modelProtocolSelectOptions.map(option => (
                        <SelectItem key={option.value} value={option.value} className="text-sm">
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <p className="text-[10px] leading-4 text-muted-foreground">
                  仅在这个模型需要偏离 API 配置默认值时设置，例如 New API 下的即梦、可灵或 Sora 视频。
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-1.5 border-t border-border/35 pt-2 text-[10px] text-muted-foreground">
                <span>有效协议：</span>
                <span className={cn(
                  "rounded px-1.5 py-0.5",
                  effectiveModelProtocol ? "bg-violet-500/10 text-violet-600" : "bg-destructive/10 text-destructive"
                )}>
                  {effectiveModelProtocol
                    ? formatSemanticLabel(effectiveModelProtocol, MODEL_PROTOCOL_LABELS)
                    : "未配置"}
                </span>
                <ChevronRight className="h-3 w-3" />
                <span className="rounded bg-sky-500/10 px-1.5 py-0.5 text-sky-600">
                  {modelProtocolOverride
                    ? "模型覆盖"
                    : apiConfigDefaultProtocol
                      ? `${MODEL_TYPE_LABELS[form.modelType]} API 配置默认`
                      : "缺少协议来源"}
                </span>
                <ChevronRight className="h-3 w-3" />
                <span className="rounded bg-muted px-1.5 py-0.5 text-foreground/75">
                  {selectedApiConfig?.name || "未选择 API 配置"}
                </span>
              </div>
              {!protocolConfigured && (
                <p className="text-[10px] leading-4 text-destructive">
                  当前模型和 API 配置都没有设置 {MODEL_TYPE_LABELS[form.modelType]} 请求协议，保存前必须补充一处。
                </p>
              )}
            </div>
          )}

          <CapabilityPresetSelect
            selectedCode={form.capabilityPresetCode}
            presets={compatibleCapabilityPresets}
            hasCustomConfig={hasCustomCapabilityConfig}
            onSelectCustom={selectCustomCapabilities}
            onApplyPreset={applyCapabilityPreset}
          />

          {/* 默认模型 */}
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => updateField("defaultModel", !form.defaultModel)}
              className={cn(
                "relative w-9 h-5 rounded-full transition-colors duration-200",
                form.defaultModel ? "bg-primary" : "bg-muted-foreground/30"
              )}
            >
              <span
                className={cn(
                  "absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200",
                  form.defaultModel && "translate-x-4"
                )}
              />
            </button>
            <Label className="text-xs text-muted-foreground cursor-pointer" onClick={() => updateField("defaultModel", !form.defaultModel)}>
              设为默认模型
            </Label>
          </div>

          <button
            type="button"
            onClick={() => setShowAdvanced(value => !value)}
            className="flex w-full items-center justify-between rounded-xl border border-border/40 bg-muted/15 px-3 py-2.5 text-left transition-colors hover:border-primary/35"
          >
            <span>
              <span className="block text-sm font-medium">高级设置</span>
              <span className="mt-0.5 block text-[10px] text-muted-foreground">模型能力、接口模式与自定义 JSON</span>
            </span>
            <ChevronRight className={cn("h-4 w-4 text-muted-foreground transition-transform", showAdvanced && "rotate-90")} />
          </button>

          {showAdvanced && showOpenAiResponsesMode && (
            <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
              <div>
                <Label className="text-[11px] text-muted-foreground">OpenAI 兼容对话接口</Label>
              </div>

              <Select
                value={apiModeValue}
                onValueChange={v => updateOpenAiResponsesMode(String(v))}
                items={[
                  { value: "__unset__", label: "自动" },
                  { value: "chat_completions", label: "Chat Completions" },
                  { value: "responses", label: "Responses API" },
                ]}
              >
                <SelectTrigger className="w-full text-sm">
                  <SelectValue placeholder="选择对话接口模式" />
                </SelectTrigger>
                <SelectContent className="text-sm">
                  <SelectGroup>
                    <SelectItem value="__unset__" className="text-sm">自动</SelectItem>
                    <SelectItem value="chat_completions" className="text-sm">Chat Completions</SelectItem>
                    <SelectItem value="responses" className="text-sm">Responses API</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          )}

          {showAdvanced && <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <Label className="text-xs text-muted-foreground">高级能力</Label>
              {selectedPlatform && (
                <span className="px-2 py-1 rounded-md bg-background/80 text-[10px] text-muted-foreground">
                  {PLATFORM_LABELS[selectedPlatform] || selectedPlatform}
                </span>
              )}
            </div>

            <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(240px,1fr))]">
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">最大并发数</Label>
                <Input
                  type="number"
                  min={0}
                  step={1}
                  placeholder="默认 5"
                  value={getPositiveNumberValue(form.maxConcurrency)}
                  onChange={e => updateMetaNumberField("maxConcurrency", e.target.value)}
                  className="text-sm font-mono"
                />
              </div>

              {form.modelType === 1 && (
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">上下文窗口</Label>
                  <Input
                    type="number"
                    min={0}
                    step={1}
                    placeholder="例如：128000"
                    value={getPositiveNumberValue(form.contextWindow)}
                    onChange={e => updateMetaNumberField("contextWindow", e.target.value)}
                    className="text-sm font-mono"
                  />
                </div>
              )}
            </div>

            {form.modelType === 1 && (
              <div className="space-y-2.5">
                <ToggleSettingCard
                  checked={!!form.supportReasoning}
                  title="支持思考"
                  description="控制推理能力。"
                  onToggle={() => {
                    setForm(prev => {
                      const nextSupportReasoning = !prev.supportReasoning;
                      return {
                        ...prev,
                        supportReasoning: nextSupportReasoning,
                        config: nextSupportReasoning ? prev.config : stripReasoningConfig(prev.config),
                      };
                    });
                  }}
                />

                <ToggleSettingCard
                  checked={!!form.supportVision}
                  title="支持视觉输入"
                  description="支持图片输入。"
                  onToggle={() => updateField("supportVision", !form.supportVision)}
                />
              </div>
            )}
          </div>}

          {/* 模型配置 */}
          {showAdvanced && <ModelConfigForm
            modelType={form.modelType}
            platform={selectedPlatform}
            supportReasoning={!!form.supportReasoning}
            configJson={stringifyConfigObject(effectiveModelConfig)}
            onChange={updateCustomCapabilityConfig}
          />}
        </div>

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button size="sm" onClick={handleSave} disabled={saving || !form.name.trim() || !form.code.trim() || !form.apiConfigId || !protocolConfigured}>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            {editingModel ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
