"use client";

import { useEffect, useState } from "react";
import { useUnsavedDialog } from "@/components/settings/use-unsaved-dialog";
import { ChevronRight, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
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
  type MultimodalInputTransports,
  type MultimodalInputType,
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
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { CapabilityPresetSelect } from "./capability-preset-select";
import { filterModelPresets } from "@/lib/settings/model-preset-filter";
import { applyModelsDevMatch, findOfficialGooglePreset } from "@/lib/settings/models-dev-match";
import { ModelConfigForm } from "./model-config-form";
import { SponsorModelOptions } from "./sponsor-model-options";
import type { ProviderModelCandidate } from "@/lib/api/provider-catalog";
import { MultimodalCapabilityEditor } from "./multimodal-capability-editor";
import { ReasoningEffortLevelsEditor } from "./reasoning-effort-levels-editor";
import { ComfyUiModelWorkflowField } from "./comfyui-model-workflow-field";
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
  providers: ProviderDefinition[];
  defaultApiConfigId?: number;
  initialCandidate?: ProviderModelCandidate | null;
  onSaved: () => void;
}

type AiModelFormState = AiModelCreateReq & { id?: number; status?: number };

export function AiModelDialog({ open, onOpenChange, editingModel, apiConfigs, providers, defaultApiConfigId, initialCandidate, onSaved }: AiModelDialogProps) {
  const initialProviderId = apiConfigs.find((config) => config.id === defaultApiConfigId)?.providerId;
  const initialModelType = providers.find((provider) => provider.id === initialProviderId)?.supportedModelTypes[0] ?? 1;
  const [saving, setSaving] = useState(false);
  const [comfyUiCapabilityValid, setComfyUiCapabilityValid] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [form, setForm] = useState<AiModelFormState>({
    name: "",
    code: "",
    capabilityPresetCode: "",
    modelProtocol: "",
    modelType: 1,
    maxConcurrency: 5,
    supportVision: false,
    multimodalInputTypes: [],
    multimodalInputTransports: {},
    supportReasoning: false,
    reasoningEffortLevels: [],
    contextWindow: 0,
  });
  const [presets, setPresets] = useState<ModelPreset[]>([]);

  const [original, setOriginal] = useState<AiModelFormState | null>(null);
  const changeOpen = useUnsavedDialog(open, !!original && JSON.stringify(form) !== JSON.stringify(original), saving, onOpenChange);

  useEffect(() => {
    if (open) {
      const initializeForm = (value: AiModelFormState) => { setForm(value); setOriginal(value); };
      aiModelApi.presets().then(setPresets).catch((error) => {
        console.error("加载模型预设失败:", error);
        toastApiError(error, "加载模型预设失败");
      });
      if (editingModel) {
        initializeForm({
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
          multimodalInputTypes: editingModel.multimodalInputTypes,
          multimodalInputTransports: editingModel.multimodalInputTransports,
          supportReasoning: editingModel.supportReasoning,
          reasoningEffortLevels: editingModel.reasoningEffortLevels ?? [],
          contextWindow: editingModel.contextWindow ?? 0,
          apiConfigId: editingModel.apiConfigId ?? undefined,
          comfyuiWorkflowId: editingModel.comfyuiWorkflowId ?? undefined,
          status: editingModel.status,
        });
      } else {
        initializeForm({
          name: initialCandidate?.name ?? "",
          code: initialCandidate?.id ?? "",
          capabilityPresetCode: initialCandidate?.capabilityPresetCode ?? "",
          modelProtocol: initialCandidate?.modelProtocol ?? "",
          modelType: initialCandidate?.modelType ?? initialModelType,
          config: initialCandidate?.config,
          maxConcurrency: 5,
          defaultModel: false,
          supportVision: initialCandidate?.textCapabilities?.inputTypes.includes("image") ?? false,
          multimodalInputTypes: initialCandidate?.textCapabilities?.inputTypes ?? [],
          multimodalInputTransports: initialCandidate?.textCapabilities?.inputTransports ?? {},
          supportReasoning: initialCandidate?.textCapabilities?.reasoning ?? false,
          reasoningEffortLevels: initialCandidate?.textCapabilities?.reasoningEfforts ?? [],
          contextWindow: initialCandidate?.textCapabilities?.contextWindow ?? 0,
          apiConfigId: defaultApiConfigId,
          comfyuiWorkflowId: undefined,
        });
      }
      setShowAdvanced(!!initialCandidate?.configurationHint);
    }
  }, [open, editingModel, defaultApiConfigId, initialModelType, initialCandidate]);

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
  const selectedProvider = providers.find((provider) => provider.id === selectedApiConfig?.providerId);
  const modelTypeOptions = MODEL_TYPE_OPTIONS.filter((option) =>
    !selectedProvider || selectedProvider.supportedModelTypes.includes(option.value));
  const selectedPlatform = selectedApiConfig?.platform;
  const requiresComfyUiWorkflow = selectedPlatform === "comfyui" && (form.modelType === 2 || form.modelType === 3);

  const selectedCapabilityPreset = form.capabilityPresetCode
    ? presets.find(p => p.code === form.capabilityPresetCode) || null
    : null;
  const visiblePresets = filterModelPresets(presets, form.modelType, selectedPlatform, selectedProvider);
  const selectedPresetConfig = selectedCapabilityPreset?.config as Record<string, unknown> | undefined;
  const effectiveModelConfig = mergeConfigObjects(selectedPresetConfig || {}, parseConfigJson(form.config));
  const showOpenAiResponsesMode = form.modelType === 1 && supportsOpenAiResponsesMode(selectedPlatform);
  const apiModeValue = getFirstConfigString(effectiveModelConfig, OPENAI_RESPONSE_MODE_CONFIG_KEYS) ?? "__unset__";
  const googlePlatform = ["gemini", "vertex_ai", "vertex_ai_api_key"].includes(selectedPlatform || "");
  const hasRequestProtocol = form.modelType >= 1 && (form.modelType <= 3 || googlePlatform);
  const apiConfigDefaultProtocol = getApiConfigDefaultProtocol(selectedApiConfig, form.modelType);
  const protocolOptions = getProtocolOptionsForModelType(form.modelType, selectedApiConfig?.providerId, providers)
    .filter(option => form.modelType <= 3 || !googlePlatform || option.value === (selectedPlatform === "gemini" ? "gemini" : "vertex_ai"));
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
  const protocolConfigured = !hasRequestProtocol || protocolOptions.some((option) => option.value === effectiveModelProtocol);
  const comfyUiWorkflowConfigured = !requiresComfyUiWorkflow || (Boolean(form.comfyuiWorkflowId) && comfyUiCapabilityValid);
  const compatibleCapabilityPresets = visiblePresets.filter(preset => isCapabilityPresetCompatible({
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
    if ((!preset.providerId || preset.providerId === nextApiConfig?.providerId) && isCapabilityPresetCompatible({
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
      name: previous.name || preset.name,
      code: previous.code || preset.modelCode || preset.code,
      modelProtocol: previous.modelProtocol || apiConfigDefaultProtocol ? previous.modelProtocol
        : googlePlatform ? selectedPlatform === "gemini" ? "gemini" : "vertex_ai" : preset.modelProtocol || "",
      config: "",
      supportReasoning: preset.supportReasoning ?? previous.supportReasoning,
      reasoningEffortLevels: preset.reasoningEffortLevels ?? previous.reasoningEffortLevels,
      contextWindow: preset.contextWindow ?? previous.contextWindow,
      multimodalInputTypes: preset.multimodalInputTypes ?? previous.multimodalInputTypes,
      multimodalInputTransports: preset.multimodalInputTransports ?? previous.multimodalInputTransports,
      supportVision: (preset.multimodalInputTypes ?? previous.multimodalInputTypes)?.includes("image") ?? false,
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
    if (!form.name.trim() || !form.code.trim() || !form.apiConfigId || !protocolConfigured || !comfyUiWorkflowConfigured) {
      return;
    }
    setSaving(true);
    try {
      const normalizedConfig = requiresComfyUiWorkflow ? form.config : sanitizeModelConfigJson({
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
          multimodalInputTypes: form.multimodalInputTypes,
          multimodalInputTransports: form.multimodalInputTransports,
          supportReasoning: form.supportReasoning,
          reasoningEffortLevels: form.reasoningEffortLevels,
          contextWindow: form.contextWindow,
          apiConfigId: form.apiConfigId,
          comfyuiWorkflowId: requiresComfyUiWorkflow
            ? form.comfyuiWorkflowId
            : editingModel.comfyuiWorkflowId
              ? 0
              : undefined,
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
      toastApiError(err, "保存 AI 模型失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogContent className="w-[calc(100%-2rem)] sm:max-w-4xl max-h-[calc(100vh-2rem)] sm:max-h-[88vh] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>{editingModel ? "编辑 AI 模型" : "新建 AI 模型"}</DialogTitle>
          <DialogDescription>设置模型名称、调用标识和 API 配置。</DialogDescription>
        </DialogHeader>

        <DialogScrollBody className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
          {/* 模型名称 */}
          <div className="space-y-1.5">
            <Label htmlFor="model-name" className="text-xs text-muted-foreground">模型名称 <span className="text-destructive">*</span></Label>
            <Input id="model-name"
              placeholder="例如：claude-sonnet-4.5"
              value={form.name}
              onChange={e => updateField("name", e.target.value)}
              className="text-sm"
            />
          </div>

          {/* 模型标识 */}
          <div className="space-y-1.5">
            <Label htmlFor="model-code" className="text-xs text-muted-foreground">模型标识 (code) <span className="text-destructive">*</span></Label>
            <Input id="model-code"
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
            <Label htmlFor="model-type" className="text-xs text-muted-foreground">模型类型</Label>
            <Select
              value={form.modelType}
              onValueChange={v => {
                const nextType = v as number;
                setForm(prev => {
                  let next: AiModelFormState = { ...prev, modelType: nextType };
                  if (nextType !== 1) {
                    next.supportVision = false;
                    next.multimodalInputTypes = [];
                    next.multimodalInputTransports = {};
                    next.supportReasoning = false;
                    next.reasoningEffortLevels = [];
                    next.contextWindow = 0;
                  }
                  if (nextType !== prev.modelType) {
                    next.modelProtocol = "";
                    next.comfyuiWorkflowId = undefined;
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
              items={modelTypeOptions.map(o => ({ value: o.value, label: o.label }))}
            >
              <SelectTrigger id="model-type" className="w-full text-sm">
                <SelectValue placeholder="选择类型" />
              </SelectTrigger>
              <SelectContent className="text-sm">
                <SelectGroup>
                  {modelTypeOptions.map(opt => (
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
            <Label htmlFor="model-api" className="text-xs text-muted-foreground">API 配置 <span className="text-destructive">*</span></Label>
            <Select
              value={form.apiConfigId}
              onValueChange={v => {
                const nextApiConfigId = v as number;
                const nextApiConfig = apiConfigs.find(c => c.id === nextApiConfigId);
                const nextPlatform = nextApiConfig?.platform;
                const nextProvider = providers.find((provider) => provider.id === nextApiConfig?.providerId);
                setForm(prev => {
                  let nextModelType = nextPlatform === "comfyui" && prev.modelType !== 2 && prev.modelType !== 3
                    ? 2
                    : prev.modelType;
                  if (nextProvider && !nextProvider.supportedModelTypes.includes(nextModelType)) {
                    nextModelType = nextProvider.supportedModelTypes[0];
                  }
                  let next: AiModelFormState = {
                    ...prev,
                    apiConfigId: nextApiConfigId,
                    modelType: nextModelType,
                    modelProtocol: nextPlatform !== "comfyui" && getProtocolOptionsForModelType(nextModelType, nextApiConfig?.providerId, providers)
                      .some((option) => option.value === prev.modelProtocol) ? prev.modelProtocol : "",
                    comfyuiWorkflowId: undefined,
                    maxConcurrency: nextPlatform === "comfyui" ? 1 : prev.maxConcurrency,
                  };
                  if (!supportsReasoningConfig(nextPlatform)) {
                    next.supportReasoning = false;
                    next.reasoningEffortLevels = [];
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
              <SelectTrigger id="model-api" className="w-full text-sm">
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

          {requiresComfyUiWorkflow && (
            <ComfyUiModelWorkflowField
              apiConfigId={form.apiConfigId!}
              modelType={form.modelType as 2 | 3}
              value={form.comfyuiWorkflowId}
              configJson={form.config}
              capabilityPresetCode={form.capabilityPresetCode}
              configuredLimits={effectiveModelConfig}
              onValidityChange={setComfyUiCapabilityValid}
              onChange={(workflowId, config) => setForm(previous => ({
                ...previous,
                comfyuiWorkflowId: workflowId,
                config,
              }))}
            />
          )}

          {hasRequestProtocol && (
            <div className="rounded-xl border border-primary/20 bg-primary/[0.035] p-3 space-y-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <Label className="text-xs font-medium text-foreground">请求协议</Label>
                  <p className="mt-1 text-[11px] leading-4 text-muted-foreground">
                    选择模型使用的接口协议。
                  </p>
                </div>
                <span className="rounded-md border border-border/50 bg-background/70 px-2 py-1 text-[10px] text-muted-foreground">
                  可继承 API 默认设置
                </span>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="model-protocol" className="text-xs text-muted-foreground">协议覆盖（可选）</Label>
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
                  <SelectTrigger id="model-protocol" className="w-full text-sm">
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
                  继承 API 默认设置，或为当前模型单独选择。
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

          {!requiresComfyUiWorkflow && <CapabilityPresetSelect
            selectedCode={form.capabilityPresetCode}
            presets={compatibleCapabilityPresets}
            allPresets={presets}
            hasCustomConfig={hasCustomCapabilityConfig}
            modelCode={form.code}
            modelType={form.modelType}
            apiConfigId={form.apiConfigId}
            platform={selectedPlatform}
            matchedName={typeof effectiveModelConfig.catalogMatch === "object" && effectiveModelConfig.catalogMatch !== null
              ? (effectiveModelConfig.catalogMatch as { modelId?: string }).modelId : undefined}
            onSelectCustom={selectCustomCapabilities}
            onApplyPreset={applyCapabilityPreset}
            onApplyOnline={(candidate, fetchedAt) => {
              const official = findOfficialGooglePreset(candidate, presets, selectedPlatform);
              const matched = applyModelsDevMatch(form, candidate, fetchedAt, official);
              setForm({ ...matched, name: form.name || candidate.name,
                code: form.code || official?.modelCode || candidate.modelId });
              setShowAdvanced(true);
            }}
          />}

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
                        reasoningEffortLevels: nextSupportReasoning
                          ? prev.reasoningEffortLevels?.length
                            ? prev.reasoningEffortLevels
                            : selectedCapabilityPreset?.reasoningEffortLevels ?? []
                          : [],
                        config: nextSupportReasoning ? prev.config : stripReasoningConfig(prev.config),
                      };
                    });
                  }}
                />

                {form.supportReasoning ? (
                  <ReasoningEffortLevelsEditor
                    levels={form.reasoningEffortLevels ?? []}
                    onChange={(levels) => updateField("reasoningEffortLevels", levels)}
                  />
                ) : null}

                <MultimodalCapabilityEditor
                  inputTypes={form.multimodalInputTypes ?? []}
                  transports={form.multimodalInputTransports ?? {}}
                  onChange={(inputTypes: MultimodalInputType[], transports: MultimodalInputTransports) => {
                    setForm(previous => ({
                      ...previous,
                      supportVision: inputTypes.includes("image"),
                      multimodalInputTypes: inputTypes,
                      multimodalInputTransports: transports,
                    }));
                  }}
                />
              </div>
            )}
          </div>}

          {/* 模型配置 */}
          {effectiveModelProtocol === "metaso_video" && <SponsorModelOptions
            config={effectiveModelConfig}
            onChange={(key, enabled) => setForm((previous) => ({
              ...previous,
              config: JSON.stringify({ ...parseConfigJson(previous.config), [key]: enabled }),
            }))}
          />}
          {showAdvanced && <ModelConfigForm
            modelType={form.modelType}
            platform={selectedPlatform}
            supportReasoning={!!form.supportReasoning}
            configJson={stringifyConfigObject(effectiveModelConfig)}
            onChange={updateCustomCapabilityConfig}
          />}
        </DialogScrollBody>

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button size="sm" onClick={handleSave} disabled={saving || !form.name.trim() || !form.code.trim() || !form.apiConfigId || !protocolConfigured || !comfyUiWorkflowConfigured}>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            {editingModel ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
