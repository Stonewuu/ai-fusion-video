"use client";

import { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useUnsavedDialog } from "@/components/settings/use-unsaved-dialog";
import {
  aiModelApi,
  MODEL_TYPE_LABELS,
  MODEL_TYPE_OPTIONS,
  PLATFORM_LABELS,
  type AiModel,
  type AiModelUpdateReq,
  type ApiConfig,
  type ModelPreset,
} from "@/lib/api/ai-model";
import type {
  ProviderDefinition,
  ProviderModelCandidate,
} from "@/lib/api/provider-catalog";
import { aiModelIssue, type ConfigIssue } from "@/lib/settings/form-validation";
import { filterModelPresets } from "@/lib/settings/model-preset-filter";
import {
  applyModelsDevMatch,
  findOfficialGooglePreset,
} from "@/lib/settings/models-dev-match";
import { Button } from "@/components/ui/button";
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
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { CapabilityPresetSelect } from "./capability-preset-select";
import { ComfyUiModelWorkflowField } from "./comfyui-model-workflow-field";
import { SponsorModelOptions } from "./sponsor-model-options";
import {
  AiModelAdvancedSection,
  type AdvancedSectionTab,
  type AiModelFormState,
} from "./ai-model-advanced-section";
import { AiModelProtocolSection } from "./ai-model-protocol-section";
import {
  buildModelCapabilityHighlights,
  detachIncompatiblePreset,
  materializeEffectiveModelConfig,
  parseModelRawJsonDraft,
} from "./ai-model-dialog-support";
import {
  formatSemanticLabel,
  getApiConfigDefaultProtocol,
  getFirstConfigString,
  getProtocolOptionsForModelType,
  INHERIT_PROVIDER_PROTOCOL_VALUE,
  isCapabilityPresetCompatible,
  mergeConfigObjects,
  MODEL_PROTOCOL_LABELS,
  OPENAI_RESPONSE_MODE_CONFIG_KEYS,
  parseConfigJson,
  sanitizeModelConfigJson,
  stringifyConfigObject,
  supportsOpenAiResponsesMode,
  supportsReasoningConfig,
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

export function AiModelDialog({
  open,
  onOpenChange,
  editingModel,
  apiConfigs,
  providers,
  defaultApiConfigId,
  initialCandidate,
  onSaved,
}: AiModelDialogProps) {
  const initialProviderId = apiConfigs.find(
    (config) => config.id === defaultApiConfigId,
  )?.providerId;
  const initialModelType =
    providers.find((provider) => provider.id === initialProviderId)
      ?.supportedModelTypes[0] ?? 1;

  const [saving, setSaving] = useState(false);
  const [issue, setIssue] = useState<ConfigIssue | null>(null);
  const [comfyUiCapabilityValid, setComfyUiCapabilityValid] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advancedTab, setAdvancedTab] = useState<AdvancedSectionTab>("capabilities");
  const [protocolExpanded, setProtocolExpanded] = useState(false);
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [rawJsonDraft, setRawJsonDraft] = useState("");
  const [rawJsonDirty, setRawJsonDirty] = useState(false);
  const [rawJsonError, setRawJsonError] = useState<string | null>(null);
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
    status: 1,
  });
  const [original, setOriginal] = useState<AiModelFormState | null>(null);

  const hasUnsavedChanges =
    !!original
    && (JSON.stringify(form) !== JSON.stringify(original) || rawJsonDirty);
  const changeOpen = useUnsavedDialog(open, hasUnsavedChanges, saving, onOpenChange);

  useEffect(() => {
    if (!open) {
      return;
    }
    const initializeForm = (value: AiModelFormState) => {
      setForm(value);
      setOriginal(value);
    };
    aiModelApi
      .presets()
      .then(setPresets)
      .catch((error) => {
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
      setProtocolExpanded(Boolean(editingModel.modelProtocol));
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
        supportVision:
          initialCandidate?.textCapabilities?.inputTypes.includes("image") ?? false,
        multimodalInputTypes: initialCandidate?.textCapabilities?.inputTypes ?? [],
        multimodalInputTransports:
          initialCandidate?.textCapabilities?.inputTransports ?? {},
        supportReasoning: initialCandidate?.textCapabilities?.reasoning ?? false,
        reasoningEffortLevels:
          initialCandidate?.textCapabilities?.reasoningEfforts ?? [],
        contextWindow: initialCandidate?.textCapabilities?.contextWindow ?? 0,
        apiConfigId: defaultApiConfigId,
        comfyuiWorkflowId: undefined,
        status: 1,
      });
      setProtocolExpanded(Boolean(initialCandidate?.modelProtocol));
    }
    setShowAdvanced(Boolean(initialCandidate?.configurationHint));
    setAdvancedTab("capabilities");
    setRawJsonDirty(false);
    setRawJsonError(null);
    setIssue(null);
  }, [open, editingModel, defaultApiConfigId, initialModelType, initialCandidate]);

  const updateField = <K extends keyof AiModelFormState>(
    key: K,
    value: AiModelFormState[K],
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    setIssue(null);
  };

  const selectedApiConfig = apiConfigs.find((config) => config.id === form.apiConfigId);
  const selectedProvider = providers.find(
    (provider) => provider.id === selectedApiConfig?.providerId,
  );
  const modelTypeOptions = MODEL_TYPE_OPTIONS.filter(
    (option) =>
      !selectedProvider || selectedProvider.supportedModelTypes.includes(option.value),
  );
  const selectedPlatform = selectedApiConfig?.platform;
  const requiresComfyUiWorkflow =
    selectedPlatform === "comfyui" && (form.modelType === 2 || form.modelType === 3);

  const selectedCapabilityPreset = form.capabilityPresetCode
    ? presets.find((preset) => preset.code === form.capabilityPresetCode) || null
    : null;
  const visiblePresets = filterModelPresets(
    presets,
    form.modelType,
    selectedPlatform,
    selectedProvider,
  );
  const selectedPresetConfig = selectedCapabilityPreset?.config as
    | Record<string, unknown>
    | undefined;
  const effectiveModelConfig = useMemo(
    () => mergeConfigObjects(selectedPresetConfig || {}, parseConfigJson(form.config)),
    [selectedPresetConfig, form.config],
  );
  const formattedEffectiveJson = useMemo(
    () =>
      Object.keys(effectiveModelConfig).length > 0
        ? JSON.stringify(effectiveModelConfig, null, 2)
        : "",
    [effectiveModelConfig],
  );

  useEffect(() => {
    if (!rawJsonDirty) {
      setRawJsonDraft(formattedEffectiveJson);
      setRawJsonError(null);
    }
  }, [formattedEffectiveJson, rawJsonDirty]);

  const showOpenAiResponsesMode =
    form.modelType === 1 && supportsOpenAiResponsesMode(selectedPlatform);
  const apiModeValue =
    getFirstConfigString(effectiveModelConfig, OPENAI_RESPONSE_MODE_CONFIG_KEYS)
    ?? "__unset__";
  const googlePlatform = ["gemini", "vertex_ai", "vertex_ai_api_key"].includes(
    selectedPlatform || "",
  );
  const hasRequestProtocol =
    form.modelType >= 1 && (form.modelType <= 3 || googlePlatform);
  const apiConfigDefaultProtocol = getApiConfigDefaultProtocol(
    selectedApiConfig,
    form.modelType,
  );
  const protocolOptions = getProtocolOptionsForModelType(
    form.modelType,
    selectedApiConfig?.providerId,
    providers,
  ).filter(
    (option) =>
      form.modelType <= 3
      || !googlePlatform
      || option.value === (selectedPlatform === "gemini" ? "gemini" : "vertex_ai"),
  );
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
  const protocolConfigured =
    !hasRequestProtocol
    || protocolOptions.some((option) => option.value === effectiveModelProtocol);
  const compatibleCapabilityPresets = visiblePresets.filter((preset) =>
    isCapabilityPresetCompatible({
      preset,
      modelType: form.modelType,
      platform: selectedPlatform,
      effectiveProtocol: effectiveModelProtocol,
    }),
  );
  const hasCustomCapabilityConfig = Boolean(
    form.config?.trim() && form.config.trim() !== "{}",
  );

  const selectCustomCapabilities = () => {
    setForm((previous) =>
      previous.capabilityPresetCode
        ? {
            ...previous,
            capabilityPresetCode: "",
            config: materializeEffectiveModelConfig(previous, presets),
          }
        : previous,
    );
    setRawJsonDirty(false);
  };

  const applyCapabilityPreset = (preset: ModelPreset) => {
    setForm((previous) => ({
      ...previous,
      capabilityPresetCode: preset.code,
      name: previous.name || preset.name,
      code: previous.code || preset.modelCode || preset.code,
      modelProtocol:
        previous.modelProtocol || apiConfigDefaultProtocol
          ? previous.modelProtocol
          : googlePlatform
            ? selectedPlatform === "gemini"
              ? "gemini"
              : "vertex_ai"
            : preset.modelProtocol || "",
      config: "",
      supportReasoning: preset.supportReasoning ?? previous.supportReasoning,
      reasoningEffortLevels:
        preset.reasoningEffortLevels ?? previous.reasoningEffortLevels,
      contextWindow: preset.contextWindow ?? previous.contextWindow,
      multimodalInputTypes:
        preset.multimodalInputTypes ?? previous.multimodalInputTypes,
      multimodalInputTransports:
        preset.multimodalInputTransports ?? previous.multimodalInputTransports,
      supportVision:
        (preset.multimodalInputTypes ?? previous.multimodalInputTypes)?.includes(
          "image",
        ) ?? false,
    }));
    setRawJsonDirty(false);
    setIssue(null);
  };

  const updateCustomCapabilityConfig = (config: string) => {
    setForm((previous) => ({ ...previous, capabilityPresetCode: "", config }));
    setRawJsonDirty(false);
    setIssue(null);
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
      Object.entries(next).filter(
        ([, value]) => value !== "" && value !== undefined && value !== null,
      ),
    );
    updateCustomCapabilityConfig(stringifyConfigObject(cleaned));
  };

  const handleApplyRawJson = () => {
    const parsed = parseModelRawJsonDraft(rawJsonDraft);
    if (!parsed.ok) {
      setRawJsonError(parsed.error);
      return;
    }
    updateCustomCapabilityConfig(parsed.value);
    setRawJsonError(null);
    toast.success("已应用自定义 JSON 参数");
  };

  const handleSave = async () => {
    const invalid = aiModelIssue({
      name: form.name,
      code: form.code,
      apiConfigId: form.apiConfigId,
      protocolConfigured,
      requiresComfyUiWorkflow,
      comfyuiWorkflowId: form.comfyuiWorkflowId,
      comfyUiCapabilityValid,
      modelTypeLabel: MODEL_TYPE_LABELS[form.modelType] || "当前",
      rawJsonDirty,
      rawJsonDraft,
    });
    if (invalid) {
      setIssue(invalid);
      if (invalid.field === "model-protocol") {
        setProtocolExpanded(true);
      }
      if (invalid.field === "model-raw-json") {
        setShowAdvanced(true);
        setAdvancedTab("json");
        setRawJsonError(invalid.message);
      }
      requestAnimationFrame(() => {
        document.getElementById(invalid.field)?.focus();
      });
      return;
    }

    let nextConfigRaw = form.config;
    let nextPresetCode = form.capabilityPresetCode?.trim() || "";
    if (rawJsonDirty) {
      const parsed = parseModelRawJsonDraft(rawJsonDraft);
      if (!parsed.ok) {
        setRawJsonError(parsed.error);
        setShowAdvanced(true);
        setAdvancedTab("json");
        setIssue({ field: "model-raw-json", message: parsed.error });
        requestAnimationFrame(() => {
          document.getElementById("model-raw-json")?.focus();
        });
        return;
      }
      nextConfigRaw = parsed.value;
      nextPresetCode = "";
    }

    setIssue(null);
    setSaving(true);
    try {
      const normalizedConfig = requiresComfyUiWorkflow
        ? nextConfigRaw
        : sanitizeModelConfigJson({
            configJson: nextConfigRaw,
            modelType: form.modelType,
            platform: selectedPlatform,
            supportReasoning: !!form.supportReasoning,
          });
      const normalizedModelProtocol = hasRequestProtocol
        ? form.modelProtocol?.trim() || ""
        : undefined;

      if (editingModel) {
        const updateReq: AiModelUpdateReq = {
          id: editingModel.id,
          name: form.name.trim(),
          code: form.code.trim(),
          capabilityPresetCode: nextPresetCode,
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
          name: form.name.trim(),
          code: form.code.trim(),
          capabilityPresetCode: nextPresetCode || undefined,
          modelProtocol: normalizedModelProtocol,
          config: normalizedConfig,
        });
      }
      setRawJsonDirty(false);
      onSaved();
      onOpenChange(false);
      toast.success(editingModel ? "模型已保存" : "模型已创建");
    } catch (err) {
      console.error("保存 AI 模型失败:", err);
      toastApiError(err, "保存 AI 模型失败");
    } finally {
      setSaving(false);
    }
  };

  const activeCapabilityHighlights = buildModelCapabilityHighlights(
    form,
    effectiveModelConfig,
  );

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] w-[calc(100%-2rem)] flex-col overflow-hidden sm:max-h-[88vh] sm:max-w-4xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>{editingModel ? "编辑 AI 模型" : "新建 AI 模型"}</DialogTitle>
          <DialogDescription>配置模型标识、能力预设与调用参数。</DialogDescription>
        </DialogHeader>

        <DialogScrollBody className="flex flex-col gap-4 p-2">
          <fieldset disabled={saving} className="flex min-w-0 flex-col gap-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex min-w-0 flex-col gap-2">
                <Label htmlFor="model-name" className="text-xs text-muted-foreground">
                  模型名称 <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="model-name"
                  aria-invalid={issue?.field === "model-name"}
                  placeholder="例如：Claude Sonnet 4.5"
                  value={form.name}
                  onChange={(event) => updateField("name", event.target.value)}
                  className="text-sm"
                />
              </div>

              <div className="flex min-w-0 flex-col gap-2">
                <Label htmlFor="model-code" className="text-xs text-muted-foreground">
                  模型标识（code） <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="model-code"
                  aria-invalid={issue?.field === "model-code"}
                  placeholder="例如：claude-sonnet-4-5"
                  value={form.code}
                  onChange={(event) => updateField("code", event.target.value)}
                  className="font-mono text-sm"
                />
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex min-w-0 flex-col gap-2">
                <Label htmlFor="model-type" className="text-xs text-muted-foreground">
                  模型类型
                </Label>
                <Select
                  value={form.modelType}
                  onValueChange={(value) => {
                    const nextType = value as number;
                    setForm((prev) => {
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
                      next = detachIncompatiblePreset(prev, next, selectedApiConfig, presets);
                      next.config = sanitizeModelConfigJson({
                        configJson: next.config,
                        modelType: nextType,
                        platform: selectedPlatform,
                        supportReasoning: nextType === 1 ? !!prev.supportReasoning : false,
                      });
                      return next;
                    });
                    setRawJsonDirty(false);
                    setIssue(null);
                  }}
                  items={modelTypeOptions.map((option) => ({
                    value: option.value,
                    label: option.label,
                  }))}
                >
                  <SelectTrigger id="model-type" className="w-full text-sm">
                    <SelectValue placeholder="选择类型" />
                  </SelectTrigger>
                  <SelectContent className="text-sm">
                    <SelectGroup>
                      {modelTypeOptions.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value} className="text-sm">
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex min-w-0 flex-col gap-2">
                <Label htmlFor="model-api" className="text-xs text-muted-foreground">
                  API 配置 <span className="text-destructive">*</span>
                </Label>
                <Select
                  value={form.apiConfigId}
                  onValueChange={(value) => {
                    const nextApiConfigId = value as number;
                    const nextApiConfig = apiConfigs.find(
                      (config) => config.id === nextApiConfigId,
                    );
                    const nextPlatform = nextApiConfig?.platform;
                    const nextProvider = providers.find(
                      (provider) => provider.id === nextApiConfig?.providerId,
                    );
                    setForm((prev) => {
                      let nextModelType =
                        nextPlatform === "comfyui" && prev.modelType !== 2 && prev.modelType !== 3
                          ? 2
                          : prev.modelType;
                      if (
                        nextProvider
                        && !nextProvider.supportedModelTypes.includes(nextModelType)
                      ) {
                        nextModelType = nextProvider.supportedModelTypes[0];
                      }
                      let next: AiModelFormState = {
                        ...prev,
                        apiConfigId: nextApiConfigId,
                        modelType: nextModelType,
                        modelProtocol:
                          nextPlatform !== "comfyui"
                          && getProtocolOptionsForModelType(
                            nextModelType,
                            nextApiConfig?.providerId,
                            providers,
                          ).some((option) => option.value === prev.modelProtocol)
                            ? prev.modelProtocol
                            : "",
                        comfyuiWorkflowId: undefined,
                        maxConcurrency: nextPlatform === "comfyui" ? 1 : prev.maxConcurrency,
                      };
                      if (!supportsReasoningConfig(nextPlatform)) {
                        next.supportReasoning = false;
                        next.reasoningEffortLevels = [];
                      }
                      next = detachIncompatiblePreset(prev, next, nextApiConfig, presets);
                      next.config = sanitizeModelConfigJson({
                        configJson: next.config,
                        modelType: nextModelType,
                        platform: nextPlatform,
                        supportReasoning:
                          nextModelType === 1 && supportsReasoningConfig(nextPlatform)
                            ? !!prev.supportReasoning
                            : false,
                      });
                      return next;
                    });
                    setRawJsonDirty(false);
                    setIssue(null);
                  }}
                  items={apiConfigs.map((config) => ({
                    value: config.id,
                    label: `${config.name}${config.platform ? ` (${PLATFORM_LABELS[config.platform] || config.platform})` : ""}`,
                  }))}
                >
                  <SelectTrigger
                    id="model-api"
                    aria-invalid={issue?.field === "model-api"}
                    className="w-full text-sm"
                  >
                    <SelectValue placeholder="选择 API 配置" />
                  </SelectTrigger>
                  <SelectContent className="text-sm">
                    <SelectGroup>
                      {apiConfigs.map((config) => (
                        <SelectItem key={config.id} value={config.id} className="text-sm">
                          {config.name}
                          {config.platform && (
                            <span className="ml-1 text-muted-foreground">
                              ({PLATFORM_LABELS[config.platform] || config.platform})
                            </span>
                          )}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {requiresComfyUiWorkflow && (
              <div id="comfyui-workflow" tabIndex={-1}>
                <ComfyUiModelWorkflowField
                  apiConfigId={form.apiConfigId!}
                  modelType={form.modelType as 2 | 3}
                  value={form.comfyuiWorkflowId}
                  configJson={form.config}
                  capabilityPresetCode={form.capabilityPresetCode}
                  configuredLimits={effectiveModelConfig}
                  onValidityChange={setComfyUiCapabilityValid}
                  onChange={(workflowId, config) => {
                    setForm((previous) => ({
                      ...previous,
                      comfyuiWorkflowId: workflowId,
                      config,
                    }));
                    setRawJsonDirty(false);
                    setIssue(null);
                  }}
                />
              </div>
            )}

            {!requiresComfyUiWorkflow && (
              <div className="flex flex-col gap-2">
                <CapabilityPresetSelect
                  selectedCode={form.capabilityPresetCode}
                  presets={compatibleCapabilityPresets}
                  allPresets={presets}
                  hasCustomConfig={hasCustomCapabilityConfig}
                  modelCode={form.code}
                  modelType={form.modelType}
                  apiConfigId={form.apiConfigId}
                  platform={selectedPlatform}
                  matchedName={
                    typeof effectiveModelConfig.catalogMatch === "object"
                    && effectiveModelConfig.catalogMatch !== null
                      ? (effectiveModelConfig.catalogMatch as { modelId?: string }).modelId
                      : undefined
                  }
                  onSelectCustom={selectCustomCapabilities}
                  onApplyPreset={applyCapabilityPreset}
                  onApplyOnline={(candidate, fetchedAt) => {
                    const official = findOfficialGooglePreset(
                      candidate,
                      presets,
                      selectedPlatform,
                    );
                    const matched = applyModelsDevMatch(
                      form,
                      candidate,
                      fetchedAt,
                      official,
                    );
                    setForm({
                      ...matched,
                      name: form.name || candidate.name,
                      code: form.code || official?.modelCode || candidate.modelId,
                    });
                    setRawJsonDirty(false);
                    setShowAdvanced(true);
                  }}
                />
                {activeCapabilityHighlights.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    当前能力：{activeCapabilityHighlights.join(" · ")}
                  </p>
                )}
              </div>
            )}

            {hasRequestProtocol && (
              <AiModelProtocolSection
                modelType={form.modelType}
                apiConfigName={selectedApiConfig?.name}
                apiConfigDefaultProtocol={apiConfigDefaultProtocol}
                modelProtocolOverride={modelProtocolOverride}
                effectiveModelProtocol={effectiveModelProtocol}
                protocolConfigured={protocolConfigured}
                protocolExpanded={protocolExpanded}
                onToggleExpand={() => setProtocolExpanded((value) => !value)}
                options={modelProtocolSelectOptions}
                invalid={issue?.field === "model-protocol"}
                onSelectProtocol={(modelProtocol) => {
                  setForm((previous) =>
                    detachIncompatiblePreset(
                      previous,
                      { ...previous, modelProtocol },
                      selectedApiConfig,
                      presets,
                    ),
                  );
                  setIssue(null);
                }}
              />
            )}

            <div className={cn("grid gap-3", editingModel ? "sm:grid-cols-2" : "grid-cols-1")}>
              <div className="flex items-center justify-between gap-3 rounded-xl border border-border/30 bg-background/70 px-3.5 py-2.5">
                <div className="flex min-w-0 flex-col gap-0.5">
                  <Label htmlFor="model-default-switch" className="cursor-pointer text-sm font-medium">
                    设为默认模型
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    作为{MODEL_TYPE_LABELS[form.modelType]}能力的全局默认选项。
                  </p>
                </div>
                <Switch
                  id="model-default-switch"
                  checked={!!form.defaultModel}
                  onCheckedChange={(checked) => updateField("defaultModel", checked)}
                />
              </div>

              {editingModel && (
                <div className="flex items-center justify-between gap-3 rounded-xl border border-border/30 bg-background/70 px-3.5 py-2.5">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label htmlFor="model-status-switch" className="cursor-pointer text-sm font-medium">
                      启用此模型
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      停用后保留参数设置，暂不出现在模型选择器中。
                    </p>
                  </div>
                  <Switch
                    id="model-status-switch"
                    checked={form.status !== 0}
                    onCheckedChange={(enabled) => updateField("status", enabled ? 1 : 0)}
                  />
                </div>
              )}
            </div>

            {effectiveModelProtocol === "metaso_video" && (
              <SponsorModelOptions
                config={effectiveModelConfig}
                onChange={(key, enabled) => {
                  setForm((previous) => ({
                    ...previous,
                    config: JSON.stringify({
                      ...parseConfigJson(previous.config),
                      [key]: enabled,
                    }),
                  }));
                  setRawJsonDirty(false);
                }}
              />
            )}

            <AiModelAdvancedSection
              open={showAdvanced}
              onOpenChange={setShowAdvanced}
              activeTab={advancedTab}
              onActiveTabChange={setAdvancedTab}
              form={form}
              setForm={setForm}
              selectedPlatform={selectedPlatform}
              selectedCapabilityPreset={selectedCapabilityPreset}
              effectiveModelConfig={effectiveModelConfig}
              showOpenAiResponsesMode={showOpenAiResponsesMode}
              apiModeValue={apiModeValue}
              onUpdateOpenAiResponsesMode={updateOpenAiResponsesMode}
              onUpdateCustomCapabilityConfig={updateCustomCapabilityConfig}
              rawJsonDraft={rawJsonDraft}
              rawJsonDirty={rawJsonDirty}
              rawJsonError={rawJsonError}
              onRawJsonDraftChange={(value) => {
                setRawJsonDraft(value);
                setRawJsonDirty(value !== formattedEffectiveJson);
                setRawJsonError(null);
                setIssue(null);
              }}
              onApplyRawJson={handleApplyRawJson}
              onResetRawJson={() => {
                setRawJsonDraft(formattedEffectiveJson);
                setRawJsonDirty(false);
                setRawJsonError(null);
                setIssue(null);
              }}
              invalidField={issue?.field}
            />
          </fieldset>
        </DialogScrollBody>

        {issue && (
          <p id="model-form-error" role="alert" className="shrink-0 text-sm text-destructive">
            {issue.message}
          </p>
        )}

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" />}>取消</DialogClose>
          <Button type="button" onClick={handleSave} disabled={saving}>
            {saving && <Loader2 className="animate-spin" />}
            {editingModel ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
