"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Bot,
  Settings2,
  Loader2,
  Plus,
  Trash2,
  Edit2,
  Eye,
  EyeOff,
  Star,
  ChevronRight,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  aiModelApi,
  apiConfigApi,
  PLATFORM_OPTIONS,
  PLATFORM_LABELS,
  MODEL_TYPE_OPTIONS,
  MODEL_TYPE_LABELS,
  type AiModel,
  type ApiConfig,
  type AiModelCreateReq,
  type AiModelUpdateReq,
  type ApiConfigSaveReq,
  type ModelPreset,
  type RemoteModel,
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
import { Sparkles, Check, CloudDownload, Search } from "lucide-react";
import {
  containerVariants,
  itemVariants,
  platformIconColors,
  maskSecret,
  getPlatformFields,
} from "../_shared";

// ============================================================
// API 配置 Dialog
// ============================================================

interface ApiConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingConfig: ApiConfig | null;
  onSaved: () => void;
}

function ApiConfigDialog({ open, onOpenChange, editingConfig, onSaved }: ApiConfigDialogProps) {
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
          apiUrl: editingConfig.apiUrl || "",
          autoAppendV1Path: editingConfig.autoAppendV1Path ?? true,
          apiKey: editingConfig.apiKey || "",
          appId: editingConfig.appId || "",
          appSecret: editingConfig.appSecret || "",
          status: editingConfig.status,
          remark: editingConfig.remark || "",
        });
      } else {
        setForm({ name: "", platform: "openai_compatible", apiUrl: "", autoAppendV1Path: true, apiKey: "", appId: "", appSecret: "", status: 1 });
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
      if (editingConfig) {
        await apiConfigApi.update(form);
      } else {
        await apiConfigApi.create(form);
      }
      onSaved();
      onOpenChange(false);
    } catch (err) {
      console.error("保存 API 配置失败:", err);
    } finally {
      setSaving(false);
    }
  };

  const fields = getPlatformFields(form.platform || "");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{editingConfig ? "编辑 API 配置" : "新建 API 配置"}</DialogTitle>
          <DialogDescription>
            配置外部 AI 服务的 API 接入信息
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
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

          {/* 平台选择 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">平台</Label>
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
                <SelectValue placeholder="选择平台" />
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
          </div>

          {/* 动态平台字段 */}
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
                    className="min-h-28 text-sm"
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
                    开启后将请求发送到 /v1/chat/completions、/v1/models；关闭后改为不带 /v1 的路径。
                  </p>
                </div>
              </div>
            </div>
          )}

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

        <DialogFooter>
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button size="sm" onClick={handleSave} disabled={saving || !form.name.trim()}>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            {editingConfig ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// 模型配置可视化表单
// ============================================================

/** 解析 config JSON 字符串为对象，解析失败返回空对象 */
function parseConfigJson(json: string | undefined | null): Record<string, unknown> {
  if (!json) return {};
  try {
    const parsed = JSON.parse(json);
    return typeof parsed === "object" && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}

function isConfigRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function mergeConfigObjects(baseConfig: Record<string, unknown>, overrideConfig: Record<string, unknown>): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...baseConfig };

  Object.entries(overrideConfig).forEach(([key, value]) => {
    if (isConfigRecord(value) && isConfigRecord(merged[key])) {
      merged[key] = mergeConfigObjects(merged[key] as Record<string, unknown>, value);
      return;
    }
    merged[key] = value;
  });

  return merged;
}

function deepEqualConfigValue(left: unknown, right: unknown): boolean {
  if (left === right) {
    return true;
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((item, index) => deepEqualConfigValue(item, right[index]));
  }
  if (isConfigRecord(left) && isConfigRecord(right)) {
    const leftKeys = Object.keys(left);
    const rightKeys = Object.keys(right);
    return leftKeys.length === rightKeys.length && leftKeys.every(key => deepEqualConfigValue(left[key], right[key]));
  }
  return false;
}

function diffConfigObjects(baseConfig: Record<string, unknown>, nextConfig: Record<string, unknown>): Record<string, unknown> {
  const diff: Record<string, unknown> = {};

  Object.entries(nextConfig).forEach(([key, value]) => {
    if (deepEqualConfigValue(baseConfig[key], value)) {
      return;
    }
    diff[key] = value;
  });

  return diff;
}

/** 常见宽高比 */
const COMMON_ASPECT_RATIOS = ["1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2", "21:9"];

/** 常见分辨率档位 */
const COMMON_TIERS = ["1K", "2K", "3K", "4K"];

const OPENAI_REASONING_PLATFORMS = new Set([
  "openai_compatible",
  "openai",
  "deepseek",
  "zhipu",
  "moonshot",
  "volcengine",
  "siliconflow",
]);

function normalizePlatform(platform: string | null | undefined): string {
  return (platform || "").toLowerCase();
}

function isOpenAiReasoningPlatform(platform: string | null | undefined): boolean {
  return OPENAI_REASONING_PLATFORMS.has(normalizePlatform(platform));
}

function isAnthropicReasoningPlatform(platform: string | null | undefined): boolean {
  return normalizePlatform(platform) === "anthropic";
}

function isDashScopeReasoningPlatform(platform: string | null | undefined): boolean {
  return normalizePlatform(platform) === "dashscope";
}

const REASONING_CONFIG_KEYS = [
  "includeReasoning",
  "include_reasoning",
  "reasoningEffort",
  "reasoning_effort",
  "thinkingBudget",
  "thinking_budget",
  "thinking",
];

function supportsReasoningConfig(platform: string | null | undefined): boolean {
  return (
    isOpenAiReasoningPlatform(platform) ||
    isAnthropicReasoningPlatform(platform) ||
    isDashScopeReasoningPlatform(platform)
  );
}

function stripReasoningConfig(configJson: string | undefined): string {
  const next = { ...parseConfigJson(configJson) };
  REASONING_CONFIG_KEYS.forEach((key) => {
    delete next[key];
  });
  return Object.keys(next).length > 0 ? JSON.stringify(next) : "";
}

function getPositiveNumberValue(value: number | undefined): number | "" {
  return value !== undefined && value > 0 ? value : "";
}

function getConfigBooleanValue(value: unknown): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return value === "true";
  return false;
}

function getConfigNumberValue(value: unknown): number | "" {
  if (typeof value === "number" && !Number.isNaN(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? "" : parsed;
  }
  return "";
}

function getOptionalConfigNumber(value: unknown): number | undefined {
  const numericValue = getConfigNumberValue(value);
  return typeof numericValue === "number" ? numericValue : undefined;
}

function getConfigStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0) : [];
}

interface CapabilityChipDef {
  label: string;
  tone: "positive" | "muted" | "info";
}

interface GenerationCapabilityView {
  chips: CapabilityChipDef[];
  summary: string;
}

function findMatchingPreset(model: AiModel, platform: string | null | undefined, presets: ModelPreset[]): ModelPreset | null {
  return presets.find(preset => {
    if (preset.code !== model.code || preset.modelType !== model.modelType) {
      return false;
    }
    return !platform || preset.platform === platform;
  }) || null;
}

function getCapabilityChipClassName(tone: CapabilityChipDef["tone"]): string {
  switch (tone) {
    case "positive":
      return "bg-emerald-500/10 text-emerald-500 border-emerald-500/20";
    case "info":
      return "bg-sky-500/10 text-sky-500 border-sky-500/20";
    default:
      return "bg-muted/60 text-muted-foreground border-border/40";
  }
}

function buildImageCapabilityView(config: Record<string, unknown>): GenerationCapabilityView {
  const supportsReferenceImages = getConfigBooleanValue(config.supportReferenceImages);
  const maxReferenceImages = getOptionalConfigNumber(config.maxReferenceImages);
  const aspectRatioCount = getConfigStringArray(config.supportedAspectRatios).length;
  const supportedSizes = isConfigRecord(config.supportedSizes) ? Object.keys(config.supportedSizes).length : 0;

  const chips: CapabilityChipDef[] = [supportsReferenceImages
    ? { label: maxReferenceImages && maxReferenceImages > 0 ? `参考图 ≤${maxReferenceImages}` : "支持参考图", tone: "positive" }
    : { label: "仅文生图", tone: "muted" }];

  if (aspectRatioCount > 0) {
    chips.push({ label: `${aspectRatioCount} 种比例`, tone: "info" });
  }
  if (supportedSizes > 0) {
    chips.push({ label: `${supportedSizes} 个尺寸档`, tone: "info" });
  }

  return {
    chips,
    summary: supportsReferenceImages
      ? `支持参考图输入${maxReferenceImages && maxReferenceImages > 0 ? `，最多 ${maxReferenceImages} 张` : ""}`
      : "不支持参考图输入，当前模型只适合文生图。",
  };
}

function buildVideoCapabilityView(config: Record<string, unknown>): GenerationCapabilityView {
  const supportsFirstFrame = getConfigBooleanValue(config.supportFirstFrame);
  const supportsLastFrame = getConfigBooleanValue(config.supportLastFrame);
  const supportsReferenceImages = getConfigBooleanValue(config.supportReferenceImages);
  const supportsReferenceVideos = getConfigBooleanValue(config.supportReferenceVideos);
  const supportsReferenceAudios = getConfigBooleanValue(config.supportReferenceAudios);
  const minImageInputs = getOptionalConfigNumber(config.minImageInputs);
  const maxImageInputs = getOptionalConfigNumber(config.maxImageInputs);
  const maxReferenceImages = getOptionalConfigNumber(config.maxReferenceImages);
  const maxReferenceVideos = getOptionalConfigNumber(config.maxReferenceVideos);
  const maxReferenceAudios = getOptionalConfigNumber(config.maxReferenceAudios);

  const chips: CapabilityChipDef[] = [
    { label: supportsFirstFrame ? "首帧" : "无首帧", tone: supportsFirstFrame ? "positive" : "muted" },
    { label: supportsLastFrame ? "尾帧" : "无尾帧", tone: supportsLastFrame ? "positive" : "muted" },
    {
      label: supportsReferenceImages
        ? maxReferenceImages && maxReferenceImages > 0 ? `参考图 ≤${maxReferenceImages}` : "参考图"
        : "无参考图",
      tone: supportsReferenceImages ? "positive" : "muted",
    },
    {
      label: supportsReferenceVideos
        ? maxReferenceVideos && maxReferenceVideos > 0 ? `参考视频 ≤${maxReferenceVideos}` : "参考视频"
        : "无参考视频",
      tone: supportsReferenceVideos ? "positive" : "muted",
    },
    {
      label: supportsReferenceAudios
        ? maxReferenceAudios && maxReferenceAudios > 0 ? `参考音频 ≤${maxReferenceAudios}` : "参考音频"
        : "无参考音频",
      tone: supportsReferenceAudios ? "positive" : "muted",
    },
  ];

  if (minImageInputs !== undefined || maxImageInputs !== undefined) {
    const imageInputLabel = minImageInputs !== undefined && maxImageInputs !== undefined
      ? `图输 ${minImageInputs}-${maxImageInputs} 张`
      : minImageInputs !== undefined
        ? `图输 ≥${minImageInputs} 张`
        : `图输 ≤${maxImageInputs} 张`;
    chips.push({ label: imageInputLabel, tone: "info" });
  }

  const summaryParts = [
    supportsFirstFrame ? "支持首帧" : "不支持首帧",
    supportsLastFrame ? "支持尾帧" : "不支持尾帧",
    supportsReferenceImages ? "支持参考图" : "不支持参考图",
    supportsReferenceVideos ? "支持参考视频" : "不支持参考视频",
    supportsReferenceAudios ? "支持参考音频" : "不支持参考音频",
  ];

  return {
    chips,
    summary: summaryParts.join("，") + "。",
  };
}

function buildGenerationCapabilityView(model: AiModel, config: Record<string, unknown>): GenerationCapabilityView | null {
  if (model.modelType === 2) {
    return buildImageCapabilityView(config);
  }
  if (model.modelType === 3) {
    return buildVideoCapabilityView(config);
  }
  return null;
}

function getChatSamplingFields(platform: string | null | undefined): ConfigFieldDef[] {
  const normalized = normalizePlatform(platform);
  const fields: ConfigFieldDef[] = [
    { key: "temperature", label: "Temperature", type: "range", min: 0, max: 2, step: 0.1, defaultValue: 0.7, hint: "控制输出随机性，值越高越随机" },
  ];

  if (normalized === "vertex_ai" || normalized === "vertexai" || normalized === "gemini") {
    return fields;
  }

  fields.push({ key: "topP", label: "Top P", type: "range", min: 0, max: 1, step: 0.05, defaultValue: 1, hint: "核心采样概率阈值" });

  if (normalized === "ollama") {
    return fields;
  }

  fields.push({ key: "maxTokens", label: "Max Tokens", type: "number", min: 1, max: 1000000, step: 1, placeholder: "例如：4096", hint: "单次请求最大输出 token 数" });
  return fields;
}

// ---------- supportedSizes 编辑器 ----------

type SizesMap = Record<string, Record<string, string>>;

function SupportedSizesEditor({
  value,
  onChange,
}: {
  value: SizesMap | undefined;
  onChange: (v: SizesMap | undefined) => void;
}) {
  const [collapsedTiers, setCollapsedTiers] = useState<Set<string>>(new Set());
  const [newTierName, setNewTierName] = useState("");

  const sizes = value || {};
  const tierNames = Object.keys(sizes);

  const toggleCollapse = (tier: string) => {
    setCollapsedTiers(prev => {
      const next = new Set(prev);
      if (next.has(tier)) {
        next.delete(tier);
      } else {
        next.add(tier);
      }
      return next;
    });
  };

  const addTier = (tierName: string) => {
    const name = tierName.trim();
    if (!name || sizes[name]) return;
    const next = { ...sizes, [name]: {} };
    onChange(next);
    setNewTierName("");
  };

  const removeTier = (tier: string) => {
    const next = { ...sizes };
    delete next[tier];
    onChange(Object.keys(next).length > 0 ? next : undefined);
  };

  const addRatio = (tier: string, ratio: string) => {
    if (!ratio.trim()) return;
    const next = { ...sizes, [tier]: { ...sizes[tier], [ratio.trim()]: "" } };
    onChange(next);
  };

  const updateResolution = (tier: string, ratio: string, resolution: string) => {
    const next = { ...sizes, [tier]: { ...sizes[tier], [ratio]: resolution } };
    onChange(next);
  };

  const removeRatio = (tier: string, ratio: string) => {
    const tierData = { ...sizes[tier] };
    delete tierData[ratio];
    const next = { ...sizes, [tier]: tierData };
    onChange(next);
  };

  const fillCommonRatios = (tier: string) => {
    const tierData = { ...sizes[tier] };
    COMMON_ASPECT_RATIOS.forEach(r => {
      if (!(r in tierData)) tierData[r] = "";
    });
    onChange({ ...sizes, [tier]: tierData });
  };

  const availableTierChips = COMMON_TIERS.filter(t => !sizes[t]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-muted-foreground">支持的尺寸 (supportedSizes)</label>
      </div>

      {tierNames.map(tier => {
        const isCollapsed = collapsedTiers.has(tier);
        const ratios = sizes[tier] || {};
        const ratioEntries = Object.entries(ratios);

        return (
          <div key={tier} className="rounded-lg border border-border/30 bg-background overflow-hidden">
            <div className="flex items-center justify-between px-3 py-1.5 bg-muted/30 border-b border-border/20">
              <button
                type="button"
                onClick={() => toggleCollapse(tier)}
                className="flex items-center gap-1.5 text-[11px] font-medium text-foreground/80 hover:text-foreground transition-colors"
              >
                <ChevronRight className={cn("h-3 w-3 transition-transform duration-200", !isCollapsed && "rotate-90")} />
                {tier}
                <span className="text-[10px] text-muted-foreground font-normal">({ratioEntries.length} 项)</span>
              </button>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => fillCommonRatios(tier)}
                  className="text-[9px] text-muted-foreground hover:text-primary transition-colors px-1"
                  title="填充常见比例"
                >
                  +常见比例
                </button>
                <button
                  type="button"
                  onClick={() => removeTier(tier)}
                  className="text-[9px] text-muted-foreground hover:text-destructive transition-colors"
                  title="删除档位"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            </div>

            {!isCollapsed && (
              <div className="p-2 space-y-1">
                {ratioEntries.map(([ratio, resolution]) => (
                  <div key={ratio} className="flex items-center gap-1.5">
                    <span className="text-[10px] text-muted-foreground w-10 shrink-0 text-right font-mono">{ratio}</span>
                    <Input
                      placeholder="例如：1024x1024"
                      value={resolution}
                      onChange={e => updateResolution(tier, ratio, e.target.value)}
                      className="text-[10px] font-mono h-6 flex-1"
                    />
                    <button
                      type="button"
                      onClick={() => removeRatio(tier, ratio)}
                      className="text-[9px] text-muted-foreground hover:text-destructive transition-colors shrink-0"
                    >
                      ✕
                    </button>
                  </div>
                ))}

                {(() => {
                  const existingRatios = new Set(Object.keys(ratios));
                  const available = COMMON_ASPECT_RATIOS.filter(r => !existingRatios.has(r));
                  if (available.length === 0) return null;
                  return (
                    <div className="flex flex-wrap gap-1 pt-1 border-t border-border/10 mt-1">
                      {available.map(r => (
                        <button
                          key={r}
                          type="button"
                          onClick={() => addRatio(tier, r)}
                          className="text-[9px] px-1.5 py-0.5 rounded border border-dashed border-border/40 text-muted-foreground hover:border-primary/40 hover:text-primary transition-colors"
                        >
                          +{r}
                        </button>
                      ))}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        );
      })}

      <div className="flex items-center gap-1.5">
        {availableTierChips.map(t => (
          <button
            key={t}
            type="button"
            onClick={() => addTier(t)}
            className="text-[10px] px-2 py-0.5 rounded-md border border-dashed border-border/40 text-muted-foreground hover:border-primary/40 hover:text-primary transition-colors"
          >
            +{t}
          </button>
        ))}
        <div className="flex items-center gap-1 ml-auto">
          <Input
            placeholder="自定义档位"
            value={newTierName}
            onChange={e => setNewTierName(e.target.value)}
            onKeyDown={e => e.key === "Enter" && addTier(newTierName)}
            className="text-[10px] h-5 w-20 font-mono"
          />
          <button
            type="button"
            onClick={() => addTier(newTierName)}
            disabled={!newTierName.trim()}
            className="text-[10px] px-1.5 py-0.5 rounded text-primary hover:bg-primary/10 transition-colors disabled:opacity-40"
          >
            添加
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------- supportedAspectRatios 编辑器 ----------

function AspectRatiosEditor({
  value,
  onChange,
  label,
  hint,
  presetOptions,
}: {
  value: string[] | undefined;
  onChange: (v: string[] | undefined) => void;
  label?: string;
  hint?: string;
  presetOptions?: string[];
}) {
  const [customRatio, setCustomRatio] = useState("");
  const selected = new Set(value || []);
  const commonOptions = presetOptions || COMMON_ASPECT_RATIOS;

  const toggle = (ratio: string) => {
    const next = new Set(selected);
    if (next.has(ratio)) {
      next.delete(ratio);
    } else {
      next.add(ratio);
    }
    const arr = Array.from(next);
    onChange(arr.length > 0 ? arr : undefined);
  };

  const addCustom = () => {
    const r = customRatio.trim();
    if (!r || selected.has(r)) return;
    onChange([...Array.from(selected), r]);
    setCustomRatio("");
  };

  return (
    <div className="space-y-2.5">
      <label className="text-[11px] text-muted-foreground">{label || "支持的比例 (supportedAspectRatios)"}</label>
      {hint && <p className="text-[10px] text-muted-foreground/70 -mt-1">{hint}</p>}
      <div className="flex flex-wrap gap-1.5">
        {commonOptions.map(ratio => (
          <button
            key={ratio}
            type="button"
            onClick={() => toggle(ratio)}
            className={cn(
              "inline-flex items-center gap-1 text-[10px] px-2.5 py-1 rounded-full border transition-all duration-150",
              selected.has(ratio)
                ? "border-primary/40 bg-primary/15 text-primary font-medium shadow-sm shadow-primary/5"
                : "border-dashed border-border/50 text-muted-foreground/60 hover:border-primary/30 hover:text-foreground"
            )}
          >
            {selected.has(ratio) && <Check className="h-2.5 w-2.5" />}
            {ratio}
          </button>
        ))}
        {Array.from(selected)
          .filter(r => !commonOptions.includes(r))
          .map(ratio => (
            <button
              key={ratio}
              type="button"
              onClick={() => toggle(ratio)}
              className="inline-flex items-center gap-1 text-[10px] px-2.5 py-1 rounded-full border border-primary/40 bg-primary/15 text-primary font-medium shadow-sm shadow-primary/5 transition-all duration-150"
            >
              <Check className="h-2.5 w-2.5" />
              {ratio}
              <span className="text-primary/50 ml-0.5">✕</span>
            </button>
          ))}
      </div>
      <div className="flex items-center gap-2 pt-1 border-t border-border/20">
        <Input
          placeholder="自定义值，如：16:9"
          value={customRatio}
          onChange={e => setCustomRatio(e.target.value)}
          onKeyDown={e => e.key === "Enter" && addCustom()}
          className="text-[10px] h-6 w-50 font-mono"
        />
        <button
          type="button"
          onClick={addCustom}
          disabled={!customRatio.trim()}
          className="text-[10px] px-2 py-0.5 rounded-md text-primary hover:bg-primary/10 transition-colors disabled:opacity-30"
        >
          + 添加
        </button>
      </div>
    </div>
  );
}

// ---------- ModelConfigForm 主组件 ----------

interface ConfigFieldDef {
  key: string;
  label: string;
  type: "number" | "range" | "supported-sizes" | "aspect-ratios";
  min?: number;
  max?: number;
  step?: number;
  defaultValue?: number;
  placeholder?: string;
  hint?: string;
  presetOptions?: string[];
}

function getConfigFieldsByModelType(modelType: number, platform?: string | null): ConfigFieldDef[] {
  switch (modelType) {
    case 1:
      return getChatSamplingFields(platform);
    case 2:
      return [
        { key: "defaultWidth", label: "默认宽度", type: "number", min: 256, max: 8192, step: 64, placeholder: "例如：1024" },
        { key: "defaultHeight", label: "默认高度", type: "number", min: 256, max: 8192, step: 64, placeholder: "例如：1024" },
        { key: "minPixels", label: "最小像素数", type: "number", min: 0, step: 1, placeholder: "例如：921600", hint: "生成图像的最小总像素数限制" },
        { key: "maxPixels", label: "最大像素数", type: "number", min: 0, step: 1, placeholder: "例如：16777216", hint: "生成图像的最大总像素数限制" },
        { key: "supportedSizes", label: "支持的尺寸", type: "supported-sizes" },
        { key: "supportedAspectRatios", label: "支持的比例", type: "aspect-ratios" },
      ];
    case 3:
      return [
        { key: "supportedResolutions", label: "支持的分辨率", type: "aspect-ratios", hint: "如 480p, 720p, 1080p", presetOptions: ["480p", "720p", "1080p", "2K", "4K"] },
        { key: "supportedAspectRatios", label: "支持的宽高比", type: "aspect-ratios", hint: "如 16:9, 9:16, 1:1" },
        { key: "minDuration", label: "最短时长（秒）", type: "number", min: 1, max: 60, step: 1, placeholder: "例如：4" },
        { key: "maxDuration", label: "最长时长（秒）", type: "number", min: 1, max: 60, step: 1, placeholder: "例如：15" },
        { key: "defaultDuration", label: "默认时长（秒）", type: "number", min: 1, max: 60, step: 1, placeholder: "例如：5" },
      ];
    default:
      return [];
  }
}

function ToggleSettingCard({
  checked,
  title,
  description,
  onToggle,
}: {
  checked: boolean;
  title: string;
  description: string;
  onToggle: () => void;
}) {
  return (
    <div className="rounded-lg border border-border/30 bg-background/70 px-3 py-2.5">
      <div className="flex items-start gap-3">
        <button
          type="button"
          onClick={onToggle}
          className={cn(
            "relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors duration-200",
            checked ? "bg-primary" : "bg-muted-foreground/30"
          )}
        >
          <span
            className={cn(
              "absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200",
              checked && "translate-x-4"
            )}
          />
        </button>
        <div className="min-w-0">
          <Label className="cursor-pointer text-xs text-muted-foreground" onClick={onToggle}>
            {title}
          </Label>
          <p className="mt-1 text-[10px] leading-5 text-muted-foreground/70">{description}</p>
        </div>
      </div>
    </div>
  );
}

function CapabilityNumberField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  min,
  max,
  step = 1,
  disabled = false,
}: {
  label: string;
  value: number | "";
  onChange: (value: string) => void;
  placeholder?: string;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <div className={cn("space-y-1.5", disabled && "opacity-55")}>
      <Label className="text-[11px] text-muted-foreground">{label}</Label>
      <Input
        type="number"
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        placeholder={placeholder}
        value={value}
        onChange={e => onChange(e.target.value)}
        className="text-xs font-mono h-8"
      />
      {hint && <p className="text-[10px] text-muted-foreground/70">{hint}</p>}
    </div>
  );
}

function ModelConfigForm({
  modelType,
  platform,
  supportReasoning,
  configJson,
  baseConfig,
  onChange,
}: {
  modelType: number;
  platform?: string | null;
  supportReasoning: boolean;
  configJson: string | undefined;
  baseConfig?: Record<string, unknown>;
  onChange: (json: string) => void;
}) {
  const [showRawJson, setShowRawJson] = useState(false);
  const [rawJsonDraft, setRawJsonDraft] = useState("");
  const [rawJsonError, setRawJsonError] = useState(false);

  const effectiveBaseConfig = baseConfig || {};
  const hasBaseConfig = Object.keys(effectiveBaseConfig).length > 0;
  const configObj = mergeConfigObjects(effectiveBaseConfig, parseConfigJson(configJson));
  const normalizedPlatform = normalizePlatform(platform);
  const fields = getConfigFieldsByModelType(modelType, platform);
  const showReasoningConfig = modelType === 1 && supportReasoning;
  const includeReasoningMode = configObj.includeReasoning === true
    ? "true"
    : configObj.includeReasoning === false
      ? "false"
      : "auto";
  const reasoningEffortValue = typeof configObj.reasoningEffort === "string"
    ? configObj.reasoningEffort
    : "__unset__";

  const emitChange = (next: Record<string, unknown>) => {
    const cleaned = Object.fromEntries(
      Object.entries(next).filter(([, v]) => v !== "" && v !== undefined && v !== null)
    );
    const overrides = diffConfigObjects(effectiveBaseConfig, cleaned);
    onChange(Object.keys(overrides).length > 0 ? JSON.stringify(overrides) : "");
  };

  const updateSimpleField = (key: string, raw: string) => {
    const next = { ...configObj };
    if (raw === "" || raw === undefined) {
      delete next[key];
    } else {
      const num = parseFloat(raw);
      next[key] = isNaN(num) ? raw : num;
    }
    emitChange(next);
  };

  const updateComplexField = (key: string, value: unknown) => {
    const next = { ...configObj };
    if (value === undefined || value === null) {
      delete next[key];
    } else {
      next[key] = value;
    }
    emitChange(next);
  };

  const updateSelectField = (key: string, rawValue: string) => {
    const next = { ...configObj };
    if (rawValue === "auto" || rawValue === "__unset__") {
      delete next[key];
    } else if (rawValue === "true" || rawValue === "false") {
      next[key] = rawValue === "true";
    } else {
      next[key] = rawValue;
    }
    emitChange(next);
  };

  const handleToggleRawJson = () => {
    if (!showRawJson) {
      setRawJsonDraft(Object.keys(configObj).length > 0 ? JSON.stringify(configObj, null, 2) : "");
      setRawJsonError(false);
    }
    setShowRawJson(!showRawJson);
  };

  const handleRawJsonApply = () => {
    if (!rawJsonDraft.trim()) {
      onChange("");
      setShowRawJson(false);
      return;
    }
    try {
      const parsed = JSON.parse(rawJsonDraft);
      if (!isConfigRecord(parsed)) {
        setRawJsonError(true);
        return;
      }
      emitChange(parsed);
      setRawJsonError(false);
      setShowRawJson(false);
    } catch {
      setRawJsonError(true);
    }
  };

  const supportsImageReferenceInputs = getConfigBooleanValue(configObj.supportReferenceImages);
  const supportsFirstFrame = getConfigBooleanValue(configObj.supportFirstFrame);
  const supportsLastFrame = getConfigBooleanValue(configObj.supportLastFrame);
  const supportsVideoReferenceImages = getConfigBooleanValue(configObj.supportReferenceImages);
  const supportsReferenceVideos = getConfigBooleanValue(configObj.supportReferenceVideos);
  const supportsReferenceAudios = getConfigBooleanValue(configObj.supportReferenceAudios);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">模型参数配置</Label>
        <button
          type="button"
          onClick={handleToggleRawJson}
          className="text-[10px] text-muted-foreground hover:text-foreground transition-colors px-1.5 py-0.5 rounded hover:bg-muted/50"
        >
          {showRawJson ? "切换表单" : "编辑 JSON"}
        </button>
      </div>

      {hasBaseConfig && (
        <div className="rounded-lg border border-primary/15 bg-primary/5 px-3 py-2">
          <p className="text-[10px] leading-5 text-muted-foreground/80">
            当前表单展示的是预设默认值与当前模型覆盖项合并后的生效配置。保存时只会写入与预设不同的差异字段，避免把 preset 默认值全部固化到数据库。
          </p>
        </div>
      )}

      {showRawJson ? (
        <div className="space-y-2">
          <textarea
            value={rawJsonDraft}
            onChange={e => {
              setRawJsonDraft(e.target.value);
              setRawJsonError(false);
            }}
            placeholder='{"temperature": 0.7, "maxTokens": 4096}'
            className={cn(
              "w-full min-h-[120px] rounded-lg border bg-background px-3 py-2 text-xs font-mono resize-y",
              "focus:outline-none focus:ring-2 focus:ring-ring/30",
              rawJsonError && "border-destructive focus:ring-destructive/30"
            )}
          />
          {rawJsonError && (
            <p className="text-[10px] text-destructive">JSON 格式错误，请检查语法</p>
          )}
          <div className="flex justify-end gap-1.5">
            <button
              type="button"
              onClick={() => setShowRawJson(false)}
              className="text-[10px] px-2 py-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
            >
              取消
            </button>
            <button
              type="button"
              onClick={handleRawJsonApply}
              className="text-[10px] px-2 py-1 rounded bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
            >
              应用
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {fields.length > 0 ? (
            <>
              {(() => {
                const simpleFields = fields.filter(f => f.type === "number" || f.type === "range");
                if (simpleFields.length === 0) return null;
                return (
                  <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                    {simpleFields.map(field => {
                      const value = configObj[field.key];
                      const numValue = typeof value === "number" ? value : (typeof value === "string" ? parseFloat(value) : undefined);

                      if (field.type === "range") {
                        return (
                          <div key={field.key} className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <label className="text-[11px] text-muted-foreground">{field.label}</label>
                              <span className="text-[11px] font-mono text-foreground/80 tabular-nums min-w-[3ch] text-right">
                                {numValue !== undefined && !isNaN(numValue) ? numValue : "—"}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <input
                                type="range"
                                min={field.min ?? 0}
                                max={field.max ?? 1}
                                step={field.step ?? 0.1}
                                value={numValue !== undefined && !isNaN(numValue) ? numValue : field.defaultValue ?? field.min ?? 0}
                                onChange={e => updateSimpleField(field.key, e.target.value)}
                                className="flex-1 h-1.5 accent-primary cursor-pointer"
                              />
                              <button
                                type="button"
                                onClick={() => updateSimpleField(field.key, "")}
                                className="text-[9px] text-muted-foreground hover:text-destructive transition-colors shrink-0"
                                title="清除"
                              >
                                ✕
                              </button>
                            </div>
                            {field.hint && <p className="text-[10px] text-muted-foreground/70">{field.hint}</p>}
                          </div>
                        );
                      }

                      return (
                        <div key={field.key} className="space-y-1">
                          <label className="text-[11px] text-muted-foreground">{field.label}</label>
                          <Input
                            type="number"
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            placeholder={field.placeholder}
                            value={numValue !== undefined && !isNaN(numValue) ? numValue : ""}
                            onChange={e => updateSimpleField(field.key, e.target.value)}
                            className="text-xs font-mono h-8"
                          />
                          {field.hint && <p className="text-[10px] text-muted-foreground/70">{field.hint}</p>}
                        </div>
                      );
                    })}
                  </div>
                );
              })()}

              {fields.some(f => f.type === "supported-sizes") && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3">
                  <SupportedSizesEditor
                    value={configObj.supportedSizes as SizesMap | undefined}
                    onChange={v => updateComplexField("supportedSizes", v)}
                  />
                </div>
              )}

              {fields
                .filter(f => f.type === "aspect-ratios")
                .map(field => (
                  <div key={field.key} className="rounded-lg border border-border/40 bg-muted/20 p-3">
                    <AspectRatiosEditor
                      label={field.label}
                      hint={field.hint}
                      presetOptions={field.presetOptions}
                      value={configObj[field.key] as string[] | undefined}
                      onChange={v => updateComplexField(field.key, v)}
                    />
                  </div>
                ))}

              {modelType === 2 && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                  <div>
                    <Label className="text-[11px] text-muted-foreground">参考图能力</Label>
                    <p className="text-[10px] text-muted-foreground/70 mt-1">控制 generate_image 是否允许传 imageUrls，以及允许的参考图数量。</p>
                  </div>

                  <ToggleSettingCard
                    checked={supportsImageReferenceInputs}
                    title="支持参考图输入"
                    description="开启后，agent 才会把 imageUrls 传给当前图片模型。关闭时会要求只走文生图。"
                    onToggle={() => {
                      const nextEnabled = !supportsImageReferenceInputs;
                      emitChange({
                        ...configObj,
                        supportReferenceImages: nextEnabled,
                        minReferenceImages: nextEnabled ? configObj.minReferenceImages : 0,
                        maxReferenceImages: nextEnabled ? configObj.maxReferenceImages : 0,
                      });
                    }}
                  />

                  <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
                    <CapabilityNumberField
                      label="最少参考图数量"
                      value={getConfigNumberValue(configObj.minReferenceImages)}
                      onChange={value => updateSimpleField("minReferenceImages", value)}
                      min={0}
                      step={1}
                      disabled={!supportsImageReferenceInputs}
                      placeholder="例如：0"
                      hint="通常填 0；只有模型明确要求至少上传多张参考图时才需要设置。"
                    />
                    <CapabilityNumberField
                      label="最多参考图数量"
                      value={getConfigNumberValue(configObj.maxReferenceImages)}
                      onChange={value => updateSimpleField("maxReferenceImages", value)}
                      min={0}
                      step={1}
                      disabled={!supportsImageReferenceInputs}
                      placeholder="例如：3"
                      hint="限制单次 generate_image 可传入的 imageUrls 数量。"
                    />
                  </div>
                </div>
              )}

              {modelType === 3 && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                  <div>
                    <Label className="text-[11px] text-muted-foreground">多模态输入能力</Label>
                    <p className="text-[10px] text-muted-foreground/70 mt-1">控制 generate_video 是否允许首帧、尾帧、参考图、参考视频和参考音频，以及对应数量上限。</p>
                  </div>

                  <div className="grid gap-2.5 [grid-template-columns:repeat(auto-fit,minmax(240px,1fr))]">
                    <ToggleSettingCard
                      checked={supportsFirstFrame}
                      title="支持首帧图"
                      description="允许传 firstFrameImageUrl 来锁定开场画面。"
                      onToggle={() => updateComplexField("supportFirstFrame", !supportsFirstFrame)}
                    />
                    <ToggleSettingCard
                      checked={supportsLastFrame}
                      title="支持尾帧图"
                      description="允许传 lastFrameImageUrl 来约束结尾画面。"
                      onToggle={() => updateComplexField("supportLastFrame", !supportsLastFrame)}
                    />
                    <ToggleSettingCard
                      checked={supportsVideoReferenceImages}
                      title="支持参考图"
                      description="允许传 referenceImageUrls；适合角色、场景或多图参考。"
                      onToggle={() => {
                        const nextEnabled = !supportsVideoReferenceImages;
                        emitChange({
                          ...configObj,
                          supportReferenceImages: nextEnabled,
                          maxReferenceImages: nextEnabled ? configObj.maxReferenceImages : 0,
                        });
                      }}
                    />
                    <ToggleSettingCard
                      checked={supportsReferenceVideos}
                      title="支持参考视频"
                      description="允许传 referenceVideoUrls，用于动作或镜头风格参考。"
                      onToggle={() => {
                        const nextEnabled = !supportsReferenceVideos;
                        emitChange({
                          ...configObj,
                          supportReferenceVideos: nextEnabled,
                          maxReferenceVideos: nextEnabled ? configObj.maxReferenceVideos : 0,
                        });
                      }}
                    />
                    <ToggleSettingCard
                      checked={supportsReferenceAudios}
                      title="支持参考音频"
                      description="允许传 referenceAudioUrls，用于节奏或音频条件参考。"
                      onToggle={() => {
                        const nextEnabled = !supportsReferenceAudios;
                        emitChange({
                          ...configObj,
                          supportReferenceAudios: nextEnabled,
                          maxReferenceAudios: nextEnabled ? configObj.maxReferenceAudios : 0,
                        });
                      }}
                    />
                  </div>

                  <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
                    <CapabilityNumberField
                      label="最少图片输入数"
                      value={getConfigNumberValue(configObj.minImageInputs)}
                      onChange={value => updateSimpleField("minImageInputs", value)}
                      min={0}
                      step={1}
                      placeholder="例如：1"
                      hint="计数包含 firstFrameImageUrl、lastFrameImageUrl 和 referenceImageUrls。"
                    />
                    <CapabilityNumberField
                      label="最多图片输入数"
                      value={getConfigNumberValue(configObj.maxImageInputs)}
                      onChange={value => updateSimpleField("maxImageInputs", value)}
                      min={0}
                      step={1}
                      placeholder="例如：3"
                      hint="用于限制图片类输入总数，避免首尾帧与参考图一起超限。"
                    />
                    <CapabilityNumberField
                      label="最多参考图数量"
                      value={getConfigNumberValue(configObj.maxReferenceImages)}
                      onChange={value => updateSimpleField("maxReferenceImages", value)}
                      min={0}
                      step={1}
                      disabled={!supportsVideoReferenceImages}
                      placeholder="例如：3"
                      hint="referenceImageUrls 的单独上限。"
                    />
                    <CapabilityNumberField
                      label="最多参考视频数量"
                      value={getConfigNumberValue(configObj.maxReferenceVideos)}
                      onChange={value => updateSimpleField("maxReferenceVideos", value)}
                      min={0}
                      step={1}
                      disabled={!supportsReferenceVideos}
                      placeholder="例如：1"
                      hint="referenceVideoUrls 的单独上限。"
                    />
                    <CapabilityNumberField
                      label="最多参考音频数量"
                      value={getConfigNumberValue(configObj.maxReferenceAudios)}
                      onChange={value => updateSimpleField("maxReferenceAudios", value)}
                      min={0}
                      step={1}
                      disabled={!supportsReferenceAudios}
                      placeholder="例如：1"
                      hint="referenceAudioUrls 的单独上限。"
                    />
                  </div>
                </div>
              )}

              {showReasoningConfig && isOpenAiReasoningPlatform(normalizedPlatform) && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                  <div>
                    <Label className="text-[11px] text-muted-foreground">思考内容返回</Label>
                    <p className="text-[10px] text-muted-foreground/70 mt-1">自动模式下，只要开启“支持思考”或填写推理参数，后端会自动请求 reasoning 内容。</p>
                  </div>

                  <div className="space-y-1.5">
                    <Select
                      value={includeReasoningMode}
                      onValueChange={v => updateSelectField("includeReasoning", String(v))}
                      items={[
                        { value: "auto", label: "自动" },
                        { value: "true", label: "显式返回思考内容" },
                        { value: "false", label: "显式关闭思考内容返回" },
                      ]}
                    >
                      <SelectTrigger className="w-full text-sm">
                        <SelectValue placeholder="选择思考内容返回策略" />
                      </SelectTrigger>
                      <SelectContent className="text-sm">
                        <SelectGroup>
                          <SelectItem value="auto" className="text-sm">自动</SelectItem>
                          <SelectItem value="true" className="text-sm">显式返回思考内容</SelectItem>
                          <SelectItem value="false" className="text-sm">显式关闭思考内容返回</SelectItem>
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(240px,1fr))]">
                    <div className="space-y-1.5">
                      <Label className="text-[11px] text-muted-foreground">Reasoning Effort</Label>
                      <Select
                        value={reasoningEffortValue}
                        onValueChange={v => updateSelectField("reasoningEffort", String(v))}
                        items={[
                          { value: "__unset__", label: "自动" },
                          { value: "low", label: "Low" },
                          { value: "medium", label: "Medium" },
                          { value: "high", label: "High" },
                        ]}
                      >
                        <SelectTrigger className="w-full text-sm">
                          <SelectValue placeholder="选择推理强度" />
                        </SelectTrigger>
                        <SelectContent className="text-sm">
                          <SelectGroup>
                            <SelectItem value="__unset__" className="text-sm">自动</SelectItem>
                            <SelectItem value="low" className="text-sm">Low</SelectItem>
                            <SelectItem value="medium" className="text-sm">Medium</SelectItem>
                            <SelectItem value="high" className="text-sm">High</SelectItem>
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-1.5">
                      <Label className="text-[11px] text-muted-foreground">Thinking Budget</Label>
                      <Input
                        type="number"
                        min={0}
                        step={1}
                        placeholder="例如：2048"
                        value={getPositiveNumberValue(typeof configObj.thinkingBudget === "number" ? configObj.thinkingBudget : undefined)}
                        onChange={e => updateSimpleField("thinkingBudget", e.target.value)}
                        className="text-xs font-mono h-8"
                      />
                      <p className="text-[10px] text-muted-foreground/70">面向支持 reasoning budget 的 OpenAI 兼容渠道。</p>
                    </div>
                  </div>
                </div>
              )}

              {showReasoningConfig && isAnthropicReasoningPlatform(normalizedPlatform) && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">Thinking Budget</Label>
                  <Input
                    type="number"
                    min={0}
                    step={1}
                    placeholder="例如：1024"
                    value={getPositiveNumberValue(typeof configObj.thinkingBudget === "number" ? configObj.thinkingBudget : undefined)}
                    onChange={e => updateSimpleField("thinkingBudget", e.target.value)}
                    className="text-xs font-mono h-8"
                  />
                  <p className="text-[10px] text-muted-foreground/70">Claude 开启“支持思考”后，未填写时后端默认使用 1024。更细粒度的 thinking 结构仍可通过 JSON 编辑器覆盖。</p>
                </div>
              )}

              {showReasoningConfig && isDashScopeReasoningPlatform(normalizedPlatform) && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3">
                  <p className="text-[10px] text-muted-foreground/70">DashScope 的思考模式由下方“支持思考”能力开关控制；如果后续需要 provider 专属参数，仍可使用 JSON 编辑器补充。</p>
                </div>
              )}
            </>
          ) : (
            <p className="text-[10px] text-muted-foreground/60 italic">当前模型类型无预定义配置项，可点击「编辑 JSON」手动配置</p>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================
// 从远程 API 获取模型列表 Dialog
// ============================================================

interface FetchRemoteModelsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  apiConfig: ApiConfig;
  existingModelCodes: Set<string>;
  onAdded: () => void;
}

function FetchRemoteModelsDialog({
  open,
  onOpenChange,
  apiConfig,
  existingModelCodes,
  onAdded,
}: FetchRemoteModelsDialogProps) {
  const [loading, setLoading] = useState(false);
  const [remoteModels, setRemoteModels] = useState<RemoteModel[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [adding, setAdding] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [modelType, setModelType] = useState<number>(1);

  const hasUnknownModelTypes = remoteModels.some(model => model.modelType == null);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const models = await apiConfigApi.remoteModels(apiConfig.id);
      setRemoteModels(models);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "获取模型列表失败";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [apiConfig.id]);

  useEffect(() => {
    if (open) {
      setRemoteModels([]);
      setError(null);
      setSelectedIds(new Set());
      setSearchQuery("");
      fetchModels();
    }
  }, [fetchModels, open]);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    const filtered = filteredModels;
    const allSelected = filtered.every(m => selectedIds.has(m.id));
    if (allSelected) {
      const next = new Set(selectedIds);
      filtered.forEach(m => next.delete(m.id));
      setSelectedIds(next);
    } else {
      const next = new Set(selectedIds);
      filtered.forEach(m => {
        if (!existingModelCodes.has(m.id)) {
          next.add(m.id);
        }
      });
      setSelectedIds(next);
    }
  };

  const handleAdd = async () => {
    if (selectedIds.size === 0) return;
    setAdding(true);
    try {
      for (const modelId of selectedIds) {
        const remoteModel = remoteModels.find(model => model.id === modelId);
        await aiModelApi.create({
          name: modelId,
          code: modelId,
          modelType: remoteModel?.modelType ?? modelType,
          apiConfigId: apiConfig.id,
        });
      }
      onAdded();
      onOpenChange(false);
    } catch (err) {
      console.error("添加模型失败:", err);
    } finally {
      setAdding(false);
    }
  };

  const filteredModels = remoteModels.filter(m => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return m.id.toLowerCase().includes(q) || (m.ownedBy && m.ownedBy.toLowerCase().includes(q));
  });

  const selectableCount = filteredModels.filter(m => !existingModelCodes.has(m.id)).length;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg flex flex-col max-h-[85vh]">
        <DialogHeader className="shrink-0">
          <DialogTitle>获取可用模型</DialogTitle>
          <DialogDescription>
            从 {apiConfig.name} 获取远程可用模型列表，选择后点击添加
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3 min-h-0">
          {/* 搜索框 + 模型类型选择 */}
          <div className="flex items-center gap-2 shrink-0">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                placeholder="搜索模型..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="text-sm pl-8 h-8"
              />
            </div>
            {hasUnknownModelTypes ? (
              <Select
                value={modelType}
                onValueChange={v => setModelType(v as number)}
                items={MODEL_TYPE_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
              >
                <SelectTrigger className="w-[148px] text-xs h-8">
                  <SelectValue placeholder="默认模型类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    {MODEL_TYPE_OPTIONS.map(opt => (
                      <SelectItem key={opt.value} value={opt.value} className="text-xs">
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            ) : (
              <div className="inline-flex h-8 items-center rounded-lg border border-emerald-500/20 bg-emerald-500/8 px-2.5 text-[10px] text-emerald-600 shrink-0">
                已自动识别模型类型
              </div>
            )}
          </div>

          {hasUnknownModelTypes && (
            <p className="-mt-1 text-[10px] text-muted-foreground">
              已识别类型的模型会按返回值自动导入；仅未识别类型的模型才会使用右侧默认类型。
            </p>
          )}

          {/* 模型列表 */}
          {loading ? (
            <div className="flex items-center justify-center py-10">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">正在获取模型列表...</span>
            </div>
          ) : error ? (
            <div className="text-center py-8 space-y-2">
              <p className="text-sm text-destructive">{error}</p>
              <Button variant="outline" size="sm" onClick={fetchModels}>
                重试
              </Button>
            </div>
          ) : remoteModels.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-sm text-muted-foreground">未获取到模型</p>
            </div>
          ) : (
            <>
              {/* 全选 + 计数 */}
              <div className="flex items-center justify-between px-1 shrink-0">
                <button
                  type="button"
                  onClick={toggleSelectAll}
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {filteredModels.filter(m => !existingModelCodes.has(m.id)).every(m => selectedIds.has(m.id)) && selectableCount > 0
                    ? "取消全选"
                    : "全选可添加"}
                </button>
                <span className="text-[10px] text-muted-foreground">
                  共 {filteredModels.length} 个模型，已选 {selectedIds.size} 个
                </span>
              </div>

              <div className="overflow-y-auto min-h-0 max-h-[400px] -mx-1 px-1 space-y-0.5">
                {filteredModels.map(model => {
                  const alreadyExists = existingModelCodes.has(model.id);
                  const isSelected = selectedIds.has(model.id);

                  return (
                    <button
                      key={model.id}
                      type="button"
                      onClick={() => !alreadyExists && toggleSelect(model.id)}
                      disabled={alreadyExists}
                      className={cn(
                        "flex items-center gap-3 w-full px-3 py-2 rounded-lg text-left transition-all duration-150",
                        alreadyExists
                          ? "opacity-50 cursor-not-allowed"
                          : isSelected
                            ? "bg-primary/10 border border-primary/30"
                            : "hover:bg-muted/50 border border-transparent"
                      )}
                    >
                      {/* 选择框 */}
                      <div className={cn(
                        "w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-colors",
                        alreadyExists
                          ? "bg-muted border-border"
                          : isSelected
                            ? "bg-primary border-primary"
                            : "border-border/60"
                      )}>
                        {(isSelected || alreadyExists) && (
                          <Check className={cn("h-3 w-3", alreadyExists ? "text-muted-foreground" : "text-primary-foreground")} />
                        )}
                      </div>

                      {/* 模型信息 */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <p className="text-sm font-mono truncate">{model.id}</p>
                          {model.modelType != null && (
                            <span className={cn(
                              "shrink-0 rounded px-1.5 py-0.5 text-[10px]",
                              model.modelType === 2
                                ? "bg-sky-500/10 text-sky-500"
                                : model.modelType === 3
                                  ? "bg-emerald-500/10 text-emerald-500"
                                  : "bg-muted text-muted-foreground"
                            )}>
                              {MODEL_TYPE_LABELS[model.modelType] || `类型${model.modelType}`}
                            </span>
                          )}
                        </div>
                        {model.ownedBy && (
                          <p className="text-[10px] text-muted-foreground">{model.ownedBy}</p>
                        )}
                      </div>

                      {/* 已添加标记 */}
                      {alreadyExists && (
                        <span className="text-[10px] text-muted-foreground shrink-0 px-1.5 py-0.5 rounded bg-muted">
                          已添加
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </div>

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button
            size="sm"
            onClick={handleAdd}
            disabled={adding || selectedIds.size === 0}
          >
            {adding && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            添加 {selectedIds.size > 0 ? `(${selectedIds.size})` : ""}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// AI 模型 Dialog
// ============================================================

interface AiModelDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingModel: AiModel | null;
  apiConfigs: ApiConfig[];
  defaultApiConfigId?: number;
  onSaved: () => void;
}

function AiModelDialog({ open, onOpenChange, editingModel, apiConfigs, defaultApiConfigId, onSaved }: AiModelDialogProps) {
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<AiModelCreateReq & { id?: number; status?: number }>({
    name: "",
    code: "",
    modelType: 1,
    maxConcurrency: 5,
    supportVision: false,
    supportReasoning: false,
    contextWindow: 0,
  });
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [selectedPresetCode, setSelectedPresetCode] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      aiModelApi.presets().then(setPresets).catch(console.error);
      if (editingModel) {
        setForm({
          id: editingModel.id,
          name: editingModel.name,
          code: editingModel.code,
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
        setSelectedPresetCode(null);
      } else {
        setForm({
          name: "",
          code: "",
          modelType: 1,
          maxConcurrency: 5,
          defaultModel: false,
          supportVision: false,
          supportReasoning: false,
          contextWindow: 0,
          apiConfigId: defaultApiConfigId,
        });
        setSelectedPresetCode(null);
      }
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
  const selectedPreset = selectedPresetCode
    ? presets.find(p => p.code === selectedPresetCode) || null
    : null;
  const matchedPreset = selectedPreset || presets.find(p => {
    if (p.code !== form.code || p.modelType !== form.modelType) {
      return false;
    }
    return !selectedPlatform || p.platform === selectedPlatform;
  }) || null;
  const visiblePresets = !selectedPlatform
    ? presets
    : presets.filter(p => p.platform === selectedPlatform);

  const handleSave = async () => {
    if (!form.name.trim() || !form.code.trim() || !form.apiConfigId) return;
    setSaving(true);
    try {
      const normalizedConfig = form.modelType === 1 && form.supportReasoning && supportsReasoningConfig(selectedPlatform)
        ? form.config
        : stripReasoningConfig(form.config);

      if (editingModel) {
        const updateReq: AiModelUpdateReq = {
          id: editingModel.id,
          name: form.name,
          code: form.code,
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
      <DialogContent className="sm:max-w-lg flex flex-col max-h-[85vh]">
        <DialogHeader className="shrink-0">
          <DialogTitle>{editingModel ? "编辑 AI 模型" : "新建 AI 模型"}</DialogTitle>
          <DialogDescription>
            配置可用的 AI 模型参数
          </DialogDescription>
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
                      setSelectedPresetCode(preset.code);
                      setForm(prev => ({
                        ...prev,
                        name: preset.name,
                        code: preset.code,
                        modelType: preset.modelType,
                        description: preset.description,
                        config: "",
                      }));
                    }}
                    className={cn(
                      "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs transition-all duration-200",
                      "border",
                      selectedPresetCode === preset.code
                        ? "border-primary/50 bg-primary/10 text-primary font-medium"
                        : "border-border/30 text-muted-foreground hover:border-primary/30 hover:text-foreground hover:bg-white/5"
                    )}
                  >
                    {selectedPresetCode === preset.code && <Check className="h-3 w-3" />}
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>
          )}

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
            <p className="text-[10px] text-muted-foreground">对应 API 中实际使用的 model 名称</p>
          </div>

          {/* 模型类型 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">模型类型</Label>
            <Select
              value={form.modelType}
              onValueChange={v => {
                const nextType = v as number;
                setForm(prev => {
                  const next = { ...prev, modelType: nextType };
                  if (nextType !== 1) {
                    next.supportVision = false;
                    next.supportReasoning = false;
                    next.contextWindow = 0;
                    next.config = stripReasoningConfig(prev.config);
                  }
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

          {/* 关联 API 配置 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">关联 API 配置 <span className="text-destructive">*</span></Label>
            <Select
              value={form.apiConfigId}
              onValueChange={v => {
                const nextApiConfigId = v as number;
                const nextPlatform = apiConfigs.find(c => c.id === nextApiConfigId)?.platform;
                setForm(prev => {
                  const next = { ...prev, apiConfigId: nextApiConfigId };
                  if (!supportsReasoningConfig(nextPlatform)) {
                    next.supportReasoning = false;
                    next.config = stripReasoningConfig(prev.config);
                  }
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
            <p className="text-[10px] text-muted-foreground">模型将使用该 API 配置中的密钥进行调用</p>
          </div>

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

          <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
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
                <p className="text-[10px] text-muted-foreground/70">用于模型级别的并发治理，留空或置空时回落到默认值 5。</p>
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
                  <p className="text-[10px] text-muted-foreground/70">用于标记该模型的上下文容量；留空表示不额外声明。</p>
                </div>
              )}
            </div>

            {form.modelType === 1 && (
              <div className="space-y-2.5">
                <ToggleSettingCard
                  checked={!!form.supportReasoning}
                  title="支持思考"
                  description="作为模型能力兜底开关；在 Anthropic、DashScope、OpenAI 兼容渠道会触发 reasoning 默认逻辑。"
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
                  description="用于标记该对话模型支持图片等多模态输入，便于后续业务端做能力过滤。"
                  onToggle={() => updateField("supportVision", !form.supportVision)}
                />
              </div>
            )}
          </div>

          {/* 模型配置 */}
          <ModelConfigForm
            modelType={form.modelType}
            platform={selectedPlatform}
            supportReasoning={!!form.supportReasoning}
            configJson={form.config}
            baseConfig={matchedPreset?.config as Record<string, unknown> | undefined}
            onChange={json => updateField("config", json)}
          />
        </div>

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button size="sm" onClick={handleSave} disabled={saving || !form.name.trim() || !form.code.trim() || !form.apiConfigId}>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            {editingModel ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// 主页面
// ============================================================

export default function AiModelsPage() {
  // AI 模型列表
  const [models, setModels] = useState<AiModel[]>([]);
  const [modelPresets, setModelPresets] = useState<ModelPreset[]>([]);

  // API 配置列表
  const [configs, setConfigs] = useState<ApiConfig[]>([]);
  const [configsLoading, setConfigsLoading] = useState(true);

  // Dialog 状态
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<ApiConfig | null>(null);
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<AiModel | null>(null);
  const [modelDialogApiConfigId, setModelDialogApiConfigId] = useState<number | undefined>(undefined);
  const [fetchModelsDialogOpen, setFetchModelsDialogOpen] = useState(false);
  const [fetchModelsConfig, setFetchModelsConfig] = useState<ApiConfig | null>(null);

  const loadModels = useCallback(async () => {
    try {
      const data = await aiModelApi.list();
      setModels(data);
    } catch (err) {
      console.error("加载 AI 模型列表失败:", err);
    }
  }, []);

  const loadConfigs = useCallback(async () => {
    try {
      setConfigsLoading(true);
      const data = await apiConfigApi.list();
      setConfigs(data);
    } catch (err) {
      console.error("加载 API 配置列表失败:", err);
    } finally {
      setConfigsLoading(false);
    }
  }, []);

  const loadModelPresets = useCallback(async () => {
    try {
      const data = await aiModelApi.presets();
      setModelPresets(data);
    } catch (err) {
      console.error("加载模型预设失败:", err);
    }
  }, []);

  useEffect(() => {
    loadModels();
    loadConfigs();
    loadModelPresets();
  }, [loadConfigs, loadModelPresets, loadModels]);

  const handleDeleteModel = async (id: number) => {
    if (!confirm("确定要删除该 AI 模型吗？")) return;
    try {
      await aiModelApi.delete(id);
      await loadModels();
    } catch (err) {
      console.error("删除模型失败:", err);
    }
  };

  const handleDeleteConfig = async (id: number) => {
    if (!confirm("确定要删除该 API 配置吗？")) return;
    try {
      await apiConfigApi.delete(id);
      await loadConfigs();
    } catch (err) {
      console.error("删除配置失败:", err);
    }
  };

  return (
    <motion.div
      className="max-w-[1200px]"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* 页面标题 */}
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">AI 服务管理</h1>
        <p className="text-muted-foreground mt-1">
          管理 API 配置和 AI 模型
        </p>
      </motion.div>

      {/* ========== AI 服务管理（统一卡片式） ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <div className="flex items-center justify-between mb-3 px-1">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            API 配置与模型
          </h3>
          <button
            onClick={() => { setEditingConfig(null); setConfigDialogOpen(true); }}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium",
              "border border-dashed border-border/40 hover:border-primary/50",
              "text-muted-foreground hover:text-primary",
              "transition-all duration-200"
            )}
          >
            <Plus className="h-3.5 w-3.5" />
            添加 API 配置
          </button>
        </div>

        <div className="space-y-4">
          {configsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : configs.length === 0 ? (
            <div className={cn(
              "rounded-xl border border-dashed border-border/30 py-10 text-center",
              "bg-card/30"
            )}>
              <Settings2 className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">还没有 API 配置</p>
              <p className="text-xs text-muted-foreground/60 mt-1">点击上方「添加 API 配置」开始</p>
            </div>
          ) : (
            configs.map((config) => {
              const pColor = platformIconColors[config.platform || ""] || { color: "text-green-400", bg: "bg-green-500/10" };
              const configModels = models.filter(m => m.apiConfigId === config.id);

              // 按类型分组
              const modelTypeGroups = [
                { type: 2, label: "图像生成" },
                { type: 3, label: "视频生成" },
                { type: 1, label: "对话" },
              ].map(g => ({
                ...g,
                models: configModels.filter(m => m.modelType === g.type),
              })).filter(g => g.models.length > 0);

              return (
                <div
                  key={config.id}
                  className={cn(
                    "rounded-xl border overflow-hidden",
                    "bg-card/50 backdrop-blur-sm border-border/50"
                  )}
                >
                  {/* ── API 配置头部 ── */}
                  <div className="flex items-center gap-3 px-4 py-3 group border-b border-border/40">
                    <div className={cn("h-9 w-9 rounded-lg flex items-center justify-center shrink-0", pColor.bg)}>
                      <Settings2 className={cn("h-4.5 w-4.5", pColor.color)} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium">{config.name}</p>
                        {config.platform && (
                          <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px] text-muted-foreground">
                            {PLATFORM_LABELS[config.platform] || config.platform}
                          </span>
                        )}
                        <div className={cn(
                          "w-1.5 h-1.5 rounded-full shrink-0",
                          config.status === 1 ? "bg-green-400" : "bg-muted-foreground/30"
                        )} />
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                        {config.apiKey && (
                          <span className="font-mono text-[10px]">{maskSecret(config.apiKey)}</span>
                        )}
                        <span>{configModels.length} 个模型</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => { setFetchModelsConfig(config); setFetchModelsDialogOpen(true); }}
                        className="p-1.5 rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
                        title="获取可用模型列表"
                      >
                        <CloudDownload className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => { setEditingConfig(config); setConfigDialogOpen(true); }}
                        className="p-1.5 rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
                      >
                        <Edit2 className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => handleDeleteConfig(config.id)}
                        className="p-1.5 rounded-md text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* ── 模型列表（按类型分组） ── */}
                  <div className="px-4 py-2">
                    {modelTypeGroups.length === 0 ? (
                      <p className="text-xs text-muted-foreground/50 text-center py-3">
                        暂无模型
                      </p>
                    ) : (
                      modelTypeGroups.map((group, gi) => {
                        const defaultModel = group.models.find(m => m.defaultModel);

                        return (
                          <div key={group.type} className={cn(gi > 0 && "mt-2 pt-2 border-t border-border/30")}>
                            {/* 类型分组标题 */}
                            <div className="flex items-center gap-2 px-1 py-1">
                              <span className="text-[11px] font-medium text-muted-foreground shrink-0">
                                {group.label}
                              </span>
                              <div className="flex-1 h-px bg-border/15" />
                              {defaultModel ? (
                                <span className="inline-flex items-center gap-1 text-[10px] text-amber-500/80 font-medium shrink-0">
                                  <Star className="h-2.5 w-2.5" />
                                  默认: {defaultModel.name}
                                </span>
                              ) : (
                                <span className="text-[10px] text-muted-foreground/30 shrink-0">未设默认</span>
                              )}
                            </div>

                            {/* 模型行 */}
                            {group.models.map((model) => (
                              (() => {
                                const matchedPreset = findMatchingPreset(model, config.platform, modelPresets);
                                const effectiveModelConfig = mergeConfigObjects(
                                  matchedPreset?.config || {},
                                  parseConfigJson(model.config)
                                );
                                const capabilityView = buildGenerationCapabilityView(model, effectiveModelConfig);

                                return (
                                  <div
                                    key={model.id}
                                    className="flex items-start gap-3 px-3 py-2.5 rounded-lg group/model hover:bg-white/5 transition-colors"
                                  >
                                    <div className="h-7 w-7 rounded-md bg-primary/8 flex items-center justify-center shrink-0 mt-0.5">
                                      <Bot className="h-3.5 w-3.5 text-primary/60" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <div className="flex items-center gap-2">
                                        <p className="text-sm font-medium">{model.name}</p>
                                        {model.defaultModel && (
                                          <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-amber-500/10 text-[10px] text-amber-500 font-medium">
                                            <Star className="h-2.5 w-2.5" />
                                            默认
                                          </span>
                                        )}
                                        <div className={cn(
                                          "w-1.5 h-1.5 rounded-full shrink-0",
                                          model.status === 1 ? "bg-green-400" : "bg-muted-foreground/30"
                                        )} />
                                      </div>
                                      <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground mt-0.5 flex-wrap">
                                        <span className="font-mono px-1 py-0.5 rounded bg-muted/40">{model.code}</span>
                                        {model.supportReasoning && (
                                          <span className="px-1 py-0.5 rounded bg-sky-500/10 text-sky-500">思考</span>
                                        )}
                                        {model.supportVision && (
                                          <span className="px-1 py-0.5 rounded bg-emerald-500/10 text-emerald-500">视觉</span>
                                        )}
                                        {model.contextWindow && model.contextWindow > 0 && (
                                          <span className="px-1 py-0.5 rounded bg-muted/50 text-muted-foreground">
                                            {model.contextWindow.toLocaleString()} ctx
                                          </span>
                                        )}
                                      </div>
                                      {capabilityView && (
                                        <div className="mt-1.5 space-y-1.5">
                                          <div className="flex flex-wrap gap-1">
                                            {capabilityView.chips.map(chip => (
                                              <span
                                                key={`${model.id}-${chip.label}`}
                                                className={cn(
                                                  "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] leading-none",
                                                  getCapabilityChipClassName(chip.tone)
                                                )}
                                              >
                                                {chip.label}
                                              </span>
                                            ))}
                                          </div>
                                          <p className="text-[10px] text-muted-foreground/75 leading-4">
                                            {capabilityView.summary}
                                          </p>
                                        </div>
                                      )}
                                    </div>

                                    {/* 操作按钮 */}
                                    <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover/model:opacity-100 transition-opacity mt-0.5">
                                      {!model.defaultModel && (
                                        <button
                                          onClick={async () => {
                                            try {
                                              const sameTypeModels = models.filter(m => m.modelType === model.modelType && m.defaultModel);
                                              for (const dm of sameTypeModels) {
                                                await aiModelApi.update({ id: dm.id, defaultModel: false });
                                              }
                                              await aiModelApi.update({ id: model.id, defaultModel: true });
                                              await loadModels();
                                            } catch (err) {
                                              console.error("设置默认模型失败:", err);
                                            }
                                          }}
                                          className={cn(
                                            "flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all",
                                            "border border-amber-500/30 text-amber-500",
                                            "hover:bg-amber-500/10 hover:border-amber-500/50"
                                          )}
                                        >
                                          <Star className="h-3 w-3" />
                                          设为默认
                                        </button>
                                      )}
                                      <button
                                        onClick={() => { setEditingModel(model); setModelDialogApiConfigId(undefined); setModelDialogOpen(true); }}
                                        className="p-1 rounded-md text-muted-foreground/50 hover:text-primary hover:bg-primary/10 transition-colors"
                                      >
                                        <Edit2 className="h-3 w-3" />
                                      </button>
                                      <button
                                        onClick={() => handleDeleteModel(model.id)}
                                        className="p-1 rounded-md text-muted-foreground/50 hover:text-destructive hover:bg-destructive/10 transition-colors"
                                      >
                                        <Trash2 className="h-3 w-3" />
                                      </button>
                                    </div>
                                  </div>
                                );
                              })()
                            ))}
                          </div>
                        );
                      })
                    )}

                    {/* 添加模型按钮 */}
                    <button
                      onClick={() => {
                        setEditingModel(null);
                        setModelDialogApiConfigId(config.id);
                        setModelDialogOpen(true);
                      }}
                      className={cn(
                        "flex items-center gap-2 w-full px-3 py-2 mt-1 rounded-lg",
                        "border border-dashed border-border/40 hover:border-primary/40",
                        "text-xs text-muted-foreground/60 hover:text-primary",
                        "transition-all duration-200"
                      )}
                    >
                      <Plus className="h-3.5 w-3.5" />
                      添加模型
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </motion.div>

      {/* Dialogs */}
      <ApiConfigDialog
        open={configDialogOpen}
        onOpenChange={setConfigDialogOpen}
        editingConfig={editingConfig}
        onSaved={() => { loadConfigs(); loadModels(); }}
      />
      <AiModelDialog
        open={modelDialogOpen}
        onOpenChange={setModelDialogOpen}
        editingModel={editingModel}
        apiConfigs={configs}
        defaultApiConfigId={modelDialogApiConfigId}
        onSaved={() => { loadModels(); }}
      />
      {fetchModelsConfig && (
        <FetchRemoteModelsDialog
          open={fetchModelsDialogOpen}
          onOpenChange={setFetchModelsDialogOpen}
          apiConfig={fetchModelsConfig}
          existingModelCodes={new Set(models.filter(m => m.apiConfigId === fetchModelsConfig.id).map(m => m.code))}
          onAdded={() => { loadModels(); }}
        />
      )}
    </motion.div>
  );
}
