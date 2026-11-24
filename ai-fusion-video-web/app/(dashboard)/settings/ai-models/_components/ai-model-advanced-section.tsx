"use client";

import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  PLATFORM_LABELS,
  type AiModelCreateReq,
  type ModelPreset,
  type MultimodalInputTransports,
  type MultimodalInputType,
} from "@/lib/api/ai-model";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { ModelConfigForm } from "./model-config-form";
import { MultimodalCapabilityEditor } from "./multimodal-capability-editor";
import { ReasoningEffortLevelsEditor } from "./reasoning-effort-levels-editor";
import {
  getPositiveNumberValue,
  isConfigRecord,
  parseConfigJson,
  stringifyConfigObject,
  stripReasoningConfig,
  ToggleSettingCard,
} from "./model-config-support";

export type AdvancedSectionTab = "capabilities" | "parameters" | "json";
export type AiModelFormState = AiModelCreateReq & { id?: number; status?: number };

interface AiModelAdvancedSectionProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  activeTab: AdvancedSectionTab;
  onActiveTabChange: (tab: AdvancedSectionTab) => void;
  form: AiModelFormState;
  setForm: React.Dispatch<React.SetStateAction<AiModelFormState>>;
  selectedPlatform?: string | null;
  selectedCapabilityPreset: ModelPreset | null;
  effectiveModelConfig: Record<string, unknown>;
  showOpenAiResponsesMode: boolean;
  apiModeValue: string;
  onUpdateOpenAiResponsesMode: (rawValue: string) => void;
  onUpdateCustomCapabilityConfig: (configJson: string) => void;
  rawJsonDraft: string;
  rawJsonDirty: boolean;
  rawJsonError: string | null;
  onRawJsonDraftChange: (value: string) => void;
  onApplyRawJson: () => void;
  onResetRawJson: () => void;
  invalidField?: string;
}

export function AiModelAdvancedSection({
  open,
  onOpenChange,
  activeTab,
  onActiveTabChange,
  form,
  setForm,
  selectedPlatform,
  selectedCapabilityPreset,
  effectiveModelConfig,
  showOpenAiResponsesMode,
  apiModeValue,
  onUpdateOpenAiResponsesMode,
  onUpdateCustomCapabilityConfig,
  rawJsonDraft,
  rawJsonDirty,
  rawJsonError,
  onRawJsonDraftChange,
  onApplyRawJson,
  onResetRawJson,
  invalidField,
}: AiModelAdvancedSectionProps) {
  const updateField = <K extends keyof AiModelFormState>(
    key: K,
    value: AiModelFormState[K],
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const updateMetaNumberField = (
    key: "maxConcurrency" | "contextWindow",
    rawValue: string,
  ) => {
    if (!rawValue.trim()) {
      updateField(key, 0);
      return;
    }
    const parsed = Number.parseInt(rawValue, 10);
    updateField(key, Number.isNaN(parsed) ? 0 : parsed);
  };

  const formatRawJson = () => {
    if (!rawJsonDraft.trim()) {
      return;
    }
    try {
      const parsed = JSON.parse(rawJsonDraft);
      if (isConfigRecord(parsed)) {
        onRawJsonDraftChange(JSON.stringify(parsed, null, 2));
      }
    } catch {
      // 语法错误时保留原文由下方校验提示
    }
  };

  const summaryChips: string[] = [];
  if (form.maxConcurrency && form.maxConcurrency !== 5) {
    summaryChips.push(`并发 ${form.maxConcurrency}`);
  }
  if (form.modelType === 1 && form.contextWindow && form.contextWindow > 0) {
    summaryChips.push(
      form.contextWindow >= 1000
        ? `${Math.round(form.contextWindow / 1000)}K 上下文`
        : `${form.contextWindow} 上下文`,
    );
  }
  if (form.modelType === 1 && form.supportReasoning) {
    summaryChips.push("思考");
  }
  if (form.modelType === 1 && (form.multimodalInputTypes?.length ?? 0) > 0) {
    summaryChips.push(`${form.multimodalInputTypes.length} 种多模态输入`);
  }
  const customKeyCount = Object.keys(parseConfigJson(form.config)).length;
  if (customKeyCount > 0) {
    summaryChips.push(`${customKeyCount} 项自定义参数`);
  }

  return (
    <Collapsible
      open={open}
      onOpenChange={onOpenChange}
      className="rounded-xl border border-border/30 bg-background/70"
    >
      <CollapsibleTrigger className="flex w-full items-center justify-between gap-3 rounded-xl px-3.5 py-3 text-left outline-none transition-colors hover:bg-muted/35 focus-visible:ring-2 focus-visible:ring-ring">
        <span className="flex min-w-0 flex-col gap-1">
          <span className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-foreground">高级设置</span>
            {rawJsonDirty && (
              <Badge
                variant="secondary"
                className="bg-amber-500/12 text-amber-700 dark:text-amber-400"
              >
                JSON 已修改
              </Badge>
            )}
          </span>
          <span className="text-xs text-muted-foreground">
            {summaryChips.length > 0
              ? summaryChips.join(" · ")
              : "并发限制、思考/多模态能力、生成参数与原始 JSON"}
          </span>
        </span>
        <ChevronRight
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform duration-200 motion-reduce:transition-none",
            open && "rotate-90",
          )}
        />
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="border-t border-border/20 p-3.5">
          <Tabs
            value={activeTab}
            onValueChange={(value) => onActiveTabChange(value as AdvancedSectionTab)}
            className="gap-4"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <TabsList aria-label="高级设置分类">
                <TabsTrigger value="capabilities">能力与限制</TabsTrigger>
                <TabsTrigger value="parameters">生成参数</TabsTrigger>
                <TabsTrigger value="json">
                  原始 JSON{rawJsonDirty ? " *" : ""}
                </TabsTrigger>
              </TabsList>
              {selectedPlatform && (
                <Badge variant="outline" className="text-xs">
                  {PLATFORM_LABELS[selectedPlatform] || selectedPlatform}
                </Badge>
              )}
            </div>

            {/* Tab 1: 能力与限制 */}
            <TabsContent value="capabilities" className="flex flex-col gap-4">
              {showOpenAiResponsesMode && (
                <div className="flex flex-col gap-2 rounded-lg border border-border/30 bg-muted/20 p-3">
                  <Label className="text-xs font-medium text-foreground">
                    OpenAI 兼容对话接口
                  </Label>
                  <Select
                    value={apiModeValue}
                    onValueChange={(value) => onUpdateOpenAiResponsesMode(String(value))}
                    items={[
                      { value: "__unset__", label: "自动" },
                      { value: "chat_completions", label: "Chat Completions" },
                      { value: "responses", label: "Responses API" },
                    ]}
                  >
                    <SelectTrigger aria-label="OpenAI 兼容对话接口" className="w-full text-sm">
                      <SelectValue placeholder="选择对话接口模式" />
                    </SelectTrigger>
                    <SelectContent className="text-sm">
                      <SelectGroup>
                        <SelectItem value="__unset__" className="text-sm">自动</SelectItem>
                        <SelectItem value="chat_completions" className="text-sm">
                          Chat Completions
                        </SelectItem>
                        <SelectItem value="responses" className="text-sm">
                          Responses API
                        </SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
                <div className="flex min-w-0 flex-col gap-2">
                  <Label htmlFor="model-max-concurrency" className="text-xs text-muted-foreground">
                    最大并发数
                  </Label>
                  <Input
                    id="model-max-concurrency"
                    type="number"
                    min={0}
                    step={1}
                    placeholder="默认 5"
                    value={getPositiveNumberValue(form.maxConcurrency)}
                    onChange={(event) =>
                      updateMetaNumberField("maxConcurrency", event.target.value)
                    }
                    className="font-mono text-sm"
                  />
                </div>

                {form.modelType === 1 && (
                  <div className="flex min-w-0 flex-col gap-2">
                    <Label htmlFor="model-context-window" className="text-xs text-muted-foreground">
                      上下文窗口（tokens）
                    </Label>
                    <Input
                      id="model-context-window"
                      type="number"
                      min={0}
                      step={1}
                      placeholder="例如：128000"
                      value={getPositiveNumberValue(form.contextWindow)}
                      onChange={(event) =>
                        updateMetaNumberField("contextWindow", event.target.value)
                      }
                      className="font-mono text-sm"
                    />
                  </div>
                )}
              </div>

              {form.modelType === 1 && (
                <div className="flex flex-col gap-3">
                  <ToggleSettingCard
                    checked={!!form.supportReasoning}
                    title="支持思考"
                    description="开启后可在助手中选择思考等级并请求推理过程。"
                    onToggle={() => {
                      setForm((prev) => {
                        const nextSupportReasoning = !prev.supportReasoning;
                        return {
                          ...prev,
                          supportReasoning: nextSupportReasoning,
                          reasoningEffortLevels: nextSupportReasoning
                            ? prev.reasoningEffortLevels?.length
                              ? prev.reasoningEffortLevels
                              : selectedCapabilityPreset?.reasoningEffortLevels ?? []
                            : [],
                          config: nextSupportReasoning
                            ? prev.config
                            : stripReasoningConfig(prev.config),
                        };
                      });
                    }}
                  />

                  {form.supportReasoning && (
                    <ReasoningEffortLevelsEditor
                      levels={form.reasoningEffortLevels ?? []}
                      onChange={(levels) => updateField("reasoningEffortLevels", levels)}
                    />
                  )}

                  <MultimodalCapabilityEditor
                    inputTypes={form.multimodalInputTypes ?? []}
                    transports={form.multimodalInputTransports ?? {}}
                    onChange={(
                      inputTypes: MultimodalInputType[],
                      transports: MultimodalInputTransports,
                    ) => {
                      setForm((previous) => ({
                        ...previous,
                        supportVision: inputTypes.includes("image"),
                        multimodalInputTypes: inputTypes,
                        multimodalInputTransports: transports,
                      }));
                    }}
                  />
                </div>
              )}
            </TabsContent>

            {/* Tab 2: 生成参数 */}
            <TabsContent value="parameters">
              <ModelConfigForm
                modelType={form.modelType}
                platform={selectedPlatform}
                supportReasoning={!!form.supportReasoning}
                configJson={stringifyConfigObject(effectiveModelConfig)}
                onChange={onUpdateCustomCapabilityConfig}
              />
            </TabsContent>

            {/* Tab 3: 原始 JSON */}
            <TabsContent value="json" className="flex flex-col gap-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs text-muted-foreground">
                  直接编辑模型参数 JSON；修改后点击“应用到表单”或直接保存弹窗均会生效。
                </p>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={formatRawJson}
                    disabled={!rawJsonDraft.trim()}
                  >
                    格式化
                  </Button>
                  {rawJsonDirty && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={onResetRawJson}
                    >
                      还原
                    </Button>
                  )}
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    disabled={!rawJsonDirty}
                    onClick={onApplyRawJson}
                  >
                    应用到表单
                  </Button>
                </div>
              </div>

              <Textarea
                id="model-raw-json"
                aria-label="自定义参数 JSON"
                aria-invalid={invalidField === "model-raw-json" || Boolean(rawJsonError)}
                value={rawJsonDraft}
                onChange={(event) => onRawJsonDraftChange(event.target.value)}
                placeholder='{"temperature": 0.7, "maxTokens": 4096}'
                spellCheck={false}
                className="min-h-44 font-mono text-xs leading-relaxed"
              />

              {rawJsonError && (
                <p role="alert" className="text-xs text-destructive">
                  {rawJsonError}
                </p>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
