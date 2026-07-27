"use client";

import {
  ArrowLeft,
  Cpu,
  Loader2,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import type { AiModel } from "@/lib/api/ai-model";
import type {
  GenerationCapabilities,
  ReferenceImageUploadAvailability,
} from "@/lib/generation-capabilities";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { GenerationModelPicker } from "./generation-model-picker";
import { GenerationOutputSettings } from "./generation-output-settings";
import { GenerationReferenceField } from "./generation-reference-field";
import type { GenerationFormState, WorkbenchMode } from "./generation-types";
import { parseLimit } from "./generation-utils";

interface GenerationAdvancedPanelProps {
  mode: WorkbenchMode;
  models: AiModel[];
  modelId?: number;
  loadingModels: boolean;
  capabilities: GenerationCapabilities | null;
  form: GenerationFormState;
  submitting: boolean;
  onModelChange: (modelId: number) => void;
  onFormChange: (patch: Partial<GenerationFormState>) => void;
  onSubmit: () => void;
  onSimple: () => void;
  referenceImageUploadAvailability: ReferenceImageUploadAvailability;
}

export function GenerationAdvancedPanel({
  mode,
  models,
  modelId,
  loadingModels,
  capabilities,
  form,
  submitting,
  onModelChange,
  onFormChange,
  onSubmit,
  onSimple,
  referenceImageUploadAvailability,
}: GenerationAdvancedPanelProps) {
  const outputLimit = parseLimit(capabilities?.maxCount || 0, 4);
  const imageInputCount =
    form.firstFrame.filter(Boolean).length +
    form.lastFrame.filter(Boolean).length +
    form.referenceImages.filter(Boolean).length;
  const missingRequiredInput = Boolean(
    capabilities &&
      ((mode === "video" && imageInputCount < capabilities.minImageInputs) ||
        (mode === "image" &&
          form.referenceImages.length > 0 &&
          form.referenceImages.length < capabilities.minReferenceImages)),
  );
  const canSubmit = Boolean(
    modelId && form.prompt.trim() && !submitting && !missingRequiredInput,
  );
  const promptSections =
    mode === "image"
      ? ["主体", "环境", "构图", "光线", "风格"]
      : ["主体", "动作", "镜头", "节奏", "声音"];
  const hasReferenceFields = Boolean(
    capabilities &&
      (capabilities.supportsFirstFrame ||
        capabilities.supportsLastFrame ||
        capabilities.supportsReferenceImages ||
        (mode === "video" &&
          (capabilities.supportsReferenceVideos ||
            capabilities.supportsReferenceAudios))),
  );

  const appendPromptSection = (section: string) => {
    const token = `${section}：`;
    if (form.prompt.includes(token)) return;
    onFormChange({
      prompt: form.prompt.trim() ? `${form.prompt.trim()}\n${token}` : token,
    });
  };

  return (
    <section className="flex h-full min-h-[520px] max-h-[72vh] flex-col overflow-hidden rounded-[26px] border border-border/55 bg-card/97 shadow-[0_8px_16px_-12px_rgba(15,23,42,.48)] backdrop-blur-xl xl:min-h-0 xl:max-h-none">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-border/35 bg-card/94 px-4 py-3 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <span className="grid h-9 w-9 place-items-center rounded-xl border border-primary/10 bg-primary/[0.065] text-primary shadow-sm">
            <SlidersHorizontal className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <h3 className="text-sm font-semibold">
              高级{mode === "image" ? "生图" : "视频生成"}
            </h3>
            <p className="truncate text-[10px] text-muted-foreground">
              提示词、素材与参数集中控制
            </p>
          </div>
        </div>
        <Button variant="ghost" size="xs" onClick={onSimple} className="rounded-xl">
          <ArrowLeft className="h-3.5 w-3.5" />
          快捷模式
        </Button>
      </header>

      <OverlayScrollArea className="min-h-0 flex-1">
        <div className="space-y-3 p-3">
          <section className="rounded-[18px] border border-border/45 bg-background/48 p-3.5">
            <div className="flex items-center gap-2.5">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                <Cpu className="h-3.5 w-3.5" />
              </span>
              <div className="min-w-0 flex-1">
                <h4 className="text-xs font-semibold">生成模型</h4>
                <p className="truncate text-[10px] text-muted-foreground">
                  切换模型后会同步可用参数
                </p>
              </div>
            </div>
            <GenerationModelPicker
              models={models}
              modelId={modelId}
              loading={loadingModels}
              onChange={onModelChange}
              className="mt-2.5 border border-border/40 bg-background/72 hover:bg-background"
            />
            {!loadingModels && models.length === 0 && (
              <p className="mt-2 rounded-xl border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-xs text-amber-600">
                未配置{mode === "image" ? "图片" : "视频"}模型
              </p>
            )}
            {capabilities && (
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                {capabilities.supportsReferenceImages && (
                  <span className="rounded-md bg-muted px-2 py-1 text-[10px] text-muted-foreground">
                    参考图 {parseLimit(capabilities.maxReferenceImages, 1)} 张
                  </span>
                )}
                {mode === "video" && (
                  <span className="rounded-md bg-muted px-2 py-1 text-[10px] text-muted-foreground">
                    {capabilities.minDuration}-{capabilities.maxDuration} 秒
                  </span>
                )}
                <span className="rounded-md bg-muted px-2 py-1 text-[10px] text-muted-foreground">
                  最多 {outputLimit} 个结果
                </span>
              </div>
            )}
          </section>

          <section className="overflow-hidden rounded-[18px] border border-border/45 bg-background/48 shadow-[0_14px_45px_-38px_rgba(15,23,42,.55)] transition-colors duration-150 focus-within:border-primary/25 motion-reduce:transition-none">
            <div className="flex items-center justify-between gap-3 px-3.5 pt-3.5">
              <div>
                <h4 className="text-xs font-semibold">创作提示词</h4>
                <p className="mt-0.5 text-[10px] text-muted-foreground">
                  清楚描述主体、场景和希望保留的细节
                </p>
              </div>
              <span className="text-[10px] tabular-nums text-muted-foreground">
                {form.prompt.length}/5000
              </span>
            </div>
            <Textarea
              value={form.prompt}
              onChange={(event) => onFormChange({ prompt: event.target.value })}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && canSubmit) {
                  event.preventDefault();
                  onSubmit();
                }
              }}
              maxLength={5000}
              rows={6}
              aria-label={mode === "image" ? "图片提示词" : "视频提示词"}
              placeholder={
                mode === "image"
                  ? "例如：雨夜霓虹街头的人像，湿润路面反光，电影感侧光，保留自然肤质与浅景深…"
                  : "例如：人物穿过雨夜街道，镜头缓慢跟随，霓虹倒影掠过画面，节奏克制…"
              }
              className="mt-1 min-h-36 resize-none rounded-none border-0 bg-transparent px-3.5 py-3 text-[14px] leading-6 shadow-none focus-visible:ring-0"
            />
            <div className="flex flex-wrap items-center gap-1.5 border-t border-border/30 bg-muted/18 px-3 py-2.5">
              <span className="mr-0.5 text-[10px] text-muted-foreground">快速补充</span>
              {promptSections.map((section) => (
                <button
                  key={section}
                  type="button"
                  onClick={() => appendPromptSection(section)}
                  className="h-6 rounded-lg border border-border/45 bg-background/70 px-2 text-[10px] text-muted-foreground transition-colors hover:border-primary/25 hover:bg-primary/5 hover:text-primary"
                >
                  + {section}
                </button>
              ))}
            </div>
          </section>

          {hasReferenceFields && capabilities && (
            <section className="rounded-[18px] border border-border/45 bg-background/38 p-3.5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h4 className="text-xs font-semibold">参考素材</h4>
                  <p className="mt-0.5 text-[10px] text-muted-foreground">
                    上传图片或粘贴 URL / Data URI
                  </p>
                </div>
                <span className="rounded-md bg-muted px-2 py-1 text-[10px] text-muted-foreground">
                  {imageInputCount} 个图片输入
                </span>
              </div>
              <div className="mt-3 space-y-2.5">
                {capabilities.supportsFirstFrame && (
                  <GenerationReferenceField
                    label="首帧图"
                    hint="视频开场画面"
                    values={form.firstFrame}
                    maxCount={1}
                    mediaType="image"
                    uploadDisabledReason={
                      !referenceImageUploadAvailability.supported
                        ? referenceImageUploadAvailability.reason
                        : undefined
                    }
                    manualInputAllowed={
                      capabilities.supportsReferenceImageUrlInput ||
                      capabilities.supportsReferenceImageBase64Input
                    }
                    onChange={(firstFrame) => onFormChange({ firstFrame })}
                  />
                )}
                {capabilities.supportsLastFrame && (
                  <GenerationReferenceField
                    label="尾帧图"
                    hint="视频结束画面"
                    values={form.lastFrame}
                    maxCount={1}
                    mediaType="image"
                    uploadDisabledReason={
                      !referenceImageUploadAvailability.supported
                        ? referenceImageUploadAvailability.reason
                        : undefined
                    }
                    manualInputAllowed={
                      capabilities.supportsReferenceImageUrlInput ||
                      capabilities.supportsReferenceImageBase64Input
                    }
                    onChange={(lastFrame) => onFormChange({ lastFrame })}
                  />
                )}
                {capabilities.supportsReferenceImages && (
                  <GenerationReferenceField
                    label="参考图片"
                    hint="主体、构图、风格或场景参考"
                    values={form.referenceImages}
                    maxCount={parseLimit(capabilities.maxReferenceImages, 12)}
                    mediaType="image"
                    uploadDisabledReason={
                      !referenceImageUploadAvailability.supported
                        ? referenceImageUploadAvailability.reason
                        : undefined
                    }
                    manualInputAllowed={
                      capabilities.supportsReferenceImageUrlInput ||
                      capabilities.supportsReferenceImageBase64Input
                    }
                    onChange={(referenceImages) =>
                      onFormChange({ referenceImages })
                    }
                  />
                )}
                {mode === "video" && capabilities.supportsReferenceVideos && (
                  <GenerationReferenceField
                    label="参考视频"
                    hint="动作、运镜或特效参考"
                    values={form.referenceVideos}
                    maxCount={parseLimit(capabilities.maxReferenceVideos, 3)}
                    mediaType="video"
                    onChange={(referenceVideos) =>
                      onFormChange({ referenceVideos })
                    }
                  />
                )}
                {mode === "video" && capabilities.supportsReferenceAudios && (
                  <GenerationReferenceField
                    label="参考音频"
                    hint="节奏、音乐、声音或对白参考"
                    values={form.referenceAudios}
                    maxCount={parseLimit(capabilities.maxReferenceAudios, 3)}
                    mediaType="audio"
                    onChange={(referenceAudios) =>
                      onFormChange({ referenceAudios })
                    }
                  />
                )}
              </div>
            </section>
          )}

          <GenerationOutputSettings
            mode={mode}
            capabilities={capabilities}
            form={form}
            onChange={onFormChange}
          />
        </div>
      </OverlayScrollArea>

      <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-border/35 bg-card/94 px-3.5 py-3 backdrop-blur-xl">
        <p
          className={cn(
            "min-w-0 flex-1 text-[10px] leading-4",
            missingRequiredInput ? "text-amber-600" : "text-muted-foreground",
          )}
        >
          {missingRequiredInput && capabilities
            ? `当前模型至少需要 ${
                mode === "video"
                  ? capabilities.minImageInputs
                  : capabilities.minReferenceImages
              } 张图片`
            : "参数会随所选模型自动调整"}
        </p>
        <Button
          variant={mode === "image" ? "image" : "video"}
          size="lg"
          disabled={!canSubmit}
          onClick={onSubmit}
          className="min-w-32 rounded-xl px-5"
        >
          {submitting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Sparkles className="h-4 w-4" />
          )}
          {submitting
            ? "提交中…"
            : mode === "image"
              ? "生成图片"
              : "生成视频"}
        </Button>
      </footer>
    </section>
  );
}
