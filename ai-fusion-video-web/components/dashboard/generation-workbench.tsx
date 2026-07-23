"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ImageIcon, Settings2, Video } from "lucide-react";
import { toast } from "sonner";
import { aiModelApi, type AiModel, type ModelPreset } from "@/lib/api/ai-model";
import { generationApi } from "@/lib/api/generation";
import { resolveGenerationCapabilities } from "@/lib/generation-capabilities";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { GenerationAdvancedPanel } from "./generation/generation-advanced-panel";
import { GenerationAssetDialog } from "./generation/generation-asset-dialog";
import { GenerationHistory } from "./generation/generation-history";
import { GenerationSimpleComposer } from "./generation/generation-simple-composer";
import { OverlayScrollArea } from "./overlay-scroll-area";
import {
  EMPTY_GENERATION_FORM,
  type ComposerMode,
  type GenerationAssetTarget,
  type GenerationFormState,
  type GenerationHistoryEntry,
  type GenerationResultItem,
  type SimpleAttachment,
  type WorkbenchMode,
} from "./generation/generation-types";
import {
  getResultMediaUrl,
  getResultPreviewUrl,
  parseLimit,
} from "./generation/generation-utils";

interface GenerationWorkbenchProps {
  mode: WorkbenchMode;
}

function clean(values: string[]) {
  return values.map((value) => value.trim()).filter(Boolean);
}

export default function GenerationWorkbench({ mode }: GenerationWorkbenchProps) {
  const modelType = mode === "image" ? 2 : 3;
  const [composerMode, setComposerMode] = useState<ComposerMode>("simple");
  const [models, setModels] = useState<AiModel[]>([]);
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [modelId, setModelId] = useState<number | undefined>();
  const [loadingModels, setLoadingModels] = useState(true);
  const [form, setForm] = useState<GenerationFormState>(EMPTY_GENERATION_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [history, setHistory] = useState<GenerationHistoryEntry[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLimit, setHistoryLimit] = useState(10);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [assetTarget, setAssetTarget] = useState<GenerationAssetTarget | null>(null);
  const composerRef = useRef<HTMLDivElement>(null);
  const [composerHeight, setComposerHeight] = useState(220);

  useEffect(() => {
    let cancelled = false;
    setLoadingModels(true);
    Promise.all([aiModelApi.listByType(modelType), aiModelApi.presets(modelType)])
      .then(([modelList, presetList]) => {
        if (cancelled) return;
        const enabled = modelList.filter((model) => model.status === 1);
        setModels(enabled);
        setPresets(presetList);
        setModelId(enabled.find((model) => model.defaultModel)?.id ?? enabled[0]?.id);
      })
      .catch(() => {
        if (!cancelled) setModels([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingModels(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modelType]);

  const selectedModel = models.find((model) => model.id === modelId);
  const capabilities = useMemo(
    () =>
      selectedModel
        ? resolveGenerationCapabilities(selectedModel, presets)
        : null,
    [presets, selectedModel],
  );

  useEffect(() => {
    if (!capabilities) return;
    setForm((current) => ({
      ...current,
      ratio: capabilities.supportedAspectRatios[0] || "",
      resolution: capabilities.supportedResolutions[0] || "",
      count: Math.min(current.count, parseLimit(capabilities.maxCount, 4)),
      duration: Math.min(
        Math.max(capabilities.defaultDuration, capabilities.minDuration),
        capabilities.maxDuration,
      ),
      referenceImages: [],
      firstFrame: [],
      lastFrame: [],
      referenceVideos: [],
      referenceAudios: [],
    }));
  }, [capabilities, modelId]);

  const loadHistory = useCallback(
    async (limit: number) => {
      setHistoryLoading(true);
      try {
        if (mode === "image") {
          const page = await generationApi.pageImageTasks(1, limit);
          const entries = await Promise.all(
            page.list.map(async (task) => ({
              task,
              items: await generationApi.listImageItems(task.id).catch(() => []),
            })),
          );
          setHistory(entries);
          setHistoryTotal(page.total);
        } else {
          const page = await generationApi.pageVideoTasks(1, limit);
          const entries = await Promise.all(
            page.list.map(async (task) => ({
              task,
              items: await generationApi.listVideoItems(task.id).catch(() => []),
            })),
          );
          setHistory(entries);
          setHistoryTotal(page.total);
        }
      } catch {
        setHistory([]);
        setHistoryTotal(0);
      } finally {
        setHistoryLoading(false);
      }
    },
    [mode],
  );

  useEffect(() => {
    void loadHistory(historyLimit);
  }, [historyLimit, loadHistory]);

  useEffect(() => {
    const element = composerRef.current;
    if (!element) return;

    const updateHeight = () => {
      setComposerHeight(Math.ceil(element.getBoundingClientRect().height));
    };
    updateHeight();

    const observer = new ResizeObserver(updateHeight);
    observer.observe(element);
    return () => observer.disconnect();
  }, [composerMode]);

  const hasRunningTask = history.some(
    ({ task }) => task.status === 0 || task.status === 1,
  );

  useEffect(() => {
    if (!hasRunningTask) return;
    const timer = window.setInterval(() => {
      void loadHistory(historyLimit);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [hasRunningTask, historyLimit, loadHistory]);

  const updateForm = (patch: Partial<GenerationFormState>) => {
    setForm((current) => ({ ...current, ...patch }));
  };

  const simpleAttachments = useMemo<SimpleAttachment[]>(
    () => [
      ...form.firstFrame.map((url) => ({ url, label: "首帧" })),
      ...form.lastFrame.map((url) => ({ url, label: "尾帧" })),
      ...form.referenceImages.map((url) => ({ url, label: "参考图" })),
    ],
    [form.firstFrame, form.lastFrame, form.referenceImages],
  );

  const maxSimpleAttachments = useMemo(() => {
    if (!capabilities) return 0;
    if (mode === "image") {
      return capabilities.supportsReferenceImages
        ? parseLimit(capabilities.maxReferenceImages, 1)
        : 0;
    }
    const slots =
      (capabilities.supportsFirstFrame ? 1 : 0) +
      (capabilities.supportsLastFrame ? 1 : 0) +
      (capabilities.supportsReferenceImages
        ? parseLimit(capabilities.maxReferenceImages, 1)
        : 0);
    return capabilities.maxImageInputs > 0
      ? Math.min(slots || capabilities.maxImageInputs, capabilities.maxImageInputs)
      : slots;
  }, [capabilities, mode]);

  const addSimpleAttachments = (urls: string[]) => {
    if (!capabilities || !urls.length) return;
    setForm((current) => {
      if (mode === "image") {
        return {
          ...current,
          referenceImages: [...current.referenceImages, ...urls].slice(
            0,
            maxSimpleAttachments,
          ),
        };
      }

      const remaining = [...urls];
      const next = {
        firstFrame: [...current.firstFrame],
        lastFrame: [...current.lastFrame],
        referenceImages: [...current.referenceImages],
      };
      if (capabilities.supportsFirstFrame && next.firstFrame.length === 0) {
        const value = remaining.shift();
        if (value) next.firstFrame = [value];
      }
      if (capabilities.supportsLastFrame && next.lastFrame.length === 0) {
        const value = remaining.shift();
        if (value) next.lastFrame = [value];
      }
      if (capabilities.supportsReferenceImages && remaining.length > 0) {
        next.referenceImages = [...next.referenceImages, ...remaining].slice(
          0,
          parseLimit(capabilities.maxReferenceImages, 1),
        );
      }
      return { ...current, ...next };
    });
  };

  const removeSimpleAttachment = (url: string) => {
    setForm((current) => ({
      ...current,
      firstFrame: current.firstFrame.filter((value) => value !== url),
      lastFrame: current.lastFrame.filter((value) => value !== url),
      referenceImages: current.referenceImages.filter((value) => value !== url),
    }));
  };

  const validate = () => {
    if (!modelId || !capabilities) return "请选择生成模型";
    if (!form.prompt.trim()) return "请输入提示词";
    const referenceImages = clean(form.referenceImages);
    if (
      mode === "image" &&
      referenceImages.length > 0 &&
      referenceImages.length < capabilities.minReferenceImages
    ) {
      return `当前模型至少需要 ${capabilities.minReferenceImages} 张参考图`;
    }
    const imageInputs =
      referenceImages.length +
      clean(form.firstFrame).length +
      clean(form.lastFrame).length;
    if (mode === "video" && imageInputs < capabilities.minImageInputs) {
      return `当前模型至少需要 ${capabilities.minImageInputs} 个图片输入`;
    }
    return "";
  };

  const submit = async () => {
    const error = validate();
    if (error) {
      toast.error(error);
      return;
    }
    if (!modelId) return;

    setSubmitting(true);
    try {
      const referenceImages = clean(form.referenceImages);
      if (mode === "image") {
        await generationApi.submitImage({
          prompt: form.prompt.trim(),
          modelId,
          refImageUrls: referenceImages.length
            ? JSON.stringify(referenceImages)
            : undefined,
          ratio: form.ratio || undefined,
          aspectRatio: form.ratio || undefined,
          resolution: form.resolution || undefined,
          count: form.count,
          category: "manual_image_generation",
        });
      } else {
        const firstFrame = clean(form.firstFrame);
        const lastFrame = clean(form.lastFrame);
        const referenceVideos = clean(form.referenceVideos);
        const referenceAudios = clean(form.referenceAudios);
        const imageInputCount =
          referenceImages.length + firstFrame.length + lastFrame.length;
        await generationApi.submitVideo({
          prompt: form.prompt.trim(),
          modelId,
          generateMode: imageInputCount > 0 ? "image2video" : "text2video",
          firstFrameImageUrl: firstFrame[0],
          lastFrameImageUrl: lastFrame[0],
          referenceImageUrls: referenceImages.length
            ? JSON.stringify(referenceImages)
            : undefined,
          referenceVideoUrls: referenceVideos.length
            ? JSON.stringify(referenceVideos)
            : undefined,
          referenceAudioUrls: referenceAudios.length
            ? JSON.stringify(referenceAudios)
            : undefined,
          ratio: form.ratio || undefined,
          resolution: form.resolution || undefined,
          duration: form.duration,
          watermark: form.watermark,
          generateAudio: form.generateAudio,
          cameraFixed: form.cameraFixed,
          seed: form.seed.trim() ? Number(form.seed) : undefined,
          count: form.count,
          category: "manual_video_generation",
        });
      }

      toast.success("任务已提交");
      setForm((current) => ({
        ...current,
        prompt: "",
        referenceImages: [],
        firstFrame: [],
        lastFrame: [],
        referenceVideos: [],
        referenceAudios: [],
      }));
      setHistoryLimit(10);
      await loadHistory(10);
    } catch (cause) {
      toast.error(cause instanceof Error ? cause.message : "提交失败");
    } finally {
      setSubmitting(false);
    }
  };

  const focusComposer = () => {
    composerRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  };

  const reusePrompt = (prompt: string) => {
    updateForm({ prompt });
    focusComposer();
  };

  const useReference = (item: GenerationResultItem) => {
    if (mode === "video" && "videoUrl" in item && item.videoUrl) {
      if (capabilities?.supportsReferenceVideos) {
        updateForm({
          referenceVideos: [
            ...form.referenceVideos,
            item.videoUrl,
          ].slice(0, parseLimit(capabilities.maxReferenceVideos, 1)),
        });
        toast.success("已加入参考视频");
        focusComposer();
        return;
      }
      const preview = getResultPreviewUrl(item);
      if (preview) addSimpleAttachments([preview]);
    } else {
      const media = getResultMediaUrl(item) || getResultPreviewUrl(item);
      if (media) addSimpleAttachments([media]);
    }
    toast.success("已加入参考素材");
    focusComposer();
  };

  const title = mode === "image" ? "生图" : "生视频";
  const PageIcon = mode === "image" ? ImageIcon : Video;
  const advancedMode = composerMode === "advanced";
  const historyBottomInset =
    composerMode === "simple" ? composerHeight + 16 : 0;

  return (
    <div className="relative flex min-h-0 w-full grow basis-0 flex-col overflow-hidden pb-20 lg:pb-3">
      <div className="w-full shrink-0 px-5 lg:px-8">
        <header
          className={cn(
            "mx-auto flex w-full items-center justify-between gap-4 pb-3",
            advancedMode ? "max-w-[1480px]" : "max-w-[1180px]",
          )}
        >
          <div className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-2xl border border-border/45 bg-card text-primary shadow-sm">
              <PageIcon className="h-4.5 w-4.5" />
            </span>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {mode === "image" ? "描述想法，生成画面" : "描述场景、动作和镜头"}
              </p>
            </div>
          </div>
          <a
            href="/settings/ai-models"
            className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}
          >
            <Settings2 className="h-3.5 w-3.5" />
            模型配置
          </a>
        </header>
      </div>

      <div
        data-slot="generation-workspace"
        className={cn(
          "min-h-0 flex-1",
          advancedMode && "px-5 lg:px-8",
        )}
      >
        <div
          className={cn(
            "h-full min-h-0",
            advancedMode &&
              "mx-auto grid w-full max-w-[1480px] gap-4 overflow-y-auto xl:grid-cols-[minmax(0,1fr)_minmax(420px,480px)] xl:overflow-hidden",
          )}
        >
          <OverlayScrollArea
            data-slot="generation-history-scroll"
            className={cn(
              "h-full min-h-0 w-full",
              advancedMode && "max-h-[42vh] xl:max-h-none",
            )}
            viewportStyle={{
              paddingBottom: historyBottomInset,
              scrollPaddingBottom: historyBottomInset,
            }}
          >
            <div className={cn("w-full", !advancedMode && "px-5 lg:px-8")}>
              <div
                className={cn(
                  "mx-auto w-full",
                  !advancedMode && "max-w-[1180px]",
                )}
              >
                <GenerationHistory
                  mode={mode}
                  entries={history}
                  models={models}
                  loading={historyLoading}
                  total={historyTotal}
                  onRefresh={() => void loadHistory(historyLimit)}
                  onLoadMore={() =>
                    setHistoryLimit((current) => Math.min(100, current + 10))
                  }
                  onUseSuggestion={reusePrompt}
                  onReusePrompt={reusePrompt}
                  onUseReference={useReference}
                  onAddAsset={(item, prompt) =>
                    setAssetTarget({ mode, item, prompt })
                  }
                />
              </div>
            </div>
          </OverlayScrollArea>

          {advancedMode && (
            <div
              data-slot="generation-composer-layer"
              className="min-h-0 xl:h-full"
            >
              <div
                data-slot="generation-composer"
                ref={composerRef}
                className="h-full min-h-0"
              >
                <GenerationAdvancedPanel
                  mode={mode}
                  models={models}
                  modelId={modelId}
                  loadingModels={loadingModels}
                  capabilities={capabilities}
                  form={form}
                  submitting={submitting}
                  onModelChange={setModelId}
                  onFormChange={updateForm}
                  onSubmit={() => void submit()}
                  onSimple={() => setComposerMode("simple")}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {!advancedMode && (
        <div
          data-slot="generation-composer-layer"
          className="pointer-events-none absolute inset-x-0 bottom-20 z-20 lg:bottom-3"
        >
          <div
            aria-hidden="true"
            className="absolute inset-x-0 -top-12 -bottom-20 bg-linear-to-t from-background via-background/35 to-transparent lg:-bottom-3"
          />
          <div className="relative z-10 w-full px-5 lg:px-8">
            <div
              data-slot="generation-composer"
              ref={composerRef}
              className="pointer-events-auto mx-auto w-full max-w-[1180px]"
            >
              <GenerationSimpleComposer
                mode={mode}
                models={models}
                modelId={modelId}
                loadingModels={loadingModels}
                capabilities={capabilities}
                form={form}
                attachments={simpleAttachments}
                maxAttachments={maxSimpleAttachments}
                minAttachments={
                  mode === "video" ? capabilities?.minImageInputs || 0 : 0
                }
                submitting={submitting}
                onModelChange={setModelId}
                onFormChange={updateForm}
                onAddAttachments={addSimpleAttachments}
                onRemoveAttachment={removeSimpleAttachment}
                onSubmit={() => void submit()}
                onAdvanced={() => setComposerMode("advanced")}
              />
            </div>
          </div>
        </div>
      )}

      {assetTarget && (
        <GenerationAssetDialog
          key={`${mode}-${assetTarget.item.id}`}
          target={assetTarget}
          open
          onOpenChange={(open) => {
            if (!open) setAssetTarget(null);
          }}
          onAttached={() => setAssetTarget(null)}
        />
      )}
    </div>
  );
}
