"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ImageIcon, Settings2, Video } from "lucide-react";
import { toast } from "sonner";
import { aiModelApi, type AiModel, type ModelPreset } from "@/lib/api/ai-model";
import {
  generationApi,
  type ImageGenerationItem,
  type VideoGenerationItem,
} from "@/lib/api/generation";
import { resolveGenerationCapabilities } from "@/lib/generation-capabilities";
import { toastApiError } from "@/lib/api/toast-api-error";
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
  type SimpleAttachmentKind,
  type SimpleAttachmentUploadOption,
  type WorkbenchMode,
} from "./generation/generation-types";
import {
  getResultMediaUrl,
  getResultPreviewUrl,
  isGenerationTaskActive,
  parseLimit,
} from "./generation/generation-utils";
import { useReferenceImageUploadAvailability } from "./generation/use-reference-image-upload-availability";

interface GenerationWorkbenchProps {
  mode: WorkbenchMode;
}

function clean(values: string[]) {
  return values.map((value) => value.trim()).filter(Boolean);
}

const MODE_TRANSITION_EASE = [0.25, 0.46, 0.45, 0.94] as const;

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
  const [cancellingTaskIds, setCancellingTaskIds] = useState<Set<string>>(new Set());
  const [assetTarget, setAssetTarget] = useState<GenerationAssetTarget | null>(null);
  const simpleComposerRef = useRef<HTMLDivElement>(null);
  const advancedComposerRef = useRef<HTMLDivElement>(null);
  const historyRef = useRef<GenerationHistoryEntry[]>([]);
  const activePollInFlightRef = useRef(false);
  const [composerHeight, setComposerHeight] = useState(220);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    historyRef.current = history;
  }, [history]);

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
      .catch((error) => {
        if (!cancelled) {
          toastApiError(error, "加载生成模型失败");
          setModels([]);
        }
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
  const referenceImageUploadAvailability = useReferenceImageUploadAvailability(
    mode,
    capabilities,
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
              items: await generationApi.listImageItems(task.id).catch((error) => {
                toastApiError(error, "加载图片生成结果失败");
                return [];
              }),
            })),
          );
          setHistory(entries);
          setHistoryTotal(page.total);
        } else {
          const page = await generationApi.pageVideoTasks(1, limit);
          const entries = await Promise.all(
            page.list.map(async (task) => ({
              task,
              items: await generationApi.listVideoItems(task.id).catch((error) => {
                toastApiError(error, "加载视频生成结果失败");
                return [];
              }),
            })),
          );
          setHistory(entries);
          setHistoryTotal(page.total);
        }
      } catch (error) {
        toastApiError(error, "加载生成历史失败");
        setHistory([]);
        setHistoryTotal(0);
      } finally {
        setHistoryLoading(false);
      }
    },
    [mode],
  );

  const refreshActiveHistory = useCallback(async () => {
    if (activePollInFlightRef.current) return;

    const activeEntries = historyRef.current.filter(({ task }) =>
      isGenerationTaskActive(task),
    );
    if (activeEntries.length === 0) return;

    activePollInFlightRef.current = true;
    try {
      const refreshedEntries = await Promise.all(
        activeEntries.map(async (entry) => {
          try {
            if (mode === "image") {
              const task = await generationApi.getImageTask(entry.task.taskId);
              const previousItems = entry.items.filter(
                (item): item is ImageGenerationItem => "imageUrl" in item,
              );
              const items =
                task.status === 0
                  ? previousItems
                  : await generationApi
                      .listImageItems(task.id)
                      .catch((error) => {
                        toastApiError(error, "刷新图片生成结果失败");
                        return previousItems;
                      });
              return {
                id: task.id,
                entry: { task, items } satisfies GenerationHistoryEntry,
              };
            }

            const task = await generationApi.getVideoTask(entry.task.taskId);
            const previousItems = entry.items.filter(
              (item): item is VideoGenerationItem => "videoUrl" in item,
            );
            const items =
              task.status === 0
                ? previousItems
                : await generationApi
                    .listVideoItems(task.id)
                    .catch((error) => {
                      toastApiError(error, "刷新视频生成结果失败");
                      return previousItems;
                    });
            return {
              id: task.id,
              entry: { task, items } satisfies GenerationHistoryEntry,
            };
          } catch {
            return null;
          }
        }),
      );

      const updates = new Map(
        refreshedEntries
          .filter((result) => result !== null)
          .map((result) => [result.id, result.entry]),
      );
      if (updates.size === 0) return;

      setHistory((current) =>
        current.map((entry) => {
          if (!isGenerationTaskActive(entry.task)) return entry;
          return updates.get(entry.task.id) ?? entry;
        }),
      );
    } finally {
      activePollInFlightRef.current = false;
    }
  }, [mode]);

  const cancelTask = async (taskId: string) => {
    setCancellingTaskIds(current => new Set(current).add(taskId));
    try {
      if (mode === "image") {
        await generationApi.cancelImage(taskId);
      } else {
        await generationApi.cancelVideo(taskId);
      }
      toast.success("生成任务已取消");
      await loadHistory(historyLimit);
    } catch (error) {
      toastApiError(error, "取消生成任务失败");
    } finally {
      setCancellingTaskIds(current => {
        const next = new Set(current);
        next.delete(taskId);
        return next;
      });
    }
  };

  const deleteTask = async (id: number) => {
    try {
      if (mode === "image") {
        await generationApi.deleteImage(id);
      } else {
        await generationApi.deleteVideo(id);
      }
      toast.success("创作记录已删除");
      await loadHistory(historyLimit);
    } catch (error) {
      toastApiError(error, "删除创作记录失败");
    }
  };

  useEffect(() => {
    void loadHistory(historyLimit);
  }, [historyLimit, loadHistory]);

  useEffect(() => {
    if (composerMode !== "simple") return;
    const element = simpleComposerRef.current;
    if (!element) return;

    const updateHeight = () => {
      setComposerHeight(Math.ceil(element.getBoundingClientRect().height));
    };
    updateHeight();

    const observer = new ResizeObserver(updateHeight);
    observer.observe(element);
    return () => observer.disconnect();
  }, [composerMode]);

  const hasRunningTask = history.some(({ task }) =>
    isGenerationTaskActive(task),
  );

  useEffect(() => {
    if (!hasRunningTask) return;

    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") {
        void refreshActiveHistory();
      }
    };
    const timer = window.setInterval(refreshWhenVisible, 4000);
    document.addEventListener("visibilitychange", refreshWhenVisible);

    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [hasRunningTask, refreshActiveHistory]);

  const updateForm = (patch: Partial<GenerationFormState>) => {
    setForm((current) => ({ ...current, ...patch }));
  };

  const simpleAttachments = useMemo<SimpleAttachment[]>(
    () => [
      ...form.firstFrame.map((url) => ({
        url,
        label: "首帧",
        kind: "firstFrame" as const,
        mediaType: "image" as const,
      })),
      ...form.lastFrame.map((url) => ({
        url,
        label: "尾帧",
        kind: "lastFrame" as const,
        mediaType: "image" as const,
      })),
      ...form.referenceImages.map((url) => ({
        url,
        label: "参考图",
        kind: "referenceImage" as const,
        mediaType: "image" as const,
      })),
      ...form.referenceVideos.map((url) => ({
        url,
        label: "参考视频",
        kind: "referenceVideo" as const,
        mediaType: "video" as const,
      })),
      ...form.referenceAudios.map((url) => ({
        url,
        label: "参考音频",
        kind: "referenceAudio" as const,
        mediaType: "audio" as const,
      })),
    ],
    [
      form.firstFrame,
      form.lastFrame,
      form.referenceAudios,
      form.referenceImages,
      form.referenceVideos,
    ],
  );

  const maxSimpleImageAttachments = useMemo(() => {
    if (!capabilities) return 0;
    if (!referenceImageUploadAvailability.supported) return 0;
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
  }, [capabilities, mode, referenceImageUploadAvailability.supported]);

  const simpleAttachmentCapacity = useMemo(() => {
    if (!capabilities) return 0;
    const imageCapacity = maxSimpleImageAttachments;
    if (mode === "image") return imageCapacity;
    return (
      imageCapacity +
      (capabilities.supportsReferenceVideos
        ? parseLimit(capabilities.maxReferenceVideos, 3)
        : 0) +
      (capabilities.supportsReferenceAudios
        ? parseLimit(capabilities.maxReferenceAudios, 3)
        : 0)
    );
  }, [capabilities, maxSimpleImageAttachments, mode]);

  const simpleAttachmentOptions = useMemo<SimpleAttachmentUploadOption[]>(() => {
    if (!capabilities) return [];

    const options: SimpleAttachmentUploadOption[] = [];
    const imageInputCount =
      form.firstFrame.length +
      form.lastFrame.length +
      form.referenceImages.length;
    const sharedImageRemaining =
      mode === "video" && capabilities.maxImageInputs > 0
        ? Math.max(0, capabilities.maxImageInputs - imageInputCount)
        : Number.POSITIVE_INFINITY;
    const imageDisabledReason = referenceImageUploadAvailability.supported
      ? undefined
      : referenceImageUploadAvailability.reason;
    const addImageOption = (
      kind: SimpleAttachmentKind,
      label: string,
      typeRemaining: number,
    ) => {
      const remaining = Math.min(typeRemaining, sharedImageRemaining);
      if (remaining > 0) {
        options.push({
          kind,
          label,
          mediaType: "image",
          remaining,
          disabledReason: imageDisabledReason,
        });
      }
    };

    if (mode === "image") {
      if (capabilities.supportsReferenceImages) {
        addImageOption(
          "referenceImage",
          "参考图片",
          Math.max(
            0,
            parseLimit(capabilities.maxReferenceImages, 1) -
              form.referenceImages.length,
          ),
        );
      }
      return options;
    }

    if (capabilities.supportsFirstFrame) {
      addImageOption("firstFrame", "首帧图片", form.firstFrame.length ? 0 : 1);
    }
    if (capabilities.supportsLastFrame) {
      addImageOption("lastFrame", "尾帧图片", form.lastFrame.length ? 0 : 1);
    }
    if (capabilities.supportsReferenceImages) {
      addImageOption(
        "referenceImage",
        "参考图片",
        Math.max(
          0,
          parseLimit(capabilities.maxReferenceImages, 1) -
            form.referenceImages.length,
        ),
      );
    }
    if (capabilities.supportsReferenceVideos) {
      const remaining = Math.max(
        0,
        parseLimit(capabilities.maxReferenceVideos, 3) -
          form.referenceVideos.length,
      );
      if (remaining > 0) {
        options.push({
          kind: "referenceVideo",
          label: "参考视频",
          mediaType: "video",
          remaining,
        });
      }
    }
    if (capabilities.supportsReferenceAudios) {
      const remaining = Math.max(
        0,
        parseLimit(capabilities.maxReferenceAudios, 3) -
          form.referenceAudios.length,
      );
      if (remaining > 0) {
        options.push({
          kind: "referenceAudio",
          label: "参考音频",
          mediaType: "audio",
          remaining,
        });
      }
    }
    return options;
  }, [capabilities, form, mode, referenceImageUploadAvailability]);

  const simpleAttachmentDisabledReason = useMemo(() => {
    if (!capabilities) return "请选择支持附件输入的模型";
    const supportsAnyAttachment =
      capabilities.supportsFirstFrame ||
      capabilities.supportsLastFrame ||
      capabilities.supportsReferenceImages ||
      (mode === "video" &&
        (capabilities.supportsReferenceVideos ||
          capabilities.supportsReferenceAudios));
    if (!supportsAnyAttachment) return "当前模型不支持附件输入";
    if (simpleAttachmentOptions.length === 0) return "附件数量已达上限";
    return "";
  }, [capabilities, mode, simpleAttachmentOptions.length]);

  const addSimpleAttachments = (
    urls: string[],
    kind?: SimpleAttachmentKind,
  ) => {
    if (!capabilities || !urls.length) return;
    const imageKind =
      !kind ||
      kind === "firstFrame" ||
      kind === "lastFrame" ||
      kind === "referenceImage";
    if (imageKind && !referenceImageUploadAvailability.supported) {
      toast.error(referenceImageUploadAvailability.reason);
      return;
    }
    setForm((current) => {
      if (kind === "referenceVideo") {
        return {
          ...current,
          referenceVideos: [...current.referenceVideos, ...urls].slice(
            0,
            parseLimit(capabilities.maxReferenceVideos, 3),
          ),
        };
      }
      if (kind === "referenceAudio") {
        return {
          ...current,
          referenceAudios: [...current.referenceAudios, ...urls].slice(
            0,
            parseLimit(capabilities.maxReferenceAudios, 3),
          ),
        };
      }
      if (mode === "image") {
        return {
          ...current,
          referenceImages: [...current.referenceImages, ...urls].slice(
            0,
            maxSimpleImageAttachments,
          ),
        };
      }

      if (kind === "firstFrame") {
        return { ...current, firstFrame: urls.slice(0, 1) };
      }
      if (kind === "lastFrame") {
        return { ...current, lastFrame: urls.slice(0, 1) };
      }
      if (kind === "referenceImage") {
        return {
          ...current,
          referenceImages: [...current.referenceImages, ...urls].slice(
            0,
            parseLimit(capabilities.maxReferenceImages, 1),
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
      referenceVideos: current.referenceVideos.filter((value) => value !== url),
      referenceAudios: current.referenceAudios.filter((value) => value !== url),
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
    const composer =
      composerMode === "advanced"
        ? advancedComposerRef.current
        : simpleComposerRef.current;
    composer?.scrollIntoView({ behavior: "smooth", block: "end" });
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
  const historyBottomInset = advancedMode ? 0 : composerHeight + 16;

  return (
    <div className="relative flex min-h-0 w-full grow basis-0 flex-col overflow-hidden pb-20 lg:pb-3">
      <div className="w-full shrink-0 px-5 lg:px-8">
        <motion.header
          layout
          transition={{
            layout: reduceMotion
              ? { duration: 0 }
              : { duration: 0.26, ease: MODE_TRANSITION_EASE },
          }}
          className={cn(
            "mx-auto flex w-full items-center justify-between gap-4 pb-3",
            advancedMode ? "max-w-none" : "max-w-[1180px]",
          )}
        >
          <motion.div layout="position" className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-2xl border border-border/45 bg-card text-primary shadow-sm">
              <PageIcon className="h-4.5 w-4.5" />
            </span>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {mode === "image" ? "描述想法，生成画面" : "描述场景、动作和镜头"}
              </p>
            </div>
          </motion.div>
          <motion.a
            layout="position"
            href="/settings/ai-models"
            className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}
          >
            <Settings2 className="h-3.5 w-3.5" />
            模型配置
          </motion.a>
        </motion.header>
      </div>

      <div
        data-slot="generation-workspace"
        className={cn("min-h-0 flex-1", advancedMode && "px-5 lg:px-8")}
      >
        <div
          className={cn(
            "relative h-full min-h-0",
            advancedMode &&
              "grid w-full gap-4 overflow-y-auto xl:grid-cols-[minmax(400px,460px)_minmax(0,1fr)] xl:overflow-visible",
          )}
        >
          <AnimatePresence initial={false} mode="popLayout">
            {advancedMode && (
              <motion.div
                key="advanced-composer"
                data-slot="generation-composer-layer"
                initial={reduceMotion ? false : { opacity: 0, x: -14 }}
                animate={{
                  opacity: 1,
                  x: 0,
                  transition: reduceMotion
                    ? { duration: 0 }
                    : { duration: 0.24, ease: MODE_TRANSITION_EASE },
                }}
                exit={{
                  opacity: 0,
                  x: reduceMotion ? 0 : -10,
                  transition: reduceMotion
                    ? { duration: 0 }
                    : { duration: 0.14, ease: "easeIn" },
                }}
                className="min-h-0 xl:h-full"
              >
                <div
                  data-slot="generation-composer"
                  ref={advancedComposerRef}
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
                    referenceImageUploadAvailability={
                      referenceImageUploadAvailability
                    }
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <motion.div
            layout
            transition={{
              layout: reduceMotion
                ? { duration: 0 }
                : { duration: 0.26, ease: MODE_TRANSITION_EASE },
            }}
            className={cn(
              "h-full min-h-0 w-full",
              advancedMode && "max-h-[42vh] xl:max-h-none",
            )}
          >
            <OverlayScrollArea
              data-slot="generation-history-scroll"
              className="h-full min-h-0 w-full"
              viewportStyle={{
                paddingBottom: historyBottomInset,
                scrollPaddingBottom: historyBottomInset,
              }}
            >
              <div
                className={cn(
                  "w-full",
                  advancedMode ? "pr-3" : "px-5 lg:px-8",
                )}
              >
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
                    onCancel={taskId => void cancelTask(taskId)}
                    onDelete={id => void deleteTask(id)}
                    cancellingTaskIds={cancellingTaskIds}
                    onAddAsset={(item, prompt) =>
                      setAssetTarget({ mode, item, prompt })
                    }
                  />
                </div>
              </div>
            </OverlayScrollArea>
          </motion.div>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {!advancedMode && (
          <motion.div
            key="simple-composer"
            data-slot="generation-composer-layer"
            initial={reduceMotion ? false : { opacity: 0, y: 14 }}
            animate={{
              opacity: 1,
              y: 0,
              transition: reduceMotion
                ? { duration: 0 }
                : { duration: 0.24, ease: MODE_TRANSITION_EASE },
            }}
            exit={{
              opacity: 0,
              y: reduceMotion ? 0 : 10,
              transition: reduceMotion
                ? { duration: 0 }
                : { duration: 0.14, ease: "easeIn" },
            }}
            className="pointer-events-none absolute inset-x-0 bottom-20 z-20 lg:bottom-3"
          >
            <div
              aria-hidden="true"
              className="absolute inset-x-0 -top-12 -bottom-20 bg-linear-to-t from-background via-background/35 to-transparent lg:-bottom-3"
            />
            <div className="relative z-10 w-full px-5 lg:px-8">
              <div
                data-slot="generation-composer"
                ref={simpleComposerRef}
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
                  attachmentOptions={simpleAttachmentOptions}
                  maxImageAttachments={maxSimpleImageAttachments}
                  attachmentCapacity={simpleAttachmentCapacity}
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
                  attachmentDisabledReason={simpleAttachmentDisabledReason}
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

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
