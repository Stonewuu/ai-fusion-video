"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ImageIcon, Settings2, Video } from "lucide-react";
import { toast } from "sonner";
import { aiModelApi, type AiModel, type ModelPreset } from "@/lib/api/ai-model";
import { resolveGenerationCapabilities, workflowInputIssue } from "@/lib/generation-capabilities";
import {
  videoFormIssue,
  videoInputState,
  videoRatioOptions,
} from "@/lib/video-model-rules";
import { toastApiError } from "@/lib/api/toast-api-error";
import { Button, buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { GenerationAdvancedPanel } from "./generation/generation-advanced-panel";
import { GenerationAssetDialog } from "./generation/generation-asset-dialog";
import { GenerationHistory } from "./generation/generation-history";
import { GenerationSimpleComposer } from "./generation/generation-simple-composer";
import { GenerationWorkspaceLayout } from "./generation/generation-workspace-layout";
import { MODE_TRANSITION_EASE } from "./generation/generation-motion";
import {
  EMPTY_GENERATION_FORM,
  type ComposerMode,
  type GenerationAssetTarget,
  type GenerationFormState,
  type GenerationResultItem,
  type SimpleAttachment,
  type SimpleAttachmentKind,
  type SimpleAttachmentUploadOption,
  type WorkbenchMode,
} from "./generation/generation-types";
import {
  getResultMediaUrl,
  getResultPreviewUrl,
  parseLimit,
  generationFormForModel,
} from "./generation/generation-utils";
import { GenerationSubmissionNotice } from "./generation/generation-submission-notice";
import { useGenerationSubmission } from "./generation/use-generation-submission";
import { useGenerationHistory } from "./generation/use-generation-history";
import { generationSubmitRequest } from "./generation/generation-submit-request";
import { useReferenceImageUploadAvailability } from "./generation/use-reference-image-upload-availability";

interface GenerationWorkbenchProps {
  mode: WorkbenchMode;
}

function clean(values: string[]) {
  return values.map((value) => value.trim()).filter(Boolean);
}

export default function GenerationWorkbench({
  mode,
}: GenerationWorkbenchProps) {
  return <GenerationWorkbenchContent key={mode} mode={mode} />;
}

function GenerationWorkbenchContent({ mode }: GenerationWorkbenchProps) {
  const modelType = mode === "image" ? 2 : 3;
  const [composerMode, setComposerMode] = useState<ComposerMode>("simple");
  const [models, setModels] = useState<AiModel[]>([]);
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [modelId, setModelId] = useState<number | undefined>();
  const [loadingModels, setLoadingModels] = useState(true);
  const [form, setForm] = useState<GenerationFormState>(EMPTY_GENERATION_FORM);
  const submission = useGenerationSubmission(mode);
  const historyState = useGenerationHistory(mode);
  const {
    entries: history,
    total: historyTotal,
    limit: historyLimit,
    setLimit: setHistoryLimit,
    loading: historyLoading,
    cancelling: cancellingTaskIds,
    load: loadHistory,
    cancel: cancelTask,
    remove: deleteTask,
  } = historyState;
  const submitting = submission.busy;
  const [assetTarget, setAssetTarget] = useState<GenerationAssetTarget | null>(
    null,
  );
  const simpleComposerRef = useRef<HTMLDivElement>(null);
  const advancedComposerRef = useRef<HTMLDivElement>(null);
  const [composerHeight, setComposerHeight] = useState(220);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      aiModelApi.listByType(modelType),
      aiModelApi.presets(modelType),
    ])
      .then(([modelList, presetList]) => {
        if (cancelled) {
          return;
        }
        const enabled = modelList.filter((model) => model.status === 1);
        setModels(enabled);
        setPresets(presetList);
        const initial = enabled.find((model) => model.defaultModel);
        setModelId(initial?.id);
        if (initial) {
          setForm((current) =>
            generationFormForModel(
              current,
              resolveGenerationCapabilities(initial, presetList),
            ),
          );
        }
      })
      .catch((error) => {
        if (!cancelled) {
          toastApiError(error, "加载生成模型失败");
          setModels([]);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingModels(false);
        }
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

  const changeModel = (id: number) => {
    const model = models.find((candidate) => candidate.id === id);
    if (!model) {
      toast.error("所选模型已不可用，请刷新模型列表");
      return;
    }
    setModelId(id);
    setForm((current) =>
      generationFormForModel(
        current,
        resolveGenerationCapabilities(model, presets),
      ),
    );
  };

  useEffect(() => {
    if (composerMode !== "simple") {
      return;
    }
    const element = simpleComposerRef.current;
    if (!element) {
      return;
    }

    const updateHeight = () => {
      setComposerHeight(Math.ceil(element.getBoundingClientRect().height));
    };
    updateHeight();

    const observer = new ResizeObserver(updateHeight);
    observer.observe(element);
    return () => observer.disconnect();
  }, [composerMode]);

  const updateForm = (patch: Partial<GenerationFormState>) => {
    setForm((current) => {
      const next = { ...current, ...patch };
      if (mode === "video" && capabilities?.videoRules) {
        const allowed = videoRatioOptions(
          capabilities.supportedAspectRatios,
          capabilities.videoRules,
          videoInputState(next),
        );
        if (allowed.length && !allowed.includes(next.ratio)) {
          next.ratio = allowed[0];
        }
      }
      return next;
    });
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
    if (!capabilities) {
      return 0;
    }
    if (!referenceImageUploadAvailability.supported) {
      return 0;
    }
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
      ? Math.min(
          slots || capabilities.maxImageInputs,
          capabilities.maxImageInputs,
        )
      : slots;
  }, [capabilities, mode, referenceImageUploadAvailability.supported]);

  const simpleAttachmentCapacity = useMemo(() => {
    if (!capabilities) {
      return 0;
    }
    const imageCapacity = maxSimpleImageAttachments;
    if (mode === "image") {
      return imageCapacity;
    }
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

  const simpleAttachmentOptions = useMemo<
    SimpleAttachmentUploadOption[]
  >(() => {
    if (!capabilities) {
      return [];
    }

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
    if (!capabilities) {
      return "请选择支持附件输入的模型";
    }
    const supportsAnyAttachment =
      capabilities.supportsFirstFrame ||
      capabilities.supportsLastFrame ||
      capabilities.supportsReferenceImages ||
      (mode === "video" &&
        (capabilities.supportsReferenceVideos ||
          capabilities.supportsReferenceAudios));
    if (!supportsAnyAttachment) {
      return "当前模型不支持附件输入";
    }
    if (simpleAttachmentOptions.length === 0) {
      return "附件数量已达上限";
    }
    return "";
  }, [capabilities, mode, simpleAttachmentOptions.length]);

  const addSimpleAttachments = (
    urls: string[],
    kind?: SimpleAttachmentKind,
  ) => {
    if (!capabilities || !urls.length) {
      return;
    }
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
        if (value) {
          next.firstFrame = [value];
        }
      }
      if (capabilities.supportsLastFrame && next.lastFrame.length === 0) {
        const value = remaining.shift();
        if (value) {
          next.lastFrame = [value];
        }
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
    if (!modelId || !capabilities) {
      return "请选择生成模型";
    }
    if (capabilities.unavailable) {
      return capabilities.unavailable;
    }
    const workflowIssue = workflowInputIssue(capabilities, {
      referenceImages: clean(form.referenceImages).length, firstFrame: clean(form.firstFrame).length,
      lastFrame: clean(form.lastFrame).length, referenceVideos: clean(form.referenceVideos).length,
      referenceAudios: clean(form.referenceAudios).length,
    });
    if (workflowIssue) {
      return workflowIssue;
    }
    if (mode === "video") {
      const issue = videoFormIssue(capabilities, form);
      if (issue) {
        return issue;
      }
    } else if (!form.prompt.trim()) {
      return "请输入提示词";
    }
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
    if (!modelId) {
      return;
    }

    const originalForm = form;
    const receipt = await submission.submit(
      generationSubmitRequest(mode, form, modelId, capabilities),
      selectedModel!.name,
    );
    if (!receipt) {
      return;
    }
    toast.success("任务已提交，可关闭页面，稍后在创作记录查看结果");
    setForm((current) =>
      current === originalForm
        ? {
            ...current,
            prompt: "",
            referenceImages: [],
            firstFrame: [],
            lastFrame: [],
            referenceVideos: [],
            referenceAudios: [],
          }
        : current,
    );
    setHistoryLimit(10);
    await loadHistory(10);
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
          referenceVideos: [...form.referenceVideos, item.videoUrl].slice(
            0,
            parseLimit(capabilities.maxReferenceVideos, 1),
          ),
        });
        toast.success("已加入参考视频");
        focusComposer();
        return;
      }
      const preview = getResultPreviewUrl(item);
      if (preview) {
        addSimpleAttachments([preview]);
      }
    } else {
      const media = getResultMediaUrl(item) || getResultPreviewUrl(item);
      if (media) {
        addSimpleAttachments([media]);
      }
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
                {mode === "image"
                  ? "描述想法，生成画面"
                  : "描述场景、动作和镜头"}
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
        <GenerationSubmissionNotice
          submission={submission}
          onConfirmed={() => void loadHistory(historyLimit)}
        />
        {capabilities?.unavailable && (
          <p role="alert" className="text-sm text-destructive">{capabilities.unavailable}</p>
        )}
      </div>

      <GenerationWorkspaceLayout
        advancedMode={advancedMode}
        reduceMotion={reduceMotion}
        composerRef={advancedComposerRef}
        historyBottomInset={historyBottomInset}
        composer={
          <GenerationAdvancedPanel
            mode={mode}
            models={models}
            modelId={modelId}
            loadingModels={loadingModels}
            capabilities={capabilities}
            form={form}
            submitting={submitting}
            submitDisabled={!submission.canSubmit || Boolean(capabilities?.unavailable)}
            onModelChange={changeModel}
            onFormChange={updateForm}
            onSubmit={() => void submit()}
            onSimple={() => setComposerMode("simple")}
            referenceImageUploadAvailability={referenceImageUploadAvailability}
          />
        }
      >
        <div className="mb-4 space-y-3">
          {historyState.error && (
            <div
              role="alert"
              className="flex items-center gap-2 text-sm text-destructive"
            >
              <p className="flex-1">{historyState.error}</p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void loadHistory(historyLimit)}
              >
                重新读取记录
              </Button>
            </div>
          )}
        </div>
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
          onCancel={(taskId) => void cancelTask(taskId)}
          cancellingTaskIds={cancellingTaskIds}
          onDelete={deleteTask}
          onAddAsset={(item, prompt) => setAssetTarget({ mode, item, prompt })}
        />
      </GenerationWorkspaceLayout>

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
                  submitDisabled={!submission.canSubmit || Boolean(capabilities?.unavailable)}
                  onModelChange={changeModel}
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
            if (!open) {
              setAssetTarget(null);
            }
          }}
          onAttached={() => setAssetTarget(null)}
        />
      )}
    </div>
  );
}
