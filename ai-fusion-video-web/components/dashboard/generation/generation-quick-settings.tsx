"use client";

import type { AiModel } from "@/lib/api/ai-model";
import type { GenerationCapabilities } from "@/lib/generation-capabilities";
import { ImageOutputPickers } from "@/components/generation/image-output-pickers";
import { GenerationModelPicker } from "./generation-model-picker";
import { GenerationDurationSlider } from "./generation-duration-slider";
import { GenerationQuickSelect } from "./generation-quick-select";
import type { GenerationFormState, WorkbenchMode } from "./generation-types";
import { parseLimit } from "./generation-utils";
import { videoDurationOptions, videoMaxDuration } from "@/lib/video-model-rules";
import { GenerationVideoOutputPickers } from "./generation-video-output-pickers";

interface GenerationQuickSettingsProps {
  mode: WorkbenchMode;
  models: AiModel[];
  modelId?: number;
  loadingModels: boolean;
  capabilities: GenerationCapabilities | null;
  form: GenerationFormState;
  onModelChange: (modelId: number) => void;
  onFormChange: (patch: Partial<GenerationFormState>) => void;
}

export function GenerationQuickSettings({
  mode,
  models,
  modelId,
  loadingModels,
  capabilities,
  form,
  onModelChange,
  onFormChange,
}: GenerationQuickSettingsProps) {
  const outputLimit = parseLimit(capabilities?.maxCount || 0, 4);
  return (
    <>
      <GenerationModelPicker
        compact
        models={models}
        modelId={modelId}
        loading={loadingModels}
        onChange={onModelChange}
      />

      {mode === "image" ? (
        <ImageOutputPickers compact capabilities={capabilities} form={form} onChange={onFormChange} />
      ) : (
        <GenerationVideoOutputPickers compact capabilities={capabilities} form={form} onChange={onFormChange} />
      )}

      <GenerationQuickSelect
        value={String(form.count)}
        placeholder="数量"
        options={Array.from({ length: outputLimit }, (_, index) => ({
          value: String(index + 1),
          label: `${index + 1} 个结果`,
        }))}
        contentClassName="min-w-28"
        onChange={(count) => onFormChange({ count: Number(count) })}
      />

      {mode === "video" && capabilities && (
        <GenerationDurationSlider
          compact
          value={form.duration}
          min={capabilities.minDuration}
          max={videoMaxDuration(capabilities, form.resolution)}
          automatic={capabilities.videoRules?.automaticDuration}
          options={videoDurationOptions(capabilities, form.resolution)}
          onChange={(duration) => onFormChange({ duration })}
        />
      )}
    </>
  );
}
