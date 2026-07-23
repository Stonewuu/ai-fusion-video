"use client";

import type { AiModel } from "@/lib/api/ai-model";
import type { GenerationCapabilities } from "@/lib/generation-capabilities";
import { GenerationModelPicker } from "./generation-model-picker";
import { GenerationDurationSlider } from "./generation-duration-slider";
import { GenerationQuickSelect } from "./generation-quick-select";
import type { GenerationFormState, WorkbenchMode } from "./generation-types";
import { parseLimit } from "./generation-utils";

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

      <GenerationQuickSelect
        value={form.ratio}
        placeholder="比例"
        options={(capabilities?.supportedAspectRatios || []).map((value) => ({
          value,
          label: `比例 ${value}`,
          menuLabel: value,
        }))}
        onChange={(ratio) => onFormChange({ ratio })}
      />

      <GenerationQuickSelect
        value={form.resolution}
        placeholder="分辨率"
        options={(capabilities?.supportedResolutions || []).map((value) => ({
          value,
          label: value,
        }))}
        contentClassName="min-w-28"
        onChange={(resolution) => onFormChange({ resolution })}
      />

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
          max={capabilities.maxDuration}
          onChange={(duration) => onFormChange({ duration })}
        />
      )}
    </>
  );
}
