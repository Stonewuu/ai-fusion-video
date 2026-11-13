"use client";

import { AspectRatioPicker, ResolutionPicker } from "@/components/generation/media-output-pickers";
import type { GenerationCapabilities } from "@/lib/generation-capabilities";
import { videoDurationForResolution, videoInputState, videoRatioOptions } from "@/lib/video-model-rules";
import type { GenerationFormState } from "./generation-types";

interface Props {
  capabilities: GenerationCapabilities | null;
  form: GenerationFormState;
  compact?: boolean;
  onChange: (patch: Partial<GenerationFormState>) => void;
}

export function GenerationVideoOutputPickers({ capabilities, form, compact = false, onChange }: Props) {
  const ratios = videoRatioOptions(capabilities?.supportedAspectRatios ?? [], capabilities?.videoRules, videoInputState(form));
  return (
    <>
      <AspectRatioPicker value={form.ratio} options={ratios} compact={compact} onChange={(ratio) => onChange({ ratio })} />
      <ResolutionPicker value={form.resolution} compact={compact}
        options={(capabilities?.supportedResolutions ?? []).map((value) => ({ value, label: value }))}
        onChange={(resolution) => {
          const duration = capabilities ? videoDurationForResolution(form.duration, capabilities, resolution) : form.duration;
          onChange({ resolution, duration });
        }} />
    </>
  );
}
