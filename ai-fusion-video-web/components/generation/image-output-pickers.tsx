"use client";

import { resolutionChoices, resolutionLabel, type ImageOutputCapabilities, type ImageOutputSelection } from "@/lib/image-output-sizes";
import { AspectRatioPicker, ResolutionPicker } from "./media-output-pickers";

interface Props {
  capabilities: ImageOutputCapabilities | null;
  form: ImageOutputSelection;
  onChange: (patch: Partial<ImageOutputSelection>) => void;
  compact?: boolean;
  disabled?: boolean;
}

export function ImageOutputPickers({ capabilities, form, onChange, compact = false, disabled = false }: Props) {
  const resolutions = resolutionChoices(capabilities, form.ratio);
  return (
    <>
      <AspectRatioPicker value={form.ratio} options={capabilities?.supportedAspectRatios ?? []}
        compact={compact} disabled={disabled} onChange={(ratio) => onChange({ ratio })} />
      <ResolutionPicker value={form.resolution} compact={compact} disabled={disabled}
        options={resolutions.map((value) => ({
          value,
          label: resolutionLabel(value),
          detail: capabilities?.imageSizes?.[value]?.[form.ratio]?.replace(/[x*]/i, " × "),
        }))}
        onChange={(resolution) => onChange({ resolution })} />
    </>
  );
}
