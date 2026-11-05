"use client";

import { ImageOutputPickers } from "@/components/generation/image-output-pickers";
import { Button } from "@/components/ui/button";
import { changeImageOutput, imageOutputCapabilities, imageOutputRequest } from "@/lib/image-output-sizes";
import { canvasImageDefaults, canvasImageOutputIssue, canvasImageSelection } from "@/lib/canvas/image-output";
import type { CanvasCapability, CanvasConfig, CanvasModelOption } from "@/lib/canvas/types";
import { CanvasDimensionsFields } from "./dimensions-fields";

export function CanvasImageOutputFields({ capability, config, model, onChange, disabled }: {
  capability: CanvasCapability;
  config: CanvasConfig;
  model?: CanvasModelOption;
  onChange: (config: CanvasConfig) => void;
  disabled: boolean;
}) {
  if (!model) {
    return <p className="text-xs text-muted-foreground">选择模型后可设置画面比例和分辨率。</p>;
  }
  const options = imageOutputCapabilities(model.parameters);
  const selection = canvasImageSelection(config, model);
  const issue = canvasImageOutputIssue(config, model);
  const size = options.imageSizes?.[selection.resolution]?.[selection.ratio];
  const custom = config.width !== undefined || config.height !== undefined;
  return (
    <section aria-label="图片输出设置" className="flex min-w-0 flex-col gap-4">
      <div className="grid min-w-0 grid-cols-2 gap-3">
        <ImageOutputPickers capabilities={options} form={selection} disabled={disabled} onChange={(patch) => {
          const next = { ...config };
          delete next.width;
          delete next.height;
          delete next.ratio;
          delete next.resolution;
          const output = imageOutputRequest(options, changeImageOutput(options, selection, patch));
          if (output.ratio) {
            next.ratio = output.ratio;
          }
          if (output.resolution) {
            next.resolution = output.resolution;
          }
          onChange(next);
        }} />
      </div>
      {size && !custom && <p className="text-xs text-muted-foreground">画面尺寸：{size.replace(/[x*]/i, " × ")} px</p>}
      {issue && <p role="alert" className="text-xs text-destructive">{issue}</p>}
      {(options.supportCustomSize || custom) && (
        <CanvasDimensionsFields capability={capability} config={config} disabled={disabled} invalid={!!issue} onChange={(next) => {
          delete next.ratio;
          delete next.resolution;
          onChange(next);
        }} />
      )}
      {custom && <Button variant="ghost" disabled={disabled} onClick={() => onChange(canvasImageDefaults(config, model))}>恢复模型默认尺寸</Button>}
    </section>
  );
}
