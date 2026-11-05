import {
  imageOutputCapabilities, imageOutputDefaults, imageOutputIssue,
  imageOutputRequest, imageOutputSelection,
} from "../image-output-sizes";
import type { CanvasConfig, CanvasModelOption } from "./types";

export function canvasImageSelection(config: CanvasConfig, model: CanvasModelOption) {
  return imageOutputSelection(imageOutputCapabilities(model.parameters), {
    ratio: String(config.ratio ?? ""), resolution: String(config.resolution ?? ""),
  });
}

export function canvasImageDefaults(config: CanvasConfig, model: CanvasModelOption): CanvasConfig {
  const next = { ...config };
  delete next.width;
  delete next.height;
  delete next.ratio;
  delete next.resolution;
  const options = imageOutputCapabilities(model.parameters);
  const values = imageOutputRequest(options, imageOutputDefaults(options));
  if (values.ratio) {
    next.ratio = values.ratio;
  }
  if (values.resolution) {
    next.resolution = values.resolution;
  }
  return next;
}

export function canvasImageOutputIssue(config: CanvasConfig, model: CanvasModelOption): string | null {
  return imageOutputIssue(imageOutputCapabilities(model.parameters), {
    ratio: typeof config.ratio === "string" ? config.ratio : undefined,
    resolution: typeof config.resolution === "string" ? config.resolution : undefined,
    width: typeof config.width === "number" ? config.width : undefined,
    height: typeof config.height === "number" ? config.height : undefined,
  });
}
