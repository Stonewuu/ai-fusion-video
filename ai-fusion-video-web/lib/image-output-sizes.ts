export type ImageSizeTable = Record<string, Record<string, string>>;

const pixels = /^(\d+)\s*[x×*]\s*(\d+)$/i;

export function imageSizeTable(value: unknown, advertised: string[]): ImageSizeTable {
  const table: ImageSizeTable = Object.create(null);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return table;
  }
  for (const [tier, ratios] of Object.entries(value)) {
    if (!ratios || typeof ratios !== "object" || Array.isArray(ratios)) {
      continue;
    }
    for (const [ratio, size] of Object.entries(ratios)) {
      if (typeof size === "string" && pixels.test(size)
        && (!advertised.length || advertised.includes(tier) || advertised.includes(size))) {
        (table[tier] ??= Object.create(null))[ratio] = size;
      }
    }
  }
  return table;
}

export interface ImageOutputOptions {
  supportedResolutions: string[];
  imageSizes?: ImageSizeTable;
  supportAutoSize?: boolean;
}

export interface ImageOutputCapabilities extends ImageOutputOptions {
  supportedAspectRatios: string[];
  pixelResolutionTiers?: string[];
  defaultAspectRatio?: string;
  defaultResolution?: string;
  supportCustomSize?: boolean;
  sizeMultiple?: number;
  minPixels?: number;
  maxPixels?: number;
  maxEdge?: number;
  maxAspectRatio?: number;
}

export interface ImageOutputSelection {
  ratio: string;
  resolution: string;
}

export function resolutionChoices(options: ImageOutputOptions | null | undefined, ratio: string): string[] {
  if (!options) {
    return [];
  }
  if (options.supportAutoSize && ratio === "auto") {
    return options.supportedResolutions.includes("auto") ? ["auto"] : [];
  }
  const parts = /^(\d+):(\d+)$/.exec(ratio);
  return options.supportedResolutions.filter((value) => {
    if (options.supportAutoSize && value === "auto") {
      return false;
    }
    const sizes = options.imageSizes?.[value];
    if (sizes) {
      return !ratio || Object.hasOwn(sizes, ratio);
    }
    const size = options.imageSizes && pixels.exec(value);
    return !size || !parts || Math.abs(+size[1] * +parts[2] - +size[2] * +parts[1]) <= 8 * Math.max(+parts[1], +parts[2]);
  });
}

const textList = (value: unknown): string[] => Array.isArray(value)
  ? value.filter((item): item is string => typeof item === "string" && item.length > 0) : [];

export function imageOutputCapabilities(config: Record<string, unknown>): ImageOutputCapabilities {
  const output = imageOutputOptions(config, textList(config.supportedResolutions));
  const declared = textList(config.supportedAspectRatios);
  const ratios = declared.length ? declared : [...new Set(Object.values(output.imageSizes).flatMap(Object.keys))];
  const constraints: Partial<ImageOutputCapabilities> = {};
  for (const key of ["sizeMultiple", "minPixels", "maxPixels", "maxEdge", "maxAspectRatio"] as const) {
    if (typeof config[key] === "number") {
      constraints[key] = config[key];
    }
  }
  return {
    ...output,
    ...constraints,
    supportedAspectRatios: output.supportedResolutions.length
      ? ratios.filter((ratio) => resolutionChoices(output, ratio).length > 0) : ratios,
    defaultAspectRatio: typeof config.defaultAspectRatio === "string" ? config.defaultAspectRatio : undefined,
    defaultResolution: typeof config.defaultResolution === "string" ? config.defaultResolution : undefined,
    supportCustomSize: config.supportCustomSize === true,
  };
}

export function imageOutputDefaults(options: ImageOutputCapabilities): ImageOutputSelection {
  const ratio = options.defaultAspectRatio && options.supportedAspectRatios.includes(options.defaultAspectRatio)
    ? options.defaultAspectRatio : options.supportedAspectRatios[0] ?? "";
  const choices = resolutionChoices(options, ratio);
  const preferred = imageOutputSelection(options, { ratio, resolution: options.defaultResolution ?? "" }).resolution;
  return { ratio, resolution: choices.includes(preferred) ? preferred : choices[0] ?? "" };
}

/** 将按像素提交的模型参数还原成界面档位，保存时仍使用模型声明的格式。 */
export function imageOutputSelection(options: ImageOutputCapabilities, selection: ImageOutputSelection): ImageOutputSelection {
  const tier = options.pixelResolutionTiers?.find((value) => options.imageSizes?.[value]?.[selection.ratio] === selection.resolution);
  return { ...selection, resolution: tier ?? selection.resolution };
}

export function changeImageOutput(options: ImageOutputCapabilities, current: ImageOutputSelection,
  patch: Partial<ImageOutputSelection>): ImageOutputSelection {
  const next = { ...current, ...patch };
  if (patch.ratio !== undefined) {
    const choices = resolutionChoices(options, next.ratio);
    if (!choices.includes(next.resolution)) {
      next.resolution = choices[0] ?? "";
    }
  }
  return next;
}

export function imageOutputRequest(options: ImageOutputCapabilities, selection: ImageOutputSelection) {
  return {
    ratio: selection.ratio || undefined,
    resolution: options.pixelResolutionTiers?.includes(selection.resolution)
      ? options.imageSizes?.[selection.resolution]?.[selection.ratio] : selection.resolution || undefined,
  };
}

export function imageOutputIssue(options: ImageOutputCapabilities,
  value: { ratio?: string; resolution?: string; width?: number; height?: number }): string | null {
  const selection = imageOutputSelection(options, { ratio: value.ratio ?? "", resolution: value.resolution ?? "" });
  const custom = value.width !== undefined || value.height !== undefined;
  if (!custom && value.ratio === "" && options.supportedAspectRatios.length) {
    return "请选择画面比例";
  }
  if (!custom && value.resolution === "" && options.supportedResolutions.length) {
    return "请选择当前画面比例支持的分辨率";
  }
  if (selection.ratio && options.supportedAspectRatios.length && !options.supportedAspectRatios.includes(selection.ratio)) {
    return "请选择当前模型支持的画面比例";
  }
  if (selection.resolution && !resolutionChoices(options, selection.ratio).includes(selection.resolution)) {
    return "请选择当前画面比例支持的分辨率";
  }
  if (!custom) {
    return null;
  }
  const { width, height } = value;
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width! <= 0 || height! <= 0) {
    return "请同时填写有效的图片宽度和高度";
  }
  if (selection.ratio === "auto" || selection.resolution === "auto") {
    return "自动尺寸不能同时设置自定义宽高";
  }
  const size = options.imageSizes?.[selection.resolution]?.[selection.ratio] ?? selection.resolution;
  const expected = pixels.exec(size);
  if (expected && (+expected[1] !== width || +expected[2] !== height)) {
    return "自定义宽高与所选分辨率不一致，请选择一种尺寸设置方式";
  }
  if (options.sizeMultiple && (width! % options.sizeMultiple !== 0 || height! % options.sizeMultiple !== 0)) {
    return `图片宽高须为 ${options.sizeMultiple} 的倍数`;
  }
  const total = width! * height!;
  if (options.minPixels && total < options.minPixels || options.maxPixels && total > options.maxPixels) {
    return "图片总像素超出当前模型允许范围";
  }
  if (options.maxEdge && Math.max(width!, height!) > options.maxEdge) {
    return `图片最长边不能超过 ${options.maxEdge} 像素`;
  }
  if (options.maxAspectRatio && Math.max(width!, height!) / Math.min(width!, height!) > options.maxAspectRatio) {
    return "图片长短边比例超出当前模型允许范围";
  }
  const ratio = /^(\d+):(\d+)$/.exec(selection.ratio);
  if (!expected && ratio && Math.abs(width! * +ratio[2] - height! * +ratio[1]) > Math.max(+ratio[1], +ratio[2]) * (options.sizeMultiple ?? 1) / 2) {
    return "图片宽高与画面比例不一致";
  }
  return null;
}

export function resolutionLabel(value: string) {
  if (value === "auto") {
    return "自动";
  }
  if (value === "standard") {
    return "标准";
  }
  const size = pixels.exec(value);
  return size ? `长边 ${Math.max(+size[1], +size[2])} 像素` : value;
}

export function imageOutputOptions(config: Record<string, unknown>, advertised: string[]) {
  const imageSizes = imageSizeTable(config.supportedSizes, advertised);
  const tiers = Object.keys(imageSizes);
  const mappedPixels = new Set(Object.values(imageSizes).flatMap((ratios) => Object.values(ratios)));
  const supportedResolutions = tiers.length
    ? [...new Set([...tiers, ...advertised.filter((value) => !mappedPixels.has(value))])]
    : [...advertised];
  if (config.supportAutoSize === true && !advertised.length) {
    supportedResolutions.push("auto");
  }
  const pixelResolutionTiers = advertised.length ? tiers.filter((tier) => !advertised.includes(tier)) : [];
  return { imageSizes, supportedResolutions, pixelResolutionTiers, supportAutoSize: config.supportAutoSize === true };
}
