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

interface OutputOptions {
  supportedResolutions: string[];
  imageSizes?: ImageSizeTable;
  supportAutoSize?: boolean;
}

export function resolutionChoices(options: OutputOptions | null | undefined, ratio: string): string[] {
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
