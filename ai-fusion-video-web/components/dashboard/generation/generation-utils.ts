import type {
  GenerationResultItem,
  GenerationTask,
  WorkbenchMode,
} from "./generation-types";

export function parseLimit(limit: number, fallback: number) {
  return limit > 0 ? limit : fallback;
}

export function getResultMediaUrl(item: GenerationResultItem) {
  return "imageUrl" in item ? item.imageUrl : item.videoUrl;
}

export function getResultPreviewUrl(item: GenerationResultItem) {
  if ("imageUrl" in item) return item.thumbnailUrl || item.imageUrl;
  return item.coverUrl || item.firstFrameUrl || item.lastFrameUrl;
}

export function getTaskStatus(task: GenerationTask) {
  if (task.status === 2) return { label: "已完成", tone: "success" as const };
  if (task.status === 3) return { label: "生成失败", tone: "failed" as const };
  if (task.status === 1) return { label: "生成中", tone: "running" as const };
  return { label: "排队中", tone: "queued" as const };
}

export function isGenerationTaskActive(task: GenerationTask) {
  return task.status === 0 || task.status === 1;
}

export function getGenerationErrorSummary(message: string, maxLength = 160) {
  const normalized = message.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength).trimEnd()}…`;
}

export function formatGenerationDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).format(date);
}

export interface GenerationParameterItem {
  label: string;
  value: string;
}

function normalizedParameter(value: string | null | undefined) {
  const normalized = value?.trim();
  return normalized || undefined;
}

function comparableSize(value: string) {
  return value.toLowerCase().replace(/[×*]/g, "x").replace(/\s+/g, "");
}

function positiveNumber(value: number | null | undefined) {
  return typeof value === "number" && value > 0 ? value : undefined;
}

export function getGenerationParameterItems(
  task: GenerationTask,
  items: GenerationResultItem[],
): GenerationParameterItem[] {
  const parameters: GenerationParameterItem[] = [];
  const ratio = normalizedParameter(task.ratio);
  const resolution = normalizedParameter(task.resolution);

  if (ratio) parameters.push({ label: "比例", value: ratio });
  if (resolution) parameters.push({ label: "分辨率", value: resolution });

  if ("aspectRatio" in task) {
    const aspectRatio = normalizedParameter(task.aspectRatio);
    if (aspectRatio && aspectRatio !== ratio) {
      parameters.push({ label: "宽高比", value: aspectRatio });
    }

    const firstImage = items.find((item) => "imageUrl" in item);
    const width = positiveNumber(task.width)
      || (firstImage && "width" in firstImage ? positiveNumber(firstImage.width) : undefined);
    const height = positiveNumber(task.height)
      || (firstImage && "height" in firstImage ? positiveNumber(firstImage.height) : undefined);
    if (width && height) {
      const size = `${width}×${height}`;
      if (!resolution || comparableSize(resolution) !== comparableSize(size)) {
        parameters.push({ label: "尺寸", value: size });
      }
    }

    if (positiveNumber(task.count)) {
      parameters.push({ label: "数量", value: `${task.count} 张` });
    }
    return parameters;
  }

  const modeLabel = task.generateMode === "image2video"
    ? "图生视频"
    : task.generateMode === "text2video"
      ? "文生视频"
      : normalizedParameter(task.generateMode);
  if (modeLabel) parameters.unshift({ label: "模式", value: modeLabel });

  const firstVideo = items.find((item) => "videoUrl" in item);
  const duration = positiveNumber(task.duration)
    || (firstVideo && "duration" in firstVideo ? positiveNumber(firstVideo.duration) : undefined);
  if (duration) parameters.push({ label: "时长", value: `${duration} 秒` });
  if (positiveNumber(task.count)) {
    parameters.push({ label: "数量", value: `${task.count} 段` });
  }
  if (task.generateAudio === true) parameters.push({ label: "声音", value: "开启" });
  if (task.cameraFixed === true) parameters.push({ label: "镜头", value: "固定" });
  if (task.watermark === true) parameters.push({ label: "水印", value: "开启" });
  if (task.seed !== null && task.seed !== undefined) {
    parameters.push({ label: "Seed", value: String(task.seed) });
  }
  return parameters;
}

export function generationSuggestions(mode: WorkbenchMode) {
  return mode === "image"
    ? [
        "雨夜霓虹街头，电影感人像，湿润路面反光",
        "清晨山谷中的现代木屋，薄雾与柔和自然光",
        "复古科幻风机械鸟，产品摄影，精细金属质感",
      ]
    : [
        "镜头缓慢推进，人物穿过雨夜街道，霓虹倒影流动",
        "航拍越过云海与山脊，晨光逐渐照亮远处城市",
        "静物产品旋转展示，柔和棚拍光线，镜头稳定",
      ];
}
