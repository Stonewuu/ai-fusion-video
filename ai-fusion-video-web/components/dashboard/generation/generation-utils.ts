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

export function formatGenerationTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
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
