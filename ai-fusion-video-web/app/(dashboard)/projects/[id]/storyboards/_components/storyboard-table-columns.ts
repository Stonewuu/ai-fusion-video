type StoryboardTableField =
  | "shotNumber"
  | "imageUrl"
  | "frameReferences"
  | "generatedVideoUrl"
  | "videoPrompt"
  | "assets"
  | "shotType"
  | "duration"
  | "cameraAngle"
  | "cameraMovement"
  | "content"
  | "dialogue"
  | "sound"
  | "remark";

interface ColumnDef {
  label: string;
  field: StoryboardTableField;
  initW: number;
  minW: number;
  isImage?: boolean;
  isVideo?: boolean;
  multiline?: boolean;
  singleLine?: boolean;
}

export const STORYBOARD_TABLE_COLUMNS: ColumnDef[] = [
  { label: "镜号", field: "shotNumber", initW: 80, minW: 72, singleLine: true },
  { label: "画面", field: "imageUrl", initW: 80, minW: 80, isImage: true },
  { label: "首尾帧", field: "frameReferences", initW: 96, minW: 88 },
  { label: "视频", field: "generatedVideoUrl", initW: 80, minW: 80, isVideo: true },
  { label: "视频提示词", field: "videoPrompt", initW: 280, minW: 200, multiline: true },
  { label: "关联资产", field: "assets", initW: 200, minW: 144 },
  { label: "景别", field: "shotType", initW: 88, minW: 80, singleLine: true },
  { label: "时长", field: "duration", initW: 72, minW: 64, singleLine: true },
  { label: "摄像机角度", field: "cameraAngle", initW: 120, minW: 104, singleLine: true },
  { label: "运镜", field: "cameraMovement", initW: 160, minW: 120 },
  { label: "分镜内容", field: "content", initW: 320, minW: 200, multiline: true },
  { label: "对白", field: "dialogue", initW: 280, minW: 200, multiline: true },
  { label: "声音", field: "sound", initW: 160, minW: 120 },
  { label: "备注", field: "remark", initW: 160, minW: 120 },
];

const STORAGE_KEY = "fusion-storyboard-col-widths";
// 旧格式保存了整组宽度，未拖动的列也包含在内。升级这些默认值，保留用户调整过的宽度。
const LEGACY_DEFAULT_WIDTHS = [48, 80, 92, 80, 200, 160, 64, 48, 90, 80, 240, 200, 100, 100];

export function resolveStoryboardColWidths(saved: unknown): number[] {
  const legacy = Array.isArray(saved);
  const widths = legacy
    ? saved
    : saved !== null && typeof saved === "object" && "version" in saved && saved.version === 2 && "widths" in saved
      ? saved.widths
      : undefined;

  return STORYBOARD_TABLE_COLUMNS.map((column, index) => {
    const width = Array.isArray(widths) && widths.length === STORYBOARD_TABLE_COLUMNS.length
      ? widths[index]
      : undefined;
    if (typeof width !== "number" || !Number.isFinite(width) || width <= 0) {
      return column.initW;
    }
    if (legacy && width === LEGACY_DEFAULT_WIDTHS[index]) {
      return column.initW;
    }
    return Math.max(column.minW, width);
  });
}

export function loadSavedColWidths(): number[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return resolveStoryboardColWidths(raw === null ? null : JSON.parse(raw));
  } catch {
    // 浏览器中已有的无效记录或不可用的存储不影响表格显示。
    return resolveStoryboardColWidths(null);
  }
}

export function saveColWidths(widths: number[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 2, widths }));
  } catch (error) {
    console.warn("分镜列宽未能保存", error);
  }
}
