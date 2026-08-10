"use client";

import { Film, GripVertical, Plus, Trash2, Video, Play, ZoomIn, X } from "lucide-react";
import { VideoPreviewDialog } from "@/components/dashboard/video-preview-dialog";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import type { StoryboardFrameType, StoryboardItem } from "@/lib/api/storyboard";
import { EditableCell } from "./editable-cell";
import { useState, useRef, useCallback, useEffect } from "react";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { SafeImage } from "@/components/ui/safe-image";

const TOOLTIP_CONTENT_CLASS = "flex flex-col gap-0.5 items-start text-left max-w-[220px] px-2.5 py-1.5 rounded-lg text-[11px] bg-white/85 dark:bg-zinc-900/85 backdrop-blur-md border border-zinc-200/50 dark:border-zinc-800/50 text-zinc-900 dark:text-zinc-50 shadow-lg [&_.bg-foreground]:bg-white/85 [&_.fill-foreground]:fill-white/85 dark:[&_.bg-foreground]:bg-zinc-900/85 dark:[&_.fill-foreground]:fill-zinc-900/85";


/** 安全解析 ID 数组 */
function parseIds(raw: number[] | string | null | undefined): number[] {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
}

/** 格式化资产显示名称：子变体名 (主资产名) */
function getAssetDisplayName(subItemName: string | null | undefined, parentAssetName: string) {
  const subName = subItemName?.trim();
  const parentName = parentAssetName.trim();
  if (!subName || subName === "默认" || subName === "默认变体" || subName === "初始" || subName === parentName) {
    return parentName;
  }
  return `${subName} (${parentName})`;
}

/** 列定义 */
type StoryboardTableField =
  | "shotNumber"
  | "imageUrl"
  | "frameReferences"
  | "generatedVideoUrl"
  | "videoPrompt"
  | "shotType"
  | "duration"
  | "cameraAngle"
  | "cameraMovement"
  | "content"
  | "dialogue"
  | "sound"
  | "remark"
  | "assets";

interface ColumnDef {
  label: string;
  field: StoryboardTableField;
  initW: number;
  minW: number;
  isImage?: boolean;
  isVideo?: boolean;
  multiline?: boolean;
}

const COLUMNS: ColumnDef[] = [
  { label: "镜号", field: "shotNumber", initW: 48, minW: 40 },
  { label: "画面", field: "imageUrl", initW: 80, minW: 60, isImage: true },
  { label: "首尾帧", field: "frameReferences", initW: 92, minW: 82 },
  { label: "视频", field: "generatedVideoUrl", initW: 80, minW: 60, isVideo: true },
  { label: "视频提示词", field: "videoPrompt", initW: 200, minW: 80, multiline: true },
  { label: "关联资产", field: "assets", initW: 160, minW: 100 },
  { label: "景别", field: "shotType", initW: 64, minW: 50 },
  { label: "时长", field: "duration", initW: 48, minW: 40 },
  { label: "摄像机角度", field: "cameraAngle", initW: 90, minW: 60 },
  { label: "运镜", field: "cameraMovement", initW: 80, minW: 50 },
  {
    label: "分镜内容",
    field: "content",
    initW: 240,
    minW: 100,
    multiline: true,
  },
  { label: "对白", field: "dialogue", initW: 200, minW: 80, multiline: true },
  { label: "声音", field: "sound", initW: 100, minW: 60 },
  { label: "备注", field: "remark", initW: 100, minW: 60 },
];

/** 固定列宽 */
const DRAG_COL_W = 28;
const ACTION_COL_W = 56;

/** localStorage key for persisting column widths */
const COL_WIDTHS_STORAGE_KEY = "fusion-storyboard-col-widths";

/** 从 localStorage 读取保存的列宽（列数变化时自动 fallback） */
function loadSavedColWidths(): number[] {
  try {
    const raw = localStorage.getItem(COL_WIDTHS_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length === COLUMNS.length) {
        return parsed;
      }
    }
  } catch {
    // ignore
  }
  return COLUMNS.map((c) => c.initW);
}

/** 保存列宽到 localStorage */
function saveColWidths(widths: number[]) {
  try {
    localStorage.setItem(COL_WIDTHS_STORAGE_KEY, JSON.stringify(widths));
  } catch {
    // ignore
  }
}

export function StoryboardTableView({
  items,
  selectedItemId,
  onSelectItem,
  onUpdateItemField,
  onAddItem,
  onDeleteItem,
  onReorderItems,
  onVideoGen,
  onOpenFrameDialog,
  assetLookup = {},
  onEditAssets,
}: {
  items: StoryboardItem[];
  selectedItemId: number | null;
  onSelectItem: (id: number | null) => void;
  onUpdateItemField: (itemId: number, field: string, value: string | number | null) => void;
  onAddItem: () => void;
  onDeleteItem: (id: number) => void;
  onReorderItems?: (reorderedItems: StoryboardItem[]) => void;
  onVideoGen?: (itemId: number) => void;
  onOpenFrameDialog?: (item: StoryboardItem, frameType: StoryboardFrameType) => void;
  assetLookup?: Record<
    number,
    {
      item: import("@/lib/api/asset").AssetItem;
      asset: import("@/lib/api/asset").Asset;
    }
  >;
  onEditAssets?: (item: StoryboardItem) => void;
}) {
  const [colWidths, setColWidths] = useState<number[]>(loadSavedColWidths);
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string | null>(null);
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);
  const [previewImageTitle, setPreviewImageTitle] = useState<string>("");

  // ========== 行拖拽排序 ==========
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [overIdx, setOverIdx] = useState<number | null>(null);
  const dragNodeRef = useRef<HTMLDivElement | null>(null);

  const handleDragStart = useCallback(
    (e: React.DragEvent<HTMLDivElement>, idx: number) => {
      setDragIdx(idx);
      dragNodeRef.current = e.currentTarget;
      e.dataTransfer.effectAllowed = "move";
      requestAnimationFrame(() => {
        if (dragNodeRef.current) {
          dragNodeRef.current.style.opacity = "0.4";
        }
      });
    },
    []
  );

  const handleDragEnd = useCallback(() => {
    if (dragNodeRef.current) {
      dragNodeRef.current.style.opacity = "1";
    }
    if (dragIdx !== null && overIdx !== null && dragIdx !== overIdx) {
      const reordered = [...items];
      const [moved] = reordered.splice(dragIdx, 1);
      reordered.splice(overIdx, 0, moved);
      onReorderItems?.(reordered);
    }
    setDragIdx(null);
    setOverIdx(null);
    dragNodeRef.current = null;
  }, [dragIdx, overIdx, items, onReorderItems]);

  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>, idx: number) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      if (dragIdx !== null && idx !== overIdx) {
        setOverIdx(idx);
      }
    },
    [dragIdx, overIdx]
  );

  // ========== 列宽拖拽（直接 DOM 操作避免重渲染） ==========
  const [resizingCol, setResizingCol] = useState<number | null>(null);
  const tableRootRef = useRef<HTMLDivElement>(null);
  const headerScrollRef = useRef<HTMLDivElement>(null);
  const bodyScrollRef = useRef<HTMLDivElement>(null);
  const headerContentRef = useRef<HTMLDivElement>(null);
  const bodyContentRef = useRef<HTMLDivElement>(null);
  const isSyncingHorizontalScrollRef = useRef(false);
  const colWidthsRef = useRef(colWidths);

  useEffect(() => {
    colWidthsRef.current = colWidths;
  }, [colWidths]);

  /** 构建 grid-template-columns 字符串 */
  const buildGridTemplate = useCallback(
    (widths: number[]) =>
      `${DRAG_COL_W}px ${widths.map((w) => `${w}px`).join(" ")} ${ACTION_COL_W}px`,
    []
  );

  /** 根据列宽数组直接更新所有 grid 行的 gridTemplateColumns */
  const applyGridTemplate = useCallback(
    (widths: number[]) => {
      if (!tableRootRef.current) return;
      const tpl = buildGridTemplate(widths);
      const totalW = DRAG_COL_W + widths.reduce((s, w) => s + w, 0) + ACTION_COL_W;

      // 同步表头和表体的内容宽度
      [headerContentRef.current, bodyContentRef.current].forEach((content) => {
        if (!content) return;
        content.style.width = `${totalW}px`;
        content.style.minWidth = `${totalW}px`;
      });

      // 更新所有 grid 行
      const gridRows = tableRootRef.current.querySelectorAll<HTMLElement>("[data-grid-row]");
      gridRows.forEach((row) => {
        row.style.gridTemplateColumns = tpl;
      });
    },
    [buildGridTemplate]
  );

  /** 同步表头和表体的横向滚动位置 */
  const handleHorizontalScroll = useCallback((source: "header" | "body") => {
    if (isSyncingHorizontalScrollRef.current) return;

    const sourceEl = source === "header" ? headerScrollRef.current : bodyScrollRef.current;
    const targetEl = source === "header" ? bodyScrollRef.current : headerScrollRef.current;
    if (!sourceEl || !targetEl) return;

    isSyncingHorizontalScrollRef.current = true;
    targetEl.scrollLeft = sourceEl.scrollLeft;
    requestAnimationFrame(() => {
      isSyncingHorizontalScrollRef.current = false;
    });
  }, []);

  const handleResizeStart = useCallback(
    (colIdx: number, e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setResizingCol(colIdx);

      const startX = e.clientX;
      const startW = colWidthsRef.current[colIdx];
      const minW = COLUMNS[colIdx].minW;
      const liveWidths = [...colWidthsRef.current];

      const onMove = (ev: MouseEvent) => {
        const delta = ev.clientX - startX;
        liveWidths[colIdx] = Math.max(minW, startW + delta);
        applyGridTemplate(liveWidths);
      };

      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        setResizingCol(null);
        // 最终提交到 state 并持久化
        const finalWidths = [...liveWidths];
        setColWidths(finalWidths);
        saveColWidths(finalWidths);
      };

      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [applyGridTemplate]
  );

  // CSS Grid 模板
  const gridTemplate = buildGridTemplate(colWidths);
  const totalWidth =
    DRAG_COL_W + colWidths.reduce((s, w) => s + w, 0) + ACTION_COL_W;

  // ========== 空状态 ==========
  if (items.length === 0) {
    return (
      <div
        onClick={onAddItem}
        className={cn(
          "rounded-2xl border-2 border-dashed border-primary/20 p-16",
          "flex flex-col items-center justify-center text-center",
          "bg-linear-to-br from-primary/5 to-transparent hover:border-primary/40 hover:from-primary/10 transition-all duration-300 cursor-pointer shadow-sm"
        )}
      >
        <div className="h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center mb-4 text-primary transition-transform group-hover:scale-110">
          <Film className="h-6 w-6" />
        </div>
        <h3 className="text-sm font-semibold text-foreground mb-1">暂无镜头</h3>
        <p className="text-xs text-muted-foreground/80">点击添加该场次的第一个镜头</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* 表格容器：表头独立吸顶，表体负责横向滚动 */}
      <div ref={tableRootRef} className="rounded-xl border border-border/20 bg-card/30 backdrop-blur-sm">
        <div className="sticky -top-5 z-20 rounded-t-xl bg-background/95 shadow-sm backdrop-blur-sm">
          <div
            ref={headerScrollRef}
            onScroll={() => handleHorizontalScroll("header")}
            className="overflow-x-auto overflow-y-hidden [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          >
            <div
              ref={headerContentRef}
              style={{ width: totalWidth, minWidth: totalWidth }}
            >
              {/* ===== 表头 ===== */}
              <div
                className={cn(
                  "grid border-b border-border/30",
                  "bg-background/95"
                )}
                data-grid-row
                style={{ gridTemplateColumns: gridTemplate }}
              >
                {/* 拖拽占位 */}
                <div className="px-1 py-2.5" />

                {/* 数据列表头 */}
                {COLUMNS.map((col, i) => (
                  <div
                    key={col.field}
                    className="relative px-2 py-2.5 flex items-center justify-center text-[10px] font-semibold text-muted-foreground uppercase tracking-wider whitespace-nowrap select-none"
                  >
                    {col.label}
                    {/* 列宽拖拽手柄 */}
                    <div
                      onMouseDown={(e) => handleResizeStart(i, e)}
                      className={cn(
                        "absolute top-0 -right-[3px] w-[6px] h-full cursor-col-resize z-10",
                        "group/resize"
                      )}
                    >
                      <div
                        className={cn(
                          "absolute inset-y-1.5 left-1/2 -translate-x-1/2 w-[2px] rounded-full transition-colors",
                          resizingCol === i
                            ? "bg-primary/70"
                            : "bg-border/40 group-hover/resize:bg-primary/50"
                        )}
                      />
                    </div>
                  </div>
                ))}

                {/* 操作占位 */}
                <div className="px-1 py-2.5" />
              </div>
            </div>
          </div>
        </div>

        <div
          ref={bodyScrollRef}
          onScroll={() => handleHorizontalScroll("body")}
          className="overflow-x-auto"
        >
          <div
            ref={bodyContentRef}
            style={{ width: totalWidth, minWidth: totalWidth }}
          >
            {/* ===== 数据行 ===== */}
            {items.map((item, idx) => {
            const isDragging = dragIdx === idx;
            const isOver =
              overIdx === idx && dragIdx !== null && dragIdx !== idx;
            const showTopLine = isOver && dragIdx !== null && dragIdx > idx;
            const showBottomLine = isOver && dragIdx !== null && dragIdx < idx;

            return (
              <div
                key={item.id}
                draggable
                onDragStart={(e) => handleDragStart(e, idx)}
                onDragEnd={handleDragEnd}
                onDragOver={(e) => handleDragOver(e, idx)}
                onClick={() => onSelectItem(item.id)}
                className={cn(
                  "grid border-b border-border/10 last:border-b-0",
                  "transition-colors group cursor-pointer",
                  selectedItemId === item.id
                    ? "bg-primary/5 hover:bg-primary/8"
                    : "hover:bg-primary/3",
                  isDragging && "opacity-40"
                )}
                data-grid-row
                style={{
                  gridTemplateColumns: gridTemplate,
                  boxShadow: showTopLine
                    ? "0 -2px 0 0 hsl(var(--primary) / 0.6)"
                    : showBottomLine
                    ? "0 2px 0 0 hsl(var(--primary) / 0.6)"
                    : undefined,
                }}
              >
                {/* 行拖拽手柄 */}
                <div className="px-1 py-2 flex items-center justify-center">
                  <GripVertical className="h-3.5 w-3.5 text-muted-foreground/60 group-hover:text-muted-foreground/90 cursor-grab active:cursor-grabbing" />
                </div>

                {/* 数据列 */}
                {COLUMNS.map((col) => {
                  const framePreviewUrl = item.firstFrameImageUrl || item.generatedImageUrl || item.imageUrl || item.referenceImageUrl;
                  return (
                    <div
                      key={col.field}
                      className="px-2 py-2 flex items-center justify-center min-w-0 break-all"
                    >
                      {col.isImage ? (
                      <div
                        onClick={(e) => {
                          if (framePreviewUrl) {
                            e.stopPropagation();
                            setPreviewImageUrl(framePreviewUrl);
                            setPreviewImageTitle(`镜头 #${item.shotNumber || item.autoShotNumber || ""} 画面`);
                          }
                        }}
                        className={cn(
                          "flex items-center justify-center h-11 w-16 rounded-md bg-muted/20 border border-border/10 overflow-hidden shrink-0 relative group/img",
                          framePreviewUrl && "cursor-zoom-in hover:border-primary/40 transition-colors"
                        )}
                      >
                        <SafeImage
                          src={resolveMediaUrl(framePreviewUrl)}
                          alt="画面"
                          fallbackType="image"
                          className="w-full h-full object-cover transition-transform group-hover/img:scale-105"
                        />
                        {framePreviewUrl && (
                          <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/25 flex items-center justify-center opacity-0 group-hover/img:opacity-100 transition-all pointer-events-none">
                            <ZoomIn className="h-3.5 w-3.5 text-white/90" />
                          </div>
                        )}
                      </div>
                    ) : col.field === "frameReferences" ? (
                      <div className="flex items-center justify-center gap-1">
                        {(["first", "last"] as StoryboardFrameType[]).map((frameType) => {
                          const label = frameType === "first" ? "首" : "尾";
                          const title = frameType === "first" ? "首帧参考图" : "尾帧参考图";
                          const hasFrame =
                            frameType === "first"
                              ? !!item.firstFrameImageUrl
                              : !!item.lastFrameImageUrl;
                          return (
                            <Tooltip key={frameType}>
                              <TooltipTrigger
                                render={
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      onSelectItem(item.id);
                                      onOpenFrameDialog?.(item, frameType);
                                    }}
                                    disabled={!onOpenFrameDialog}
                                    className={cn(
                                      "h-7 w-8 rounded-md border text-[10px] font-semibold transition-all",
                                      "flex items-center justify-center disabled:opacity-45 disabled:cursor-not-allowed",
                                      hasFrame
                                        ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                                        : "border-border/30 bg-muted/20 text-muted-foreground hover:border-primary/40 hover:text-primary hover:bg-primary/5"
                                    )}
                                  >
                                    {label}
                                  </button>
                                }
                              />
                              <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                                {title}：{hasFrame ? "已设置" : "未设置"}
                              </TooltipContent>
                            </Tooltip>
                          );
                        })}
                      </div>
                    ) : col.isVideo ? (
                      <div className="flex items-center justify-center h-11 w-16 rounded-md bg-muted/20 border border-border/10 overflow-hidden shrink-0 relative group/video">
                        {(item.generatedVideoUrl || item.videoUrl) ? (
                          <>
                            <video
                              src={resolveMediaUrl(item.generatedVideoUrl || item.videoUrl) || ""}
                              className="w-full h-full object-cover"
                              muted
                              preload="metadata"
                              playsInline
                            />
                            <Tooltip>
                              <TooltipTrigger
                                render={
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      onSelectItem(item.id);
                                      const rawVideoUrl = item.generatedVideoUrl || item.videoUrl;
                                      if (rawVideoUrl) {
                                        setPreviewVideoUrl(rawVideoUrl);
                                      }
                                    }}
                                    className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover/video:bg-black/40 transition-colors"
                                  >
                                    <Play className="h-3.5 w-3.5 text-white/90 fill-white/90" />
                                  </button>
                                }
                              />
                              <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                                预览视频
                              </TooltipContent>
                            </Tooltip>
                          </>
                        ) : (
                          <Video className="h-3.5 w-3.5 text-muted-foreground/30" />
                        )}
                      </div>
                    ) : col.field === "assets" ? (
                      <div
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectItem(item.id);
                          onEditAssets?.(item);
                        }}
                        className="group/cell w-full h-full min-h-11 flex items-center justify-center px-1.5 py-1.5 rounded-lg border border-transparent hover:border-primary/20 hover:bg-primary/5 transition-all duration-200 cursor-pointer"
                      >
                        {(() => {
                          const charIds = parseIds(item.characterIds);
                          const sceneId = item.sceneAssetItemId;
                          const propIds = parseIds(item.propIds);

                          const charItems = charIds.map((id) => assetLookup[id]).filter(Boolean);
                          const sceneItem = sceneId ? assetLookup[sceneId] : null;
                          const propItems = propIds.map((id) => assetLookup[id]).filter(Boolean);

                          const hasAssets = charItems.length > 0 || !!sceneItem || propItems.length > 0;

                          if (!hasAssets) {
                            return (
                              <span className="text-[10px] text-muted-foreground/35 group-hover/cell:text-primary transition-colors flex items-center gap-1 font-medium select-none">
                                + 关联资产
                              </span>
                            );
                          }

                          return (
                            <div className="flex items-center justify-center w-full py-1">
                              <div className="flex items-center justify-start gap-2.5 max-w-full overflow-x-auto flex-nowrap [&::-webkit-scrollbar]:h-1 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-muted-foreground/20 [&::-webkit-scrollbar-thumb]:rounded-full hover:[&::-webkit-scrollbar-thumb]:bg-muted-foreground/35 pb-1.5 -mb-0.5 px-1.5">
                                {/* 角色 */}
                                {charItems.map((ci, idx) => (
                                  <Tooltip key={`char-${idx}`}>
                                    <TooltipTrigger
                                      render={
                                        <div className="flex flex-col items-center gap-1 group/asset cursor-pointer select-none shrink-0">
                                          <div className="relative">
                                            <SafeImage
                                              src={resolveMediaUrl(ci.item.imageUrl)}
                                              alt="avatar"
                                              fallbackType="avatar"
                                              onClick={(e) => {
                                                if (ci.item.imageUrl) {
                                                  e.stopPropagation();
                                                  setPreviewImageUrl(ci.item.imageUrl);
                                                  setPreviewImageTitle(`角色: ${getAssetDisplayName(ci.item.name, ci.asset.name)}`);
                                                }
                                              }}
                                              className="h-8 w-8 rounded-full object-cover cursor-zoom-in hover:scale-110 active:scale-95 hover:shadow-md transition-all duration-200 border border-border/40 shrink-0"
                                            />
                                          </div>
                                          <span className="text-[10px] font-semibold text-blue-500 dark:text-blue-400 truncate max-w-[56px] leading-tight text-center mt-0.5">
                                            {getAssetDisplayName(ci.item.name, ci.asset.name)}
                                          </span>
                                        </div>
                                      }
                                    />
                                    <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                                      <span className="font-semibold">角色: {getAssetDisplayName(ci.item.name, ci.asset.name)}</span>
                                      {ci.asset.description && (
                                        <span className="text-[10px] opacity-80 leading-normal break-words mt-0.5">
                                          {ci.asset.description}
                                        </span>
                                      )}
                                    </TooltipContent>
                                  </Tooltip>
                                ))}
                                {/* 场景 */}
                                {sceneItem && (
                                  <Tooltip>
                                    <TooltipTrigger
                                      render={
                                        <div className="flex flex-col items-center gap-1 group/asset cursor-pointer select-none shrink-0">
                                          <div className="relative">
                                            <SafeImage
                                              src={resolveMediaUrl(sceneItem.item.imageUrl)}
                                              alt="scene"
                                              fallbackType="scene"
                                              onClick={(e) => {
                                                if (sceneItem.item.imageUrl) {
                                                  e.stopPropagation();
                                                  setPreviewImageUrl(sceneItem.item.imageUrl);
                                                  setPreviewImageTitle(`场景: ${getAssetDisplayName(sceneItem.item.name, sceneItem.asset.name)}`);
                                                }
                                              }}
                                              className="h-8 w-8 rounded-lg object-cover cursor-zoom-in hover:scale-110 active:scale-95 hover:shadow-md transition-all duration-200 border border-border/40 shrink-0"
                                            />
                                          </div>
                                          <span className="text-[10px] font-semibold text-green-500 dark:text-green-400 truncate max-w-[56px] leading-tight text-center mt-0.5">
                                            {getAssetDisplayName(sceneItem.item.name, sceneItem.asset.name)}
                                          </span>
                                        </div>
                                      }
                                    />
                                    <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                                      <span className="font-semibold">场景: {getAssetDisplayName(sceneItem.item.name, sceneItem.asset.name)}</span>
                                      {sceneItem.asset.description && (
                                        <span className="text-[10px] opacity-80 leading-normal break-words mt-0.5">
                                          {sceneItem.asset.description}
                                        </span>
                                      )}
                                    </TooltipContent>
                                  </Tooltip>
                                )}
                                {/* 道具 */}
                                {propItems.map((pi, idx) => (
                                  <Tooltip key={`prop-${idx}`}>
                                    <TooltipTrigger
                                      render={
                                        <div className="flex flex-col items-center gap-1 group/asset cursor-pointer select-none shrink-0">
                                          <div className="relative">
                                            <SafeImage
                                              src={resolveMediaUrl(pi.item.imageUrl)}
                                              alt="prop"
                                              fallbackType="prop"
                                              onClick={(e) => {
                                                if (pi.item.imageUrl) {
                                                  e.stopPropagation();
                                                  setPreviewImageUrl(pi.item.imageUrl);
                                                  setPreviewImageTitle(`道具: ${getAssetDisplayName(pi.item.name, pi.asset.name)}`);
                                                }
                                              }}
                                              className="h-8 w-8 rounded-lg object-cover cursor-zoom-in hover:scale-110 active:scale-95 hover:shadow-md transition-all duration-200 border border-border/40 shrink-0"
                                            />
                                          </div>
                                          <span className="text-[10px] font-semibold text-amber-500 dark:text-amber-400 truncate max-w-[56px] leading-tight text-center mt-0.5">
                                            {getAssetDisplayName(pi.item.name, pi.asset.name)}
                                          </span>
                                        </div>
                                      }
                                    />
                                    <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                                      <span className="font-semibold">道具: {getAssetDisplayName(pi.item.name, pi.asset.name)}</span>
                                      {pi.asset.description && (
                                        <span className="text-[10px] opacity-80 leading-normal break-words mt-0.5">
                                          {pi.asset.description}
                                        </span>
                                      )}
                                    </TooltipContent>
                                  </Tooltip>
                                ))}
 
                                {/* 右侧内联加号 */}
                                <Tooltip>
                                  <TooltipTrigger
                                    render={
                                      <div className="flex flex-col items-center gap-1 group/add cursor-pointer select-none opacity-40 group-hover/cell:opacity-100 transition-opacity duration-200 ml-1 shrink-0">
                                        <div className="h-8 w-8 rounded-full border border-dashed border-muted-foreground/35 bg-muted/5 group-hover/add:bg-primary/5 group-hover/add:border-primary/50 group-hover/add:text-primary flex items-center justify-center text-muted-foreground/45 transition-all duration-200 shrink-0">
                                          <Plus className="h-4 w-4" />
                                        </div>
                                        <span className="text-[10px] font-semibold text-muted-foreground/60 group-hover/add:text-primary transition-colors leading-tight text-center mt-0.5">
                                          添加
                                        </span>
                                      </div>
                                    }
                                  />
                                  <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                                    添加或编辑关联资产
                                  </TooltipContent>
                                </Tooltip>
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    ) : (
                      <EditableCell
                        value={
                          col.field === "shotNumber"
                            ? item.shotNumber || String(idx + 1)
                            : col.field === "duration"
                            ? item.duration
                              ? String(item.duration)
                              : ""
                            : (item[col.field as keyof StoryboardItem] as string) ||
                              ""
                        }
                        placeholder={
                          col.field === "shotNumber" ? String(idx + 1) : ""
                        }
                        onSave={(val) =>
                          onUpdateItemField(item.id, col.field, val)
                        }
                        onCellClick={() => onSelectItem(item.id)}
                        className={cn(
                          "text-[11px] w-full",
                          col.field === "shotNumber"
                            ? "font-mono text-muted-foreground text-center"
                            : col.field === "content"
                            ? ""
                            : "text-muted-foreground text-center"
                        )}
                        multiline={col.multiline}
                      />
                      )}
                    </div>
                  );
                })}

                {/* 操作按钮 */}
                <div className="px-1 py-2 flex items-center justify-center gap-0.5">
                  {/* 生成视频 */}
                  {onVideoGen && (
                    <Tooltip>
                      <TooltipTrigger
                        render={
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onVideoGen(item.id);
                            }}
                            className="p-1 rounded opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-purple-400 hover:bg-purple-500/10 transition-all"
                          >
                            <Video className="h-3 w-3" />
                          </button>
                        }
                      />
                      <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                        生成视频
                      </TooltipContent>
                    </Tooltip>
                  )}
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteItem(item.id);
                          }}
                          className="p-1 rounded opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      }
                    />
                    <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                      删除镜头
                    </TooltipContent>
                  </Tooltip>
                </div>
              </div>
            );
          })}
          </div>
        </div>
      </div>

      {/* 添加按钮 */}
      <button
        onClick={onAddItem}
        className={cn(
          "w-full flex items-center justify-center gap-2 py-3 rounded-xl",
          "border-2 border-dashed border-border/30 hover:border-primary/30",
          "text-muted-foreground hover:text-primary transition-all",
          "text-xs font-medium"
        )}
      >
        <Plus className="h-3.5 w-3.5" />
        添加镜头
      </button>

      <VideoPreviewDialog
        open={!!previewVideoUrl}
        title="镜头视频预览"
        videoUrl={previewVideoUrl}
        onClose={() => setPreviewVideoUrl(null)}
      />

      {/* 图片大图预览灯箱 */}
      {previewImageUrl && (
        <div 
          className="modal-overlay fixed inset-0 z-[9999] flex items-center justify-center p-4 animate-in fade-in duration-200"
          onClick={() => setPreviewImageUrl(null)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center gap-3" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setPreviewImageUrl(null)}
              className="absolute -top-12 right-0 rounded-full border border-border/40 bg-background/80 p-1.5 text-foreground shadow-xl backdrop-blur-xl transition-colors hover:bg-background"
              type="button"
            >
              <X className="h-5 w-5" />
            </button>
            <SafeImage
              src={resolveMediaUrl(previewImageUrl)}
              alt={previewImageTitle}
              fallbackType="image"
              className="max-w-full max-h-[80vh] rounded-lg object-contain shadow-2xl border border-border/40 select-none pointer-events-none"
            />
            <p className="rounded-full border border-border/40 bg-background/80 px-3 py-1.5 text-xs font-medium text-foreground shadow-xl backdrop-blur-xl">
              {previewImageTitle}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
