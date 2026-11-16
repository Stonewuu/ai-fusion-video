"use client";
import { Film, GripVertical, Plus, Trash2, Video, X } from "lucide-react";
import { VideoPreviewDialog } from "@/components/dashboard/video-preview-dialog";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import type { StoryboardFrameType, StoryboardItem } from "@/lib/api/storyboard";
import { useStoryboardVirtualRows } from "./use-storyboard-virtual-rows";
import {
  StoryboardTableCell,
  TABLE_TOOLTIP_CLASS,
} from "./storyboard-table-cell";
import {
  useState,
  useRef,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
} from "react";
import { Tooltip, TooltipTrigger } from "@/components/ui/tooltip";
import { FixedTooltipContent as TooltipContent } from "@/components/shared/fixed-tooltip-content";
import { SafeImage } from "@/components/media/managed-image";
import { attachStoryboardHorizontalScroll } from "./storyboard-horizontal-scroll";
import {
  STORYBOARD_TABLE_COLUMNS as COLUMNS,
  loadSavedColWidths,
  saveColWidths,
} from "./storyboard-table-columns";
const TOOLTIP_CONTENT_CLASS = TABLE_TOOLTIP_CLASS;
/** 固定列宽 */
const DRAG_COL_W = 28;
const ACTION_COL_W = 56;
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
  onUpdateItemField: (
    item: StoryboardItem,
    field: string,
    value: string | number | null,
  ) => Promise<void>;
  onAddItem: () => void;
  onDeleteItem: (id: number) => void;
  onReorderItems?: (reorderedItems: StoryboardItem[]) => void;
  onVideoGen?: (itemId: number) => void;
  onOpenFrameDialog?: (
    item: StoryboardItem,
    frameType: StoryboardFrameType,
  ) => void;
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
    [],
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
    [dragIdx, overIdx],
  );
  // ========== 列宽拖拽（直接 DOM 操作避免重渲染） ==========
  const [resizingCol, setResizingCol] = useState<number | null>(null);
  const tableRootRef = useRef<HTMLDivElement>(null);
  const headerScrollRef = useRef<HTMLDivElement>(null);
  const bodyScrollRef = useRef<HTMLDivElement>(null);
  const headerContentRef = useRef<HTMLDivElement>(null);
  const bodyContentRef = useRef<HTMLDivElement>(null);
  const colWidthsRef = useRef(colWidths);
  const hasItems = items.length > 0;
  const [focusedItemId, setFocusedItemId] = useState<number | null>(null);
  const [editingCells, setEditingCells] = useState<Map<string, number>>(
    () => new Map(),
  );
  const handleEditingChange = useCallback(
    (itemId: number, field: string, editing: boolean) => {
      const key = `${itemId}:${field}`;
      setEditingCells((previous) => {
        if (previous.has(key) === editing) {
          return previous;
        }
        const next = new Map(previous);
        if (editing) {
          next.set(key, itemId);
        } else {
          next.delete(key);
        }
        return next;
      });
    },
    [],
  );
  const itemIndexes = useMemo(
    () => new Map(items.map((item, index) => [item.id, index])),
    [items],
  );
  const itemKey = useCallback((index: number) => items[index].id, [items]);
  const { virtualizer, setContentRef } = useStoryboardVirtualRows({
    count: items.length,
    getItemKey: itemKey,
    estimateSize: () => 128,
    pinnedIndexes: [
      dragIdx ?? -1,
      focusedItemId === null ? -1 : (itemIndexes.get(focusedItemId) ?? -1),
      ...Array.from(
        editingCells.values(),
        (itemId) => itemIndexes.get(itemId) ?? -1,
      ),
    ],
  });
  const setBodyContentRef = useCallback(
    (node: HTMLDivElement | null) => {
      bodyContentRef.current = node;
      setContentRef(node);
    },
    [setContentRef],
  );
  useLayoutEffect(() => {
    const header = headerScrollRef.current;
    const content = headerContentRef.current;
    const body = bodyScrollRef.current;
    if (!header || !content || !body) {
      return;
    }
    return attachStoryboardHorizontalScroll(header, content, body);
  }, [hasItems]);
  useEffect(() => {
    colWidthsRef.current = colWidths;
  }, [colWidths]);
  /** 构建 grid-template-columns 字符串 */
  const buildGridTemplate = useCallback(
    (widths: number[]) =>
      `${DRAG_COL_W}px ${widths.map((w) => `${w}px`).join(" ")} ${ACTION_COL_W}px`,
    [],
  );
  /** 根据列宽数组直接更新所有 grid 行的 gridTemplateColumns */
  const applyGridTemplate = useCallback(
    (widths: number[]) => {
      if (!tableRootRef.current) {
        return;
      }
      const tpl = buildGridTemplate(widths);
      const totalW =
        DRAG_COL_W + widths.reduce((s, w) => s + w, 0) + ACTION_COL_W;
      // 同步表头和表体的内容宽度
      [headerContentRef.current, bodyContentRef.current].forEach((content) => {
        if (!content) {
          return;
        }
        content.style.width = `${totalW}px`;
        content.style.minWidth = `${totalW}px`;
      });
      // 更新所有 grid 行
      const gridRows =
        tableRootRef.current.querySelectorAll<HTMLElement>("[data-grid-row]");
      gridRows.forEach((row) => {
        row.style.gridTemplateColumns = tpl;
      });
    },
    [buildGridTemplate],
  );
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
    [applyGridTemplate],
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
          "bg-linear-to-br from-primary/5 to-transparent hover:border-primary/40 hover:from-primary/10 transition-all duration-300 cursor-pointer shadow-sm",
        )}
      >
        <div className="h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center mb-4 text-primary transition-transform group-hover:scale-110">
          <Film className="h-6 w-6" />
        </div>
        <h3 className="text-sm font-semibold text-foreground mb-1">暂无镜头</h3>
        <p className="text-xs text-muted-foreground/80">
          点击添加该场次的第一个镜头
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {/* 表格容器：表头独立吸顶，表体负责横向滚动 */}
      <div
        ref={tableRootRef}
        data-storyboard-table
        className="rounded-xl border border-border/20 bg-card/30"
      >
        <div className="sticky -top-5 z-20 rounded-t-xl bg-background shadow-sm">
          <div
            ref={headerScrollRef}
            data-storyboard-header
            className="overflow-hidden"
          >
            <div
              ref={headerContentRef}
              className="will-change-transform"
              style={{ width: totalWidth, minWidth: totalWidth }}
            >
              {/* ===== 表头 ===== */}
              <div
                className={cn(
                  "grid border-b border-border/30",
                  "bg-background/95",
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
                        "group/resize",
                      )}
                    >
                      <div
                        className={cn(
                          "absolute inset-y-1.5 left-1/2 -translate-x-1/2 w-[2px] rounded-full transition-colors",
                          resizingCol === i
                            ? "bg-primary/70"
                            : "bg-border/40 group-hover/resize:bg-primary/50",
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
          data-storyboard-body
          className="overflow-x-auto"
        >
          <div
            ref={setBodyContentRef}
            className="relative"
            style={{ width: totalWidth, minWidth: totalWidth }}
          >
            {/* ===== 数据行 ===== */}
            {virtualizer.getVirtualItems().map((row) => {
              const idx = row.index;
              const item = items[idx];
              const isDragging = dragIdx === idx;
              const isOver =
                overIdx === idx && dragIdx !== null && dragIdx !== idx;
              const showTopLine = isOver && dragIdx !== null && dragIdx > idx;
              const showBottomLine =
                isOver && dragIdx !== null && dragIdx < idx;
              return (
                <div
                  key={row.key}
                  ref={virtualizer.measureElement}
                  data-index={idx}
                  data-shot-id={item.id}
                  onFocusCapture={() => setFocusedItemId(item.id)}
                  onBlurCapture={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget)) {
                      setFocusedItemId(null);
                    }
                  }}
                  draggable
                  onDragStart={(e) => handleDragStart(e, idx)}
                  onDragEnd={handleDragEnd}
                  onDragOver={(e) => handleDragOver(e, idx)}
                  onClick={() => onSelectItem(item.id)}
                  className={cn(
                    "absolute left-0 top-0 grid w-full border-b border-border/10",
                    "transition-colors group cursor-pointer",
                    selectedItemId === item.id
                      ? "bg-primary/5 hover:bg-primary/8"
                      : "hover:bg-primary/3",
                    isDragging && "opacity-40",
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
                  {COLUMNS.map((col) => (
                    <StoryboardTableCell
                      key={col.field}
                      item={item}
                      idx={idx}
                      col={col}
                      assetLookup={assetLookup}
                      onSelectItem={onSelectItem}
                      onUpdateItemField={onUpdateItemField}
                      onOpenFrameDialog={onOpenFrameDialog}
                      onEditAssets={onEditAssets}
                      onEditingChange={handleEditingChange}
                      setPreviewImageUrl={setPreviewImageUrl}
                      setPreviewImageTitle={setPreviewImageTitle}
                      setPreviewVideoUrl={setPreviewVideoUrl}
                    />
                  ))}

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
          "text-xs font-medium",
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
          <div
            className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center gap-3"
            onClick={(e) => e.stopPropagation()}
          >
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
