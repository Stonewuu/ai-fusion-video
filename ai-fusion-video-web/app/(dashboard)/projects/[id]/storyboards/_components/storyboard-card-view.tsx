"use client";
import { useState, useMemo, useCallback, useLayoutEffect } from "react";
import { createPortal } from "react-dom";
import { Film, Plus } from "lucide-react";
import { VideoPreviewDialog } from "@/components/dashboard/video-preview-dialog";
import { cn } from "@/lib/utils";
import type { StoryboardFrameType, StoryboardItem } from "@/lib/api/storyboard";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragStartEvent,
  type DragEndEvent,
  DragOverlay,
  defaultDropAnimationSideEffects,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
} from "@dnd-kit/sortable";
import { CardItemUI, SortableCardItem } from "./storyboard-card-item";
import { useStoryboardVirtualRows } from "./use-storyboard-virtual-rows";
export function StoryboardCardView({
  items,
  selectedItemId,
  onSelectItem,
  onAddItem,
  onDeleteItem,
  onReorderItems,
  onVideoGen,
  onOpenFrameDialog,
}: {
  items: StoryboardItem[];
  selectedItemId: number | null;
  onSelectItem: (id: number) => void;
  onAddItem: () => void;
  onDeleteItem: (id: number) => void;
  onReorderItems?: (reordered: StoryboardItem[]) => void;
  onVideoGen?: (itemId: number) => void;
  onOpenFrameDialog?: (
    item: StoryboardItem,
    frameType: StoryboardFrameType,
  ) => void;
}) {
  const [activeId, setActiveId] = useState<number | null>(null);
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string | null>(null);
  const [focusedId, setFocusedId] = useState<number | null>(null);
  const indexById = useMemo(
    () => new Map(items.map((item, index) => [item.id, index])),
    [items],
  );
  const getItemKey = useCallback(
    (index: number) => (index === items.length ? "add-shot" : items[index].id),
    [items],
  );
  const {
    virtualizer,
    setContentRef,
    width,
    lanes: columns,
  } = useStoryboardVirtualRows({
    count: items.length === 0 ? 0 : items.length + 1,
    getItemKey,
    estimateSize: () => 360,
    lanes: (width) => Math.max(1, Math.floor((width + 16) / 316)),
    gap: 16,
    pinnedIndexes: [
      activeId === null ? -1 : (indexById.get(activeId) ?? -1),
      focusedId === null ? -1 : (indexById.get(focusedId) ?? -1),
    ],
  });
  useLayoutEffect(() => {
    virtualizer.measure();
  }, [width, columns, virtualizer]);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );
  const handleDragStart = (event: DragStartEvent) => {
    setActiveId(event.active.id as number);
  };
  const handleDragEnd = (event: DragEndEvent) => {
    setActiveId(null);
    const { active, over } = event;
    if (!over || active.id === over.id || !onReorderItems) {
      return;
    }
    const oldIndex = items.findIndex((i) => i.id === active.id);
    const newIndex = items.findIndex((i) => i.id === over.id);
    if (oldIndex !== -1 && newIndex !== -1) {
      const reordered = arrayMove(items, oldIndex, newIndex);
      onReorderItems(reordered);
    }
  };
  const handleDragCancel = () => {
    setActiveId(null);
  };
  // 缓存，避免拖拽时重渲染整个子树
  const itemIds = useMemo(() => items.map((i) => i.id), [items]);
  const activeItem = useMemo(
    () => (activeId ? items.find((i) => i.id === activeId) : null),
    [activeId, items],
  );
  const activeIndex = useMemo(
    () => (activeId ? items.findIndex((i) => i.id === activeId) : -1),
    [activeId, items],
  );
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
        <div className="h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center mb-4 text-primary">
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
    <div className="space-y-4 relative">
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        onDragCancel={handleDragCancel}
      >
        <SortableContext items={itemIds} strategy={rectSortingStrategy}>
          {/* 修改 Grid 列数，解决卡片太小的问题 */}
          <div ref={setContentRef} className="relative">
            {virtualizer.getVirtualItems().map((row) => {
              const idx = row.index;
              const item = items[idx];
              return (
                <div
                  key={row.key}
                  data-index={idx}
                  data-shot-id={item?.id}
                  ref={virtualizer.measureElement}
                  className="absolute top-0 p-1"
                  style={{
                    left:
                      "calc(" +
                      (row.lane * 100) / columns +
                      "% + " +
                      (row.lane * 16) / columns +
                      "px)",
                    width:
                      "calc((100% - " +
                      (columns - 1) * 16 +
                      "px) / " +
                      columns +
                      ")",
                  }}
                  onFocusCapture={() => setFocusedId(item?.id ?? null)}
                  onBlurCapture={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget)) {
                      setFocusedId(null);
                    }
                  }}
                >
                  {item ? (
                    <SortableCardItem
                      item={item}
                      idx={idx}
                      isSelected={selectedItemId === item.id}
                      onSelect={() => onSelectItem(item.id)}
                      onDelete={onDeleteItem}
                      onVideoGen={onVideoGen}
                      onOpenFrameDialog={onOpenFrameDialog}
                      onPreviewVideo={setPreviewVideoUrl}
                    />
                  ) : (
                    <div
                      onClick={onAddItem}
                      className={cn(
                        "rounded-xl border-2 border-dashed border-border/40",
                        "hover:border-primary/40 hover:bg-primary/5 transition-all duration-200 cursor-pointer",
                        "flex flex-col text-muted-foreground/50 hover:text-primary group bg-card overflow-hidden",
                      )}
                    >
                      <div className="aspect-video shrink-0 border-b border-transparent bg-transparent flex items-center justify-center">
                        <div className="h-12 w-12 rounded-full bg-muted/60 group-hover:bg-primary/10 flex items-center justify-center transition-colors">
                          <Plus className="h-6 w-6 text-muted-foreground/70 group-hover:text-primary" />
                        </div>
                      </div>
                      <div className="h-[140px] p-3 flex flex-col items-center justify-center">
                        <span className="text-sm font-medium">添加镜头</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </SortableContext>

        {/* 悬浮拖拽时的替身 */}
        {typeof document !== "undefined" &&
          createPortal(
            <DragOverlay
              dropAnimation={{
                sideEffects: defaultDropAnimationSideEffects({
                  styles: { active: { opacity: "0.5" } },
                }),
              }}
            >
              {activeItem ? (
                <CardItemUI
                  item={activeItem}
                  idx={activeIndex}
                  isSelected={selectedItemId === activeItem.id}
                  isOverlay
                />
              ) : null}
            </DragOverlay>,
            document.body,
          )}
      </DndContext>

      <VideoPreviewDialog
        open={!!previewVideoUrl}
        title="镜头视频预览"
        videoUrl={previewVideoUrl}
        onClose={() => setPreviewVideoUrl(null)}
      />
    </div>
  );
}
