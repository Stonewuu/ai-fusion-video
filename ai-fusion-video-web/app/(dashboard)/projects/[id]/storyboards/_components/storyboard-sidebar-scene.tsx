"use client";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Camera, GripVertical, Trash2 } from "lucide-react";
import type { StoryboardScene } from "@/lib/api/storyboard";
import { cn } from "@/lib/utils";
export function SortableSceneItem({
  scene,
  isActive,
  onSelect,
  onDelete,
}: {
  scene: StoryboardScene;
  isActive: boolean;
  onSelect: () => void;
  onDelete: (e: React.MouseEvent) => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: scene.id });
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };
  return (
    <div
      ref={setNodeRef}
      data-sidebar-row
      style={style}
      className={cn(
        "group/scene mb-0.5 flex min-w-0 items-center rounded-lg p-1",
        isDragging && "opacity-50 z-10",
        isActive ? "bg-primary/10" : "hover:bg-primary/5",
      )}
    >
      <span
        {...attributes}
        {...listeners}
        aria-label={`拖动场次 ${scene.sceneNumber}`}
        className="flex size-6 items-center justify-center rounded-lg outline-none focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-ring/50 opacity-0 group-hover/scene:opacity-40 hover:opacity-100! text-muted-foreground cursor-grab active:cursor-grabbing transition-opacity shrink-0"
        onPointerDown={(e) => {
          e.stopPropagation();
          listeners?.onPointerDown?.(e);
        }}
      >
        <GripVertical className="h-3 w-3" />
      </span>
      <button
        type="button"
        aria-current={isActive ? "true" : undefined}
        onClick={onSelect}
        className={cn(
          "flex h-7 min-w-0 flex-1 items-center gap-1.5 rounded-lg pr-2 text-left text-[11px] leading-4 outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
          isActive
            ? "text-primary font-medium"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <Camera
          className={cn(
            "h-3 w-3 shrink-0",
            isActive ? "text-primary" : "text-muted-foreground/60",
          )}
        />
        <span className="truncate">
          {scene.sceneHeading || `场次 ${scene.sceneNumber || "?"}`}
        </span>
      </button>
      <button
        onClick={onDelete}
        className="grid size-6 shrink-0 place-items-center rounded-lg text-muted-foreground opacity-0 outline-none transition-opacity group-hover/scene:opacity-100 hover:bg-destructive/10 hover:text-destructive focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-ring/50 motion-reduce:transition-none [@media(hover:none)]:opacity-100"
        title="删除场次"
      >
        <Trash2 className="h-3 w-3" />
      </button>
    </div>
  );
}
