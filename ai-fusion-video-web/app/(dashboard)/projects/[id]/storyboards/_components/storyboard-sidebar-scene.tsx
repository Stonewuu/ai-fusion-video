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
        "group/scene flex items-center mb-0.5 rounded-md overflow-hidden",
        isDragging && "opacity-50 z-10",
        isActive ? "bg-primary/10" : "hover:bg-primary/5",
      )}
    >
      <span
        {...attributes}
        {...listeners}
        className="px-1.5 self-stretch flex items-center opacity-0 group-hover/scene:opacity-40 hover:opacity-100! text-muted-foreground cursor-grab active:cursor-grabbing transition-opacity shrink-0"
        onPointerDown={(e) => {
          e.stopPropagation();
          listeners?.onPointerDown?.(e);
        }}
      >
        <GripVertical className="h-3 w-3" />
      </span>
      <button
        onClick={onSelect}
        className={cn(
          "flex-1 flex items-center min-w-0 gap-1.5 py-1.5 pr-2 text-left text-[11px] outline-none",
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
        className="p-1.5 mr-1 flex items-center shrink-0 rounded-full opacity-0 group-hover/scene:opacity-100 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
        title="删除场次"
      >
        <Trash2 className="h-3 w-3" />
      </button>
    </div>
  );
}
