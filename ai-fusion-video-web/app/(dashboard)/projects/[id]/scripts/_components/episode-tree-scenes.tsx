"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import { Clapperboard, GripVertical, Trash2 } from "lucide-react";
import {
  DndContext,
  DragOverlay,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Button } from "@/components/ui/button";
import type { SceneItem } from "@/lib/api/script";
import { cn } from "@/lib/utils";
import type { EpisodeTreeActions } from "./episode-tree-actions";

function SortableSceneItem({
  scene,
  episodeId,
  selected,
  actions,
}: {
  scene: SceneItem;
  episodeId: number;
  selected: boolean;
  actions: EpisodeTreeActions;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: scene.id });
  return (
    <div
      ref={setNodeRef}
      data-sidebar-row
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className={cn(
        "group/scene flex min-w-0 items-center rounded-lg p-1",
        isDragging && "opacity-30",
        selected ? "bg-primary/10" : "hover:bg-primary/5",
      )}
    >
      <button
        type="button"
        {...attributes}
        {...listeners}
        aria-label={`拖动场次 ${scene.sceneNumber}`}
        className="flex size-6 shrink-0 touch-none items-center justify-center rounded-lg text-muted-foreground opacity-0 outline-none transition-opacity group-hover/scene:opacity-60 focus-visible:opacity-100 focus-visible:ring-[3px] focus-visible:ring-ring/50 [@media(hover:none)]:opacity-60"
      >
        <GripVertical className="size-3" />
      </button>
      <button
        type="button"
        onClick={() => actions.onSelectScene(scene.id, episodeId)}
        aria-current={selected ? "true" : undefined}
        className={cn(
          "flex h-7 min-w-0 flex-1 items-center gap-1.5 rounded-lg pr-2 text-left text-[11px] outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
          selected
            ? "font-medium text-foreground"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <Clapperboard className="size-3 shrink-0" />
        <span className="truncate">
          {scene.sceneNumber} {scene.sceneHeading || "场次"}
        </span>
      </button>
      <Button
        variant="destructive-ghost"
        size="icon-xs"
        title="删除场次"
        aria-label="删除场次"
        className="opacity-0 group-hover/scene:opacity-100 focus-visible:opacity-100 [@media(hover:none)]:opacity-100"
        onClick={() => actions.onDeleteScene(scene.id, episodeId)}
      >
        <Trash2 />
      </Button>
    </div>
  );
}

export function EpisodeTreeScenes({
  episodeId,
  scenes,
  selectedSceneId,
  actions,
  onDraggingChange,
  reduceMotion,
}: {
  episodeId: number;
  scenes: SceneItem[];
  selectedSceneId: number | null;
  actions: EpisodeTreeActions;
  onDraggingChange: (id: number | null) => void;
  reduceMotion: boolean;
}) {
  const [draggedScene, setDraggedScene] = useState<SceneItem | null>(null);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );
  const finishDrag = () => {
    setDraggedScene(null);
    onDraggingChange(null);
  };
  const reorder = ({ active, over }: DragEndEvent) => {
    finishDrag();
    if (!over || active.id === over.id) {
      return;
    }
    const from = scenes.findIndex((scene) => scene.id === active.id);
    const to = scenes.findIndex((scene) => scene.id === over.id);
    if (from !== -1 && to !== -1) {
      actions.onReorderScenes?.(episodeId, from, to);
    }
  };

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={({ active }) => {
        setDraggedScene(scenes.find((scene) => scene.id === active.id)!);
        onDraggingChange(episodeId);
      }}
      onDragCancel={finishDrag}
      onDragEnd={reorder}
    >
      <SortableContext
        items={scenes.map((scene) => scene.id)}
        strategy={verticalListSortingStrategy}
      >
        {scenes.map((scene) => (
          <SortableSceneItem
            key={scene.id}
            scene={scene}
            episodeId={episodeId}
            selected={selectedSceneId === scene.id}
            actions={actions}
          />
        ))}
      </SortableContext>
      {typeof document !== "undefined" &&
        createPortal(
          <DragOverlay
            dropAnimation={
              reduceMotion ? null : { duration: 150, easing: "ease-out" }
            }
          >
            {draggedScene && (
              <div className="flex items-center gap-2 rounded-lg border border-border/40 bg-popover p-2 text-xs shadow-xl">
                <Clapperboard className="size-3 shrink-0" />
                <span className="truncate">
                  {draggedScene.sceneNumber}{" "}
                  {draggedScene.sceneHeading || "场次"}
                </span>
              </div>
            )}
          </DragOverlay>,
          document.body,
        )}
    </DndContext>
  );
}
