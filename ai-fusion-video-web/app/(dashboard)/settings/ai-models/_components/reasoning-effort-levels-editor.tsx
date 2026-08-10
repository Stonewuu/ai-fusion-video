"use client";

import { useState } from "react";
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { ArrowDown, ArrowUp, GripVertical, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface ReasoningEffortLevelsEditorProps {
  levels: string[];
  onChange: (levels: string[]) => void;
  disabled?: boolean;
}

interface SortableLevelProps {
  level: string;
  index: number;
  total: number;
  disabled: boolean;
  onMove: (from: number, to: number) => void;
  onRemove: (level: string) => void;
}

function SortableLevel({ level, index, total, disabled, onMove, onRemove }: SortableLevelProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: level,
    disabled,
  });

  return (
    <div
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className={`flex items-center gap-2 rounded-lg border border-border/20 bg-background/70 p-2 ${isDragging ? "z-10 shadow-md" : ""}`}
    >
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        disabled={disabled}
        aria-label={`拖动 ${level} 调整顺序`}
        title="拖动排序"
        {...attributes}
        {...listeners}
      >
        <GripVertical />
      </Button>
      <code className="min-w-0 flex-1 truncate text-xs">{level}</code>
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        disabled={disabled || index === 0}
        aria-label={`上移 ${level}`}
        title="上移"
        onClick={() => onMove(index, index - 1)}
      >
        <ArrowUp />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        disabled={disabled || index === total - 1}
        aria-label={`下移 ${level}`}
        title="下移"
        onClick={() => onMove(index, index + 1)}
      >
        <ArrowDown />
      </Button>
      <Button
        type="button"
        variant="destructive-ghost"
        size="icon-xs"
        disabled={disabled}
        aria-label={`删除 ${level}`}
        title="删除"
        onClick={() => onRemove(level)}
      >
        <X />
      </Button>
    </div>
  );
}

export function ReasoningEffortLevelsEditor({
  levels,
  onChange,
  disabled = false,
}: ReasoningEffortLevelsEditorProps) {
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const addLevel = () => {
    const level = draft.trim();
    if (!level) {
      setError("请输入等级名称");
      return;
    }
    if (levels.includes(level)) {
      setError("等级名称不能重复");
      return;
    }
    onChange([...levels, level]);
    setDraft("");
    setError(null);
  };

  const moveLevel = (from: number, to: number) => {
    if (to < 0 || to >= levels.length || from === to) return;
    onChange(arrayMove(levels, from, to));
  };

  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) return;
    moveLevel(levels.indexOf(String(active.id)), levels.indexOf(String(over.id)));
  };

  return (
    <div className="space-y-2 rounded-lg border border-border/20 bg-background/70 p-3">
      <div>
        <p className="text-xs font-medium">思考等级</p>
        <p className="text-[10px] text-muted-foreground">
          按能力从高到低排列；助手默认选择第一项。可填写供应商自定义名称。
        </p>
      </div>

      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <SortableContext items={levels} strategy={verticalListSortingStrategy}>
          <div className="space-y-1.5">
            {levels.map((level, index) => (
              <SortableLevel
                key={level}
                level={level}
                index={index}
                total={levels.length}
                disabled={disabled}
                onMove={moveLevel}
                onRemove={(removed) => onChange(levels.filter((item) => item !== removed))}
              />
            ))}
          </div>
        </SortableContext>
      </DndContext>

      {!levels.length ? (
        <p className="rounded-lg border border-dashed border-border/30 p-3 text-center text-[10px] text-muted-foreground">
          当前没有可选等级，助手将不显示等级选择器。
        </p>
      ) : null}

      <div className="flex items-center gap-2">
        <Input
          value={draft}
          disabled={disabled}
          placeholder="例如 max、high、medium"
          aria-label="新增思考等级"
          onChange={(event) => {
            setDraft(event.target.value);
            setError(null);
          }}
          onKeyDown={(event) => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            addLevel();
          }}
        />
        <Button type="button" variant="outline" size="sm" disabled={disabled} onClick={addLevel}>
          <Plus /> 添加
        </Button>
      </div>
      {error ? <p className="text-[10px] text-destructive">{error}</p> : null}
    </div>
  );
}
