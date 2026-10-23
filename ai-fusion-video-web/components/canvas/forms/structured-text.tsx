"use client";

import { ArrowDown, ArrowUp, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { CanvasSelect } from "./canvas-select";
import { CanvasChoiceField } from "./field-controls";
import type { DialogueElement } from "@/lib/api/script";

const dialogueTypes = [
  { value: "1", label: "对白" },
  { value: "2", label: "动作" },
  { value: "3", label: "旁白" },
  { value: "4", label: "镜头指令" },
  { value: "5", label: "环境描写" },
];
export function CanvasDialogueEditor({
  value,
  onChange,
  disabled = false,
}: {
  value: DialogueElement[];
  onChange: (next: DialogueElement[]) => void;
  disabled?: boolean;
}) {
  const update = (index: number, patch: Partial<DialogueElement>) =>
    onChange(
      value.map((item, current) =>
        current === index ? { ...item, ...patch } : item,
      ),
    );
  const move = (index: number, direction: number) => {
    const next = [...value];
    [next[index], next[index + direction]] = [
      next[index + direction],
      next[index],
    ];
    onChange(next.map((item, order) => ({ ...item, sortOrder: order })));
  };
  return (
    <div className="space-y-4">
      {value.map((item, index) => (
        <div key={index} className="space-y-2 border-b border-border/20 pb-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">{index + 1}</span>
            <div className="min-w-0 flex-1">
              <CanvasSelect
                label={`第${index + 1}段类型`}
                value={String(item.type)}
                items={dialogueTypes}
                disabled={disabled}
                placeholder="段落类型"
                onChange={(next) => update(index, { type: Number(next) })}
              />
            </div>
            <Button
              variant="ghost"
              size="icon"
              title="上移段落"
              disabled={disabled || index === 0}
              onClick={() => move(index, -1)}
            >
              <ArrowUp />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="下移段落"
              disabled={disabled || index === value.length - 1}
              onClick={() => move(index, 1)}
            >
              <ArrowDown />
            </Button>
            <Button
              variant="destructive-ghost"
              size="icon"
              title="删除段落"
              disabled={disabled}
              onClick={() =>
                onChange(value.filter((_, current) => current !== index))
              }
            >
              <Trash2 />
            </Button>
          </div>
          {item.type === 1 && (
            <>
              <Input
                value={item.character_name ?? ""}
                aria-label={`第${index + 1}段角色`}
                placeholder="角色"
                disabled={disabled}
                onChange={(event) =>
                  update(index, { character_name: event.target.value })
                }
              />
              <Input
                value={item.parenthetical ?? ""}
                aria-label={`第${index + 1}段表演提示`}
                placeholder="表演提示"
                disabled={disabled}
                onChange={(event) =>
                  update(index, { parenthetical: event.target.value })
                }
              />
            </>
          )}
          <Textarea
            value={item.content}
            rows={3}
            aria-label={`第${index + 1}段内容`}
            disabled={disabled}
            onChange={(event) => update(index, { content: event.target.value })}
          />
        </div>
      ))}
      <Button
        variant="outline"
        size="sm"
        disabled={disabled}
        onClick={() =>
          onChange([
            ...value,
            { type: 1, content: "", sortOrder: value.length },
          ])
        }
      >
        <Plus />
        添加段落
      </Button>
    </div>
  );
}

export interface CanvasCharacter {
  name: string;
  assetId?: number;
  description?: string;
  importance?: string;
}
export function CanvasCharactersEditor({
  value,
  onChange,
  disabled = false,
}: {
  value: CanvasCharacter[];
  onChange: (next: CanvasCharacter[]) => void;
  disabled?: boolean;
}) {
  const update = (index: number, patch: Partial<CanvasCharacter>) =>
    onChange(
      value.map((item, current) =>
        current === index ? { ...item, ...patch } : item,
      ),
    );
  return (
    <div className="space-y-4">
      {value.map((item, index) => (
        <div key={index} className="space-y-2 border-b border-border/20 pb-4">
          <div className="flex gap-2">
            <Input
              aria-label={`人物${index + 1}姓名`}
              value={item.name}
              placeholder="人物姓名"
              disabled={disabled}
              onChange={(event) => update(index, { name: event.target.value })}
            />
            <Button
              variant="destructive-ghost"
              size="icon"
              disabled={disabled}
              title="删除人物"
              onClick={() =>
                onChange(value.filter((_, current) => current !== index))
              }
            >
              <Trash2 />
            </Button>
          </div>
          <CanvasChoiceField label={`人物${index + 1}重要程度`} value={item.importance ?? null} placeholder="选择人物定位"
            items={[{ value: "主角", label: "主角" }, { value: "配角", label: "配角" }, { value: "次要角色", label: "次要角色" }]}
            custom disabled={disabled} onChange={(importance) => update(index, { importance })} />
          <Textarea
            aria-label={`人物${index + 1}描述`}
            value={item.description ?? ""}
            placeholder="人物描述"
            rows={3}
            disabled={disabled}
            onChange={(event) =>
              update(index, { description: event.target.value })
            }
          />
        </div>
      ))}
      <Button
        variant="outline"
        size="sm"
        disabled={disabled}
        onClick={() => onChange([...value, { name: "", description: "" }])}
      >
        <Plus />
        添加人物
      </Button>
    </div>
  );
}
