"use client";

import { useEffect, useState, useRef } from "react";
import {
  Clapperboard,
  Camera,
  Eye,
  MessageSquare,
  Volume2,
  TreePalm,
  Users,
  MapPin,
  Clock,
  Loader2,
  Plus,
  Save,
  Hash,
  Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  scriptApi,
  type SceneItem,
  type DialogueElement,
} from "@/lib/api/script";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { parseDialogues, parseCharacters } from "./utils";
import { InlineEdit } from "./inline-edit";
import { ContentEditable } from "./content-editable";
import { EditableDialogueItem } from "./dialogue-item";
import { InsertButton } from "./insert-button";
import { toast } from "sonner";

export function SceneCard({
  scene,
  isSelected,
  onSelect,
  onSceneUpdated,
  onDelete,
}: {
  scene: SceneItem;
  isSelected: boolean;
  onSelect: () => void;
  onSceneUpdated: (updated: SceneItem) => void;
  onDelete?: () => void;
}) {
  const [localScene, setLocalScene] = useState(scene);
  const [localDialogues, setLocalDialogues] = useState<DialogueElement[]>(() =>
    parseDialogues(scene)
  );
  const [dirty, setDirty] = useState(false);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [openPickerIdx, setOpenPickerIdx] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  // 使用 ref 保存最新值，避免 handleSave 闭包取到旧值
  const localSceneRef = useRef(localScene);
  const localDialoguesRef = useRef(localDialogues);

  // 外部数据变化时同步（如首次加载）
  useEffect(() => {
    setLocalScene(scene);
    localSceneRef.current = scene;
    const dlgs = parseDialogues(scene);
    setLocalDialogues(dlgs);
    localDialoguesRef.current = dlgs;
    setDirty(false);
  }, [scene]);

  const sceneChars = parseCharacters(localScene);

  const updateField = (field: string, value: string) => {
    setLocalScene((prev) => {
      const next = { ...prev, [field]: value };
      localSceneRef.current = next;
      return next;
    });
    setDirty(true);
  };

  const updateDialogue = (index: number, updated: DialogueElement) => {
    setLocalDialogues((prev) => {
      const next = [...prev];
      next[index] = updated;
      localDialoguesRef.current = next;
      return next;
    });
    setDirty(true);
  };

  const deleteDialogue = (index: number) => {
    setLocalDialogues((prev) => {
      const next = prev.filter((_, i) => i !== index);
      localDialoguesRef.current = next;
      return next;
    });
    setDirty(true);
  };

  const addDialogue = (type: number) => {
    const newItem: DialogueElement = {
      type,
      content: "",
      character_name: type === 1 || type === 3 ? "" : undefined,
      parenthetical: type === 1 || type === 3 ? "" : undefined,
      sortOrder: localDialogues.length,
    };
    setLocalDialogues((prev) => {
      const next = [...prev, newItem];
      localDialoguesRef.current = next;
      return next;
    });
    setDirty(true);
  };

  const insertDialogue = (atIndex: number, type: number) => {
    const newItem: DialogueElement = {
      type,
      content: "",
      character_name: type === 1 || type === 3 ? "" : undefined,
      parenthetical: type === 1 ? "" : undefined,
      sortOrder: atIndex,
    };
    setLocalDialogues((prev) => {
      const next = [...prev.slice(0, atIndex), newItem, ...prev.slice(atIndex)];
      localDialoguesRef.current = next;
      return next;
    });
    setDirty(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      // 从 ref 读取最新值（避免闭包陈旧问题）
      const currentScene = localSceneRef.current;
      const currentDialogues = localDialoguesRef.current;
      const currentChars = parseCharacters(currentScene);

      // 保存时根据类型清理不需要的字段
      const dialoguesWithOrder = currentDialogues.map((d, i) => {
        const needsChar = d.type === 1 || d.type === 3;
        return {
          type: d.type,
          content: d.content,
          character_name: needsChar ? d.character_name : undefined,
          parenthetical: d.type === 1 || d.type === 3 ? d.parenthetical : undefined,
          sortOrder: i,
        };
      });

      const updated = await scriptApi.updateScene({
        id: currentScene.id,
        sceneNumber: currentScene.sceneNumber,
        sceneHeading: currentScene.sceneHeading,
        location: currentScene.location || undefined,
        timeOfDay: currentScene.timeOfDay || undefined,
        intExt: currentScene.intExt || undefined,
        sceneDescription: currentScene.sceneDescription || undefined,
        dialogues: JSON.stringify(dialoguesWithOrder),
        characters: JSON.stringify(currentChars),
        version: currentScene.version,
      });

      setLocalScene(updated);
      localSceneRef.current = updated;
      setLocalDialogues(parseDialogues(updated));
      localDialoguesRef.current = parseDialogues(updated);
      setDirty(false);
      onSceneUpdated(updated);
      toast.success("场次保存成功");
      } catch (err: unknown) {
        toast.error(err instanceof Error ? err.message : "保存场次失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      ref={cardRef}
      data-scene-id={scene.id}
      onClick={() => {
        onSelect();
        setOpenPickerIdx(null);
      }}
      className={cn(
        "relative rounded-xl border transition-all duration-200 group/card",
        isSelected
          ? "border-primary/60 bg-card/60 shadow-lg shadow-primary/5 ring-1 ring-primary/10"
          : "border-border/70 bg-card/30 hover:border-primary/30 hover:bg-card/40"
      )}
    >
      {/* 场次头部 */}
      <div className="px-5 pt-4 pb-3">
        <div className="flex items-center gap-3 mb-2">
          {/* 场号 */}
          <div className="flex items-center gap-1.5 shrink-0">
            <Hash className="h-3.5 w-3.5 text-muted-foreground" />
            <InlineEdit
              value={localScene.sceneNumber || ""}
              placeholder="场号"
              onChange={(val) => updateField("sceneNumber", val)}
              onDirty={() => setDirty(true)}
              className="text-sm font-bold"
            />
          </div>

          {/* 内外景 */}
          <Select
            value={localScene.intExt || ""}
            onValueChange={(val) => updateField("intExt", val ?? "")}
            items={[{ value: "内", label: "内" }, { value: "外", label: "外" }, { value: "内/外", label: "内/外" }]}
          >
            <SelectTrigger
              size="sm"
              className={cn(
                "h-auto! px-2 py-0.5 text-[11px] font-medium shrink-0 rounded-md min-w-0 w-auto gap-1 border",
                localScene.intExt === "内"
                  ? "bg-violet-500/10 text-violet-400 border-violet-500/20"
                  : localScene.intExt === "外"
                  ? "bg-orange-500/10 text-orange-400 border-orange-500/20"
                  : "bg-muted/50 text-muted-foreground border-border/20"
              )}
            >
              <SelectValue placeholder="内/外" />
            </SelectTrigger>
            <SelectContent className="min-w-16 rounded-lg">
              <SelectGroup>
                <SelectItem value="内" className="text-xs py-1 pl-2 pr-6 rounded-md">内</SelectItem>
                <SelectItem value="外" className="text-xs py-1 pl-2 pr-6 rounded-md">外</SelectItem>
                <SelectItem value="内/外" className="text-xs py-1 pl-2 pr-6 rounded-md">内/外</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>

          {/* 地点 */}
          <span className="flex items-center gap-1 text-xs text-muted-foreground">
            <MapPin className="h-3 w-3 shrink-0" />
            <InlineEdit
              value={localScene.location || ""}
              placeholder="地点"
              onChange={(val) => updateField("location", val)}
              onDirty={() => setDirty(true)}
              className="text-xs"
            />
          </span>

          {/* 时间 */}
          <span className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="h-3 w-3 shrink-0" />
            <InlineEdit
              value={localScene.timeOfDay || ""}
              placeholder="时间"
              onChange={(val) => updateField("timeOfDay", val)}
              onDirty={() => setDirty(true)}
              className="text-xs"
            />
          </span>

          {/* 保存按钮 - 始终占位避免抖动 */}
          <button
              onClick={(e) => {
                e.stopPropagation();
                handleSave();
              }}
              disabled={saving || !dirty}
              className={cn(
                "ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium",
                "bg-primary text-primary-foreground",
                "hover:opacity-90 active:scale-95 transition-all",
                saving && "opacity-50 cursor-not-allowed",
                !dirty && "opacity-0 pointer-events-none"
              )}
            >
              {saving ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Save className="h-3 w-3" />
              )}
              保存
            </button>

          {/* 删除按钮 */}
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
              className="p-1.5 rounded-lg opacity-0 group-hover/card:opacity-100 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
              title="删除场次"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {/* 场景标题 */}
        <div className="flex items-center gap-2 mb-2">
          <Clapperboard className="h-4 w-4 text-amber-400 shrink-0" />
          <InlineEdit
            value={localScene.sceneHeading || ""}
            placeholder="场景标题"
            onChange={(val) => updateField("sceneHeading", val)}
            onDirty={() => setDirty(true)}
            className="text-sm font-semibold"
          />
        </div>

        {/* 出场角色 */}
        {sceneChars.length > 0 && (
          <div className="flex items-center gap-1.5 mb-2 flex-wrap">
            <Users className="h-3 w-3 text-muted-foreground shrink-0" />
            {sceneChars.map((char, i) => (
              <span
                key={i}
                className="text-[11px] px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 font-medium"
              >
                {char}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* 场景描述 */}
      <div className="px-5 pb-3">
        <div className="border-l-2 border-primary/20 pl-3 py-0.5">
          <InlineEdit
            value={localScene.sceneDescription || ""}
            placeholder="添加场景描述..."
            onChange={(val) => updateField("sceneDescription", val)}
            onDirty={() => setDirty(true)}
            className="text-xs text-muted-foreground italic leading-relaxed"
            multiline
          />
        </div>
      </div>

      {/* 对白/动作列表 */}
      {localDialogues.length > 0 && (
        <div className="px-5 pb-3 space-y-1">
          {localDialogues.map((d, i) => (
            <div
              key={i}
              className="relative pt-1"
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {/* 行间插入按钮（hover 当前行或上一行时显示） */}
              <InsertButton
                visible={hoveredIdx === i || hoveredIdx === i - 1}
                isOpen={openPickerIdx === i}
                onToggle={() => setOpenPickerIdx(openPickerIdx === i ? null : i)}
                onInsert={(type) => insertDialogue(i, type)}
              />
              <EditableDialogueItem
                item={d}
                index={i}
                onUpdate={updateDialogue}
                onDelete={deleteDialogue}
                onDirty={() => setDirty(true)}
              />
            </div>
          ))}
          {/* 末尾插入按钮（hover 最后一行或此区域时显示） */}
          <div
            className="relative h-3"
            onMouseEnter={() => setHoveredIdx(localDialogues.length)}
            onMouseLeave={() => setHoveredIdx(null)}
          >
            <InsertButton
              visible={hoveredIdx === localDialogues.length || hoveredIdx === localDialogues.length - 1}
              isOpen={openPickerIdx === localDialogues.length}
              onToggle={() => setOpenPickerIdx(openPickerIdx === localDialogues.length ? null : localDialogues.length)}
              onInsert={(type) => insertDialogue(localDialogues.length, type)}
            />
          </div>
        </div>
      )}

      {/* 添加元素按钮 */}
      <div className="px-5 pb-4 flex items-center gap-2 flex-wrap">
        {[
          { type: 1, label: "添加对白", icon: MessageSquare },
          { type: 2, label: "添加动作", icon: Eye },
          { type: 3, label: "添加旁白", icon: Volume2 },
          { type: 4, label: "添加镜头", icon: Camera },
          { type: 5, label: "添加环境", icon: TreePalm },
        ].map(({ type, label, icon: BtnIcon }) => (
          <button
            key={type}
            onClick={(e) => {
              e.stopPropagation();
              addDialogue(type);
            }}
            className={cn(
              "flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11px]",
              "text-muted-foreground hover:text-foreground",
              "hover:bg-muted/30 transition-colors"
            )}
          >
            <Plus className="h-3 w-3" />
            <BtnIcon className="h-3 w-3" />
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
