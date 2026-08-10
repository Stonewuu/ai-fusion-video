"use client";

import { useState } from "react";
import { Trash2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import type { DialogueElement } from "@/lib/api/script";
import { dialogueTypeConfig } from "./constants";
import { ContentEditable } from "./content-editable";

export function EditableDialogueItem({
  item,
  index,
  onUpdate,
  onDelete,
  onDirty,
}: {
  item: DialogueElement;
  index: number;
  onUpdate: (index: number, updated: DialogueElement) => void;
  onDelete: (index: number) => void;
  onDirty?: () => void;
}) {
  const [showTypePicker, setShowTypePicker] = useState(false);
  const config = dialogueTypeConfig[item.type] || dialogueTypeConfig[2];
  const Icon = config.icon;
  const needsCharacter = item.type === 1 || item.type === 3;

  const handleTypeChange = (newType: number) => {
    // 切换类型时保留所有字段，方便切换回来不丢数据
    onUpdate(index, { ...item, type: newType });
    setShowTypePicker(false);
  };

  return (
    <div
      className={cn(
        "group/dl flex items-center flex-wrap gap-x-2 gap-y-0.5 px-3.5 py-2.5 rounded-lg border transition-all",
        config.cls,
        "hover:shadow-sm"
      )}
    >
      {/* 图标 — 点击切换类型 */}
      <div className="relative shrink-0 self-center">
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowTypePicker(!showTypePicker);
          }}
          className="p-0.5 rounded hover:bg-black/5 dark:hover:bg-white/10 transition-colors cursor-pointer"
          title="切换类型"
        >
          <Icon className="h-3.5 w-3.5" />
        </button>
        <AnimatePresence>
          {showTypePicker && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -4 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -4 }}
              transition={{ duration: 0.12 }}
              className="absolute left-0 top-full mt-1 z-20 bg-popover border border-border/30 rounded-lg shadow-lg p-1 min-w-28"
            >
              {Object.entries(dialogueTypeConfig).map(([typeKey, cfg]) => {
                const TypeIcon = cfg.icon;
                const typeNum = Number(typeKey);
                return (
                  <button
                    key={typeKey}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleTypeChange(typeNum);
                    }}
                    className={cn(
                      "w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-xs transition-colors",
                      typeNum === item.type
                        ? "bg-primary/10 text-primary font-medium"
                        : "hover:bg-muted/50 text-foreground"
                    )}
                  >
                    <TypeIcon className="h-3.5 w-3.5" />
                    {cfg.label}
                  </button>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 角色名 */}
      {needsCharacter && (
        <ContentEditable
          value={item.character_name || ""}
          placeholder="角色名"
          onChange={(val) =>
            onUpdate(index, { ...item, character_name: val })
          }
          onDirty={onDirty}
          maxWidth="8rem"
          className="shrink-0 text-xs font-bold px-1 py-0.5 min-w-[3em]"
        />
      )}

      {/* 情绪/动作说明 */}
      {(item.type === 1 || item.type === 3) && (
        <ContentEditable
          value={item.parenthetical || ""}
          placeholder="（语气）"
          onChange={(val) =>
            onUpdate(index, { ...item, parenthetical: val })
          }
          onDirty={onDirty}
          maxWidth="8rem"
          className="shrink-0 text-[11px] text-muted-foreground px-1 py-0.5 min-w-[3em]"
        />
      )}

      {/* 内容 */}
      <ContentEditable
        value={item.content || ""}
        placeholder="输入内容..."
        onChange={(val) =>
          onUpdate(index, { ...item, content: val })
        }
        onDirty={onDirty}
        className="flex-1 min-w-0 text-xs leading-relaxed"
      />

      {/* 删除按钮 */}
      <button
        onClick={() => onDelete(index)}
        className="shrink-0 p-1 rounded opacity-0 group-hover/dl:opacity-100 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all self-center"
        title="删除"
      >
        <Trash2 className="h-3 w-3" />
      </button>
    </div>

  );
}
