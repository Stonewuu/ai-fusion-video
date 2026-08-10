"use client";

import { useEffect, useLayoutEffect, useState, useRef } from "react";
import { Pencil } from "lucide-react";
import { cn } from "@/lib/utils";

export function EditableCell({
  value,
  placeholder,
  onSave,
  onCellClick,
  className,
  multiline,
}: {
  value: string;
  placeholder?: string;
  onSave: (val: string) => void;
  onCellClick?: () => void;
  className?: string;
  multiline?: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const ref = useRef<HTMLInputElement | HTMLTextAreaElement>(null);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  // 使用 useLayoutEffect 在浏览器绘制前同步聚焦，消除“慢半拍”的延迟感，并强制将光标移至文本末尾
  useLayoutEffect(() => {
    if (editing && ref.current) {
      const el = ref.current;
      el.focus();
      const len = el.value.length;
      el.setSelectionRange(len, len);
    }
  }, [editing]);

  // 动态调整 textarea 的高度，使其与内容高度完全一致，从而在 flex 容器中垂直居中，且不撑高/缩水单元格
  useLayoutEffect(() => {
    if (editing && multiline && ref.current) {
      const textarea = ref.current as HTMLTextAreaElement;
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [editing, multiline, draft]);

  const handleSave = () => {
    setEditing(false);
    if (draft !== value) {
      onSave(draft);
    }
  };

  const handleCancel = () => {
    setEditing(false);
    setDraft(value);
  };

  if (editing) {
    const inputClassName = cn(
      "w-full bg-transparent border-0 outline-none p-0 m-0",
      "text-xs leading-relaxed focus:ring-0 focus:outline-none",
      "text-foreground placeholder:text-muted-foreground/30",
      className
    );

    return (
      <div
        className={cn(
          "relative w-full h-full min-h-[32px] flex items-center",
          "px-2 py-1.5 rounded-md",
          "border border-primary/40 bg-background shadow-sm",
          "focus-within:ring-2 focus-within:ring-primary/30",
          "transition-[box-shadow] duration-150 motion-reduce:transition-none",
          "text-xs leading-relaxed",
          className
        )}
      >
        {multiline ? (
          <textarea
            ref={ref as React.RefObject<HTMLTextAreaElement>}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={handleSave}
            onKeyDown={(e) => {
              if (e.key === "Escape") handleCancel();
            }}
            placeholder={placeholder}
            className={cn(inputClassName, "resize-none overflow-hidden")}
          />
        ) : (
          <input
            ref={ref as React.RefObject<HTMLInputElement>}
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={handleSave}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSave();
              if (e.key === "Escape") handleCancel();
            }}
            placeholder={placeholder}
            className={inputClassName}
          />
        )}
      </div>
    );
  }

  return (
    <div
      onClick={(e) => {
        onCellClick?.();
        e.stopPropagation();
        setEditing(true);
      }}
      className={cn(
        // 撑满整个单元格
        "relative w-full h-full min-h-[32px] flex items-center",
        // hover 效果
        "px-2 py-1.5 rounded-md cursor-pointer",
        "border border-transparent",
        "hover:bg-muted/40 hover:border-border/40",
        "group/cell transition-all duration-150",
        // 字体
        "text-xs leading-relaxed",
        !value && "text-muted-foreground/30 italic",
        className
      )}
      title="点击编辑"
    >
      <span className="flex-1 whitespace-pre-wrap break-words">{value || placeholder || "\u00a0"}</span>
      <div className="absolute right-1 top-1/2 -translate-y-1/2 h-6 w-6 rounded-lg flex items-center justify-center invisible group-hover/cell:visible backdrop-blur-xl bg-white/70 shadow-sm z-10">
        <Pencil className="h-3.5 w-3.5 text-primary" />
      </div>
    </div>
  );
}
