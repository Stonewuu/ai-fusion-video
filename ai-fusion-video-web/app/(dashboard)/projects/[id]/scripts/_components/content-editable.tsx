"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

export function ContentEditable({
  value,
  placeholder,
  onChange,
  onDirty,
  className,
  maxWidth,
}: {
  value: string;
  placeholder?: string;
  onChange: (val: string) => void;
  onDirty?: () => void;
  className?: string;
  maxWidth?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);

  // 外部 value 变化时同步 DOM（非聚焦状态下）
  useEffect(() => {
    if (ref.current && document.activeElement !== ref.current) {
      ref.current.textContent = value || "";
    }
  }, [value]);

  return (
    <span
      ref={ref}
      contentEditable
      suppressContentEditableWarning
      onBlur={() => {
        const text = ref.current?.textContent?.trim() || "";
        if (text !== value) onChange(text);
      }}
      onInput={() => onDirty?.()}
      onKeyDown={(e) => {
        // 禁止回车换行
        if (e.key === "Enter") {
          e.preventDefault();
          (e.target as HTMLElement).blur();
        }
      }}
      onPaste={(e) => {
        // 粘贴时去除格式，只保留纯文本
        e.preventDefault();
        const text = e.clipboardData.getData("text/plain").replace(/\n/g, " ");
        document.execCommand("insertText", false, text);
      }}
      style={maxWidth ? { maxWidth } : undefined}
      className={cn(
        "outline-none rounded-sm cursor-text transition-colors whitespace-pre-wrap wrap-break-word",
        "hover:bg-muted/10 focus:bg-primary/5",
        className
      )}
      data-placeholder={placeholder}
    >
      {value}
    </span>
  );
}
