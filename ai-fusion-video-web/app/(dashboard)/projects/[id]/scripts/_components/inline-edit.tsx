"use client";

import { useEffect, useState, useRef } from "react";
import { cn } from "@/lib/utils";

export function InlineEdit({
  value,
  placeholder,
  onChange,
  onDirty,
  className,
  multiline,
}: {
  value: string;
  placeholder?: string;
  onChange: (val: string) => void;
  onDirty?: () => void;
  className?: string;
  multiline?: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  useEffect(() => {
    if (editing) inputRef.current?.focus();
  }, [editing]);

  const handleSave = () => {
    setEditing(false);
    if (draft.trim() !== value) {
      onChange(draft.trim());
    }
  };

  if (editing) {
    const commonProps = {
      value: draft,
      onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setDraft(e.target.value);
        onDirty?.();
      },
      onBlur: handleSave,
      onKeyDown: (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !multiline) handleSave();
        if (e.key === "Escape") {
          setEditing(false);
          setDraft(value);
        }
      },
      className: cn(
        "w-full px-1 py-0.5 rounded-md text-sm",
        "bg-transparent border border-transparent",
        "focus:outline-none focus:bg-muted/30",
        "transition-all",
        className
      ),
      placeholder,
    };

    if (multiline) {
      return (
        <textarea
          ref={inputRef as React.RefObject<HTMLTextAreaElement>}
          rows={3}
          {...commonProps}
        />
      );
    }
    return (
      <input
        ref={inputRef as React.RefObject<HTMLInputElement>}
        type="text"
        {...commonProps}
      />
    );
  }

  return (
    <span
      onClick={() => setEditing(true)}
      className={cn(
        "cursor-pointer hover:bg-muted/30 px-1 py-0.5 rounded transition-colors inline-block border border-transparent",
        !value && "text-muted-foreground/40 italic",
        className
      )}
      title="点击编辑"
    >
      {value || placeholder || "点击编辑"}
    </span>
  );
}
