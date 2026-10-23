"use client";

import { useLayoutEffect, useRef, useState } from "react";
import { Check, Loader2, Pencil, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getApiErrorMessage } from "@/lib/api/api-error";
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
  onSave: (value: string) => Promise<void>;
  onCellClick?: () => void;
  className?: string;
  multiline?: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const control = useRef<HTMLInputElement | HTMLTextAreaElement>(null);
  const pending = useRef(false);
  // 回调同时捕获开始编辑时的业务对象版本；刷新不能替换本次草稿的前置条件。
  const session = useRef<{ value: string; save: typeof onSave } | null>(null);

  useLayoutEffect(() => {
    if (!editing || !control.current) return;
    control.current.focus();
    control.current.scrollIntoView({ block: "nearest", inline: "nearest" });
    control.current.setSelectionRange(
      control.current.value.length,
      control.current.value.length,
    );
  }, [editing]);

  useLayoutEffect(() => {
    if (!editing || !multiline || !control.current) return;
    const textarea = control.current as HTMLTextAreaElement;
    textarea.style.height = "auto";
    textarea.style.height = String(Math.min(textarea.scrollHeight, 320)) + "px";
  }, [editing, multiline, draft]);

  const begin = () => {
    onCellClick?.();
    session.current = { value, save: onSave };
    setDraft(value);
    setError("");
    setEditing(true);
  };
  const cancel = () => {
    if (pending.current) return;
    session.current = null;
    setEditing(false);
    setError("");
  };
  const commit = async () => {
    const current = session.current;
    if (!current || pending.current) return;
    if (draft === current.value) {
      cancel();
      return;
    }
    pending.current = true;
    setSaving(true);
    setError("");
    try {
      await current.save(draft);
      session.current = null;
      setEditing(false);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      pending.current = false;
      setSaving(false);
    }
  };

  if (!editing)
    return (
      <button
        type="button"
        title="点击编辑"
        onClick={(event) => {
          event.stopPropagation();
          begin();
        }}
        className={cn(
          "group/cell flex min-h-9 w-full items-center gap-2 rounded-xl border border-transparent px-2 py-1.5 text-left outline-none transition-colors hover:border-border/40 hover:bg-muted/40 focus-visible:ring-2 focus-visible:ring-ring",
          className,
        )}
      >
        <span className="flex-1 whitespace-pre-wrap break-words">
          {value || placeholder || "\u00a0"}
        </span>
        <Pencil className="size-3 shrink-0 text-muted-foreground opacity-0 group-hover/cell:opacity-100 group-focus-visible/cell:opacity-100" />
      </button>
    );

  const inputClass = cn(
    "min-h-9 w-full scroll-m-1 rounded-lg border border-border/40 bg-background p-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-60",
    className,
  );
  const changed = session.current !== null && value !== session.current.value;
  return (
    <div
      className="w-full space-y-2"
      onClick={(event) => event.stopPropagation()}
    >
      {multiline ? (
        <textarea
          ref={control as React.RefObject<HTMLTextAreaElement>}
          value={draft}
          disabled={saving}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => {
            if (!error) void commit();
          }}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              event.preventDefault();
              cancel();
            }
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
              event.preventDefault();
              void commit();
            }
          }}
          placeholder={placeholder}
          aria-label="编辑内容"
          className={cn(inputClass, "resize-none overflow-y-auto")}
        />
      ) : (
        <input
          ref={control as React.RefObject<HTMLInputElement>}
          value={draft}
          disabled={saving}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => {
            if (!error) void commit();
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void commit();
            }
            if (event.key === "Escape") {
              event.preventDefault();
              cancel();
            }
          }}
          placeholder={placeholder}
          aria-label="编辑内容"
          className={inputClass}
        />
      )}
      {changed && (
        <p className="text-xs text-muted-foreground">
          内容已更新，本地草稿已保留。
        </p>
      )}
      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}
      {(error || saving) && (
        <div className="flex gap-2">
          <Button
            size="icon-xs"
            variant="outline"
            disabled={saving}
            title="重试保存"
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => void commit()}
          >
            {saving ? <Loader2 className="animate-spin" /> : <Check />}
          </Button>
          <Button
            size="icon-xs"
            variant="ghost"
            disabled={saving}
            title="放弃本地修改"
            onMouseDown={(event) => event.preventDefault()}
            onClick={cancel}
          >
            <X />
          </Button>
        </div>
      )}
    </div>
  );
}
