"use client";

import { useState } from "react";
import { Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function CanvasTagsField({ label, value, onChange, disabled }: {
  label: string; value: string[]; onChange: (value: string[]) => void; disabled?: boolean;
}) {
  const [text, setText] = useState("");
  const add = () => {
    const additions = text.split(/[,，;；\n]/).map((item) => item.trim()).filter(Boolean);
    if (additions.length) { onChange([...new Set([...value, ...additions])]); setText(""); }
  };
  return <div className="flex min-w-0 flex-col gap-3">
    {!!value.length && <div className="flex flex-wrap gap-2">
      {value.map((item, index) => <span key={`${item}-${index}`} className="inline-flex max-w-full items-center gap-1 rounded-lg border border-border/30 bg-muted/50 px-2 py-1 text-xs">
        <span className="break-all">{item}</span>
        <Button variant="ghost" size="icon-xs" aria-label={`移除${item}`} disabled={disabled} onClick={() => onChange(value.filter((_, current) => current !== index))}><X /></Button>
      </span>)}
    </div>}
    <div className="flex min-w-0 items-center gap-2">
      <Input aria-label={`添加${label}`} value={text} disabled={disabled} placeholder="输入后按回车，可用逗号分隔"
        onChange={(event) => setText(event.target.value)} onBlur={add}
        onKeyDown={(event) => { if (event.key === "Enter" && !event.nativeEvent.isComposing) { event.preventDefault(); add(); } }} />
      <Button variant="outline" size="icon" aria-label={`添加${label}`} disabled={disabled || !text.trim()} onClick={add}><Plus /></Button>
    </div>
  </div>;
}
