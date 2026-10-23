"use client";

import { useId, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { CanvasChoiceField, CanvasNumberField } from "./field-controls";
import { CanvasSelect } from "./canvas-select";

const types = [
  { value: "string", label: "文字" }, { value: "number", label: "数字" },
  { value: "boolean", label: "开关" }, { value: "object", label: "属性组" },
  { value: "array", label: "列表" }, { value: "null", label: "空值" },
];
const kindOf = (value: unknown) => value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
const empty = (type: string) => ({ string: "", number: 0, boolean: false, object: {}, array: [], null: null })[type];

export function CanvasStructuredValue({ label, value, onChange, disabled = false, arrayItemDefault = "" }: {
  label: string; value: unknown; onChange: (value: unknown) => void; disabled?: boolean; arrayItemDefault?: unknown;
}) {
  const kind = kindOf(value);
  if (kind === "object" || kind === "array") return <ValueCollection label={label} value={value as Record<string, unknown> | unknown[]} onChange={onChange} disabled={disabled} arrayItemDefault={arrayItemDefault} />;
  if (kind === "boolean") return <CanvasChoiceField label={label} value={value ? "on" : "off"} disabled={disabled}
    items={[{ value: "on", label: "开启" }, { value: "off", label: "关闭" }]} onChange={(next) => onChange(next === "on")} />;
  if (kind === "number") return <CanvasNumberField label={label} value={value as number} onChange={(next) => onChange(next ?? null)} disabled={disabled} step={0.1} />;
  if (kind === "null") return <span className="text-xs text-muted-foreground">未填写</span>;
  return <Textarea aria-label={label} value={String(value ?? "")} rows={2} disabled={disabled} onChange={(event) => onChange(event.target.value)} />;
}

function ValueCollection({ label, value, onChange, disabled, arrayItemDefault }: {
  label: string; value: Record<string, unknown> | unknown[]; onChange: (value: unknown) => void; disabled: boolean; arrayItemDefault: unknown;
}) {
  const id = useId();
  const array = Array.isArray(value);
  const entries = Object.entries(value);
  const [keyError, setKeyError] = useState<string | null>(null);
  const update = (key: string, next: unknown) => onChange(array ? value.map((item, index) => String(index) === key ? next : item) : { ...value, [key]: next });
  const remove = (key: string) => onChange(array ? value.filter((_, index) => String(index) !== key) : Object.fromEntries(entries.filter(([name]) => name !== key)));
  return <div className="flex min-w-0 flex-col gap-3">
    {entries.map(([key, item], index) => <div key={key} className="flex min-w-0 flex-col gap-2 border-b border-border/20 pb-3">
      <div className="flex min-w-0 items-center gap-2">
        {array ? <span className="flex-1 text-xs text-muted-foreground">第 {index + 1} 项</span> : <Input key={`${id}-${key}`} aria-label={`${label}第${index + 1}项名称`} defaultValue={key} disabled={disabled}
          onBlur={(event) => {
            const next = event.target.value.trim();
            if (next === key) return;
            if (!next || Object.hasOwn(value, next)) { setKeyError("名称不能为空或与其他属性重复。"); event.target.value = key; return; }
            setKeyError(null); onChange(Object.fromEntries(entries.map(([name, content]) => [name === key ? next : name, content])));
          }} />}
        <Button variant="destructive-ghost" size="icon" aria-label={`删除${array ? `第${index + 1}项` : key}`} disabled={disabled} onClick={() => remove(key)}><Trash2 /></Button>
      </div>
      <CanvasSelect label={`${label}第${index + 1}项类型`} items={types} value={kindOf(item)} placeholder="值类型" disabled={disabled} onChange={(next) => update(key, empty(next))} />
      {item !== null && typeof item === "object" ? <details className="min-w-0" open>
        <summary className="cursor-pointer text-xs text-muted-foreground">{Array.isArray(item) ? "列表内容" : "属性内容"}</summary>
        <div className="min-w-0 pt-3"><CanvasStructuredValue label={`${label} · ${key}`} value={item} onChange={(next) => update(key, next)} disabled={disabled} /></div>
      </details> : <CanvasStructuredValue label={`${label} · ${key}`} value={item} onChange={(next) => update(key, next)} disabled={disabled} />}
    </div>)}
    {keyError && <p role="alert" className="text-xs text-destructive">{keyError}</p>}
    <Button variant="outline" disabled={disabled} onClick={() => {
      if (array) onChange([...value, arrayItemDefault]);
      else { let index = entries.length + 1; while (Object.hasOwn(value, `属性${index}`)) index++; onChange({ ...value, [`属性${index}`]: "" }); }
    }}><Plus />{array ? "添加一项" : "添加属性"}</Button>
  </div>;
}

export function CanvasStructuredDocument({ label, value, jsonType, onChange, disabled }: {
  label: string; value: unknown; jsonType?: "object" | "array"; onChange: (value: unknown) => void; disabled?: boolean;
}) {
  let parsed: unknown = value ?? (jsonType === "array" ? [] : {});
  let error = false;
  if (typeof value === "string") { try { parsed = JSON.parse(value); } catch { error = true; } }
  if (!error && (jsonType === "array" ? !Array.isArray(parsed) : !parsed || typeof parsed !== "object" || Array.isArray(parsed))) error = true;
  return <div className="flex min-w-0 flex-col gap-3">
    {!error && <CanvasStructuredValue label={label} value={parsed} onChange={onChange} disabled={disabled} arrayItemDefault={jsonType === "array" ? {} : ""} />}
    <details open={error || undefined}>
      <summary className="cursor-pointer rounded-lg text-xs text-muted-foreground outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">高级：编辑 JSON</summary>
      <div className="flex flex-col gap-2 pt-3">
        {error && <p role="alert" className="text-xs text-destructive">内容格式不完整，请修正后继续。</p>}
        <Textarea aria-label={`${label} JSON`} value={typeof value === "string" ? value : JSON.stringify(parsed, null, 2)} rows={6}
          disabled={disabled} spellCheck={false} onChange={(event) => onChange(event.target.value)} />
      </div>
    </details>
  </div>;
}
