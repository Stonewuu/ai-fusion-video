"use client";

import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { CanvasBooleanField, CanvasChoiceField, CanvasNumberField } from "@/components/canvas/forms/field-controls";
import { CanvasStructuredValue } from "@/components/canvas/forms/structured-value-editor";
import type { McpJsonSchema } from "@/lib/api/canvas-mcp";

export function McpSchemaValueField({ schema, label, value, onChange, disabled }: {
  schema: McpJsonSchema; label: string; value: unknown; onChange: (value: unknown) => void; disabled: boolean;
}) {
  if (schema.enum?.length) return <CanvasChoiceField label={label} value={value === undefined ? null : JSON.stringify(value)} disabled={disabled}
    items={schema.enum.map((item) => ({ value: JSON.stringify(item), label: typeof item === "string" ? item : JSON.stringify(item) }))}
    placeholder={`选择${label}`} onChange={(next) => onChange(JSON.parse(next))} />;
  if (schema.type === "boolean") return <CanvasBooleanField label={label} value={typeof value === "boolean" ? value : undefined} disabled={disabled} onChange={onChange}
    defaultLabel={schema.default === true ? "默认开启" : schema.default === false ? "默认关闭" : "未设置"} defaultTitle="不指定此参数" />;
  if (schema.type === "number" || schema.type === "integer") return <CanvasNumberField label={label} value={typeof value === "number" ? value : undefined}
    min={typeof schema.minimum === "number" ? schema.minimum : undefined} max={typeof schema.maximum === "number" ? schema.maximum : undefined}
    step={typeof schema.multipleOf === "number" ? schema.multipleOf : schema.type === "integer" ? 1 : 0.1} disabled={disabled} onChange={onChange} />;
  if (schema.type === "string") return typeof schema.maxLength === "number" && schema.maxLength <= 256
    ? <Input aria-label={label} value={typeof value === "string" ? value : ""} maxLength={schema.maxLength} disabled={disabled} onChange={(event) => onChange(event.target.value)} />
    : <Textarea aria-label={label} value={typeof value === "string" ? value : ""} rows={3} maxLength={typeof schema.maxLength === "number" ? schema.maxLength : undefined}
        disabled={disabled} onChange={(event) => onChange(event.target.value)} />;
  if (schema.type === "object" && schema.properties && (value === undefined || value !== null && typeof value === "object" && !Array.isArray(value))) {
    const data = (value ?? {}) as Record<string, unknown>;
    return <div className="flex min-w-0 flex-col gap-4 border-l border-border/30 pl-3">
      {Object.entries(schema.properties).map(([key, field]) => <div key={key} className="flex min-w-0 flex-col gap-2">
        <span className="text-sm font-medium">{field.title ?? key}{schema.required?.includes(key) ? " *" : ""}</span>
        <McpSchemaValueField schema={field} label={`${label} · ${field.title ?? key}`} value={data[key]} disabled={disabled}
          onChange={(next) => { const result = { ...data }; if (next === undefined) delete result[key]; else result[key] = next; onChange(result); }} />
        {field.description && <p className="text-xs text-muted-foreground">{field.description}</p>}
      </div>)}
    </div>;
  }
  if (schema.type === "array" && schema.items && typeof schema.items === "object" && !Array.isArray(schema.items) && (value === undefined || Array.isArray(value))) {
    const items = (value ?? []) as unknown[];
    const itemSchema = schema.items as McpJsonSchema;
    return <div className="flex min-w-0 flex-col gap-3">
      {items.map((item, index) => <div key={index} className="flex min-w-0 flex-col gap-2 border-b border-border/20 pb-3">
        <div className="flex items-center justify-between gap-2"><span className="text-xs text-muted-foreground">第 {index + 1} 项</span>
          <Button variant="destructive-ghost" size="icon-xs" disabled={disabled} aria-label={`移除${label}第${index + 1}项`} onClick={() => onChange(items.filter((_, current) => current !== index))}><Trash2 /></Button></div>
        <McpSchemaValueField schema={itemSchema} label={`${label}第${index + 1}项`} value={item} disabled={disabled}
          onChange={(next) => onChange(items.map((previous, current) => index === current ? next ?? null : previous))} />
      </div>)}
      <Button variant="outline" disabled={disabled || typeof schema.maxItems === "number" && items.length >= schema.maxItems} onClick={() => onChange([...items,
        itemSchema.type === "object" ? {} : itemSchema.type === "array" ? [] : itemSchema.type === "boolean" ? false : ["number", "integer"].includes(String(itemSchema.type)) ? 0 : "",
      ])}><Plus />添加一项</Button>
    </div>;
  }
  return <CanvasStructuredValue label={label} value={value ?? null} onChange={onChange} disabled={disabled} />;
}
