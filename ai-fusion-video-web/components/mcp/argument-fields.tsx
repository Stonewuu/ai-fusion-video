"use client";

import { useId } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { McpSchemaValueField } from "./schema-value-field";
import type { McpJsonSchema } from "@/lib/api/canvas-mcp";

export function McpArgumentFields({ schema, value, onChange, omitted, disabled }: {
  schema: McpJsonSchema; value: string; onChange: (value: string) => void; omitted: string[]; disabled: boolean;
}) {
  const id = useId();
  let parsed: Record<string, unknown> | null = null;
  try {
    const data = JSON.parse(value);
    if (data && typeof data === "object" && !Array.isArray(data)) parsed = data;
  } catch { /* 保留用户尚未写完的 JSON，完整运行前由服务端校验。 */ }
  const update = (key: string, next: unknown) => {
    if (!parsed) return;
    const result: Record<string, unknown> = Object.assign(Object.create(null), parsed);
    if (next === undefined) delete result[key]; else result[key] = next;
    onChange(JSON.stringify(result, null, 2));
  };
  const fields = Object.entries(schema.properties ?? {}).filter(([key]) => !omitted.includes(key));
  return (
    <div className="flex flex-col gap-4">
      {parsed ? fields.map(([key, field]) => {
        const current = parsed![key];
        const label = field.title ?? key;
        const required = schema.required?.includes(key);
        const primitive = ["string", "number", "integer", "boolean"].includes(String(field.type));
        return <div key={key} className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2">
            <label htmlFor={`${id}-${key}`} className="text-sm font-medium">{label}{required ? " *" : ""}</label>
            {current !== undefined ? <Button size="xs" variant="ghost" aria-label={`重置${label}`} disabled={disabled} onClick={() => update(key, undefined)}>重置</Button> : null}
          </div>
          <McpSchemaValueField schema={field} label={label} value={current} disabled={disabled} onChange={(next) => update(key, next)} />
          {field.description && primitive ? <p className="text-xs text-muted-foreground whitespace-pre-wrap">{field.description}</p> : null}
        </div>;
      }) : <p role="alert" className="text-sm text-destructive">参数暂不是有效的 JSON 对象，请在下方继续编辑。</p>}
      <details open={!parsed || !fields.length}
        className="rounded-lg border border-border/20 bg-background/70 p-3">
        <summary className="cursor-pointer text-sm">高级：完整参数（JSON）</summary>
        <div className="flex flex-col gap-2 pt-3">
          <label htmlFor={`${id}-json`} className="text-xs text-muted-foreground">完整参数</label>
          <Textarea id={`${id}-json`} rows={6} value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)} />
          {omitted.length ? <p className="text-xs text-muted-foreground">已连接的内容和自动生成的参数会在运行时填入。</p> : null}
        </div>
      </details>
    </div>
  );
}
