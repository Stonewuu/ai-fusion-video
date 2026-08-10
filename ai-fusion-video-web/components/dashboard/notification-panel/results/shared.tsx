"use client";

import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { getFieldLabel } from "../constants";

export function formatResultValue(val: unknown): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "boolean") return val ? "是" : "否";
  if (typeof val === "number") return String(val);
  if (typeof val === "string") {
    return val.length > 150 ? `${val.slice(0, 150)}…` : val;
  }
  if (Array.isArray(val)) return `[${val.length} 项]`;
  if (typeof val === "object") return `{${Object.keys(val).length} 个字段}`;
  return String(val);
}

export function GenericResult({ data }: { data: unknown }) {
  if (typeof data !== "object" || data === null) {
    return (
      <p className="text-xs text-muted-foreground">
        {formatResultValue(data)}
      </p>
    );
  }

  const obj = data as Record<string, unknown>;
  if (typeof obj.status === "string" && typeof obj.message === "string") {
    return (
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground inline-flex items-center gap-1">
          {obj.status === "error" || obj.status === "failed" ? (
            <AlertTriangle className="h-3.5 w-3.5 text-yellow-500 shrink-0" />
          ) : (
            <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />
          )}
          {obj.message as string}
        </p>
        {Object.entries(obj)
          .filter(([key]) => key !== "status" && key !== "message")
          .slice(0, 4)
          .map(([key, value]) => (
            <p key={key} className="text-[10px] text-muted-foreground/60">
              {getFieldLabel(key)}: {formatResultValue(value)}
            </p>
          ))}
      </div>
    );
  }

  if (Array.isArray(data)) {
    return (
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground">
          返回 <span className="font-medium text-foreground">{data.length}</span> 条记录
        </p>
        {data.slice(0, 5).map((item, index) => (
          <div
            key={index}
            className="flex items-center gap-2 text-xs text-muted-foreground/90"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 shrink-0" />
            <span>
              {typeof item === "object" && item !== null
                ? String(
                    (item as Record<string, unknown>).name ||
                      (item as Record<string, unknown>).title ||
                      (item as Record<string, unknown>).id ||
                      JSON.stringify(item).slice(0, 80)
                  )
                : formatResultValue(item)}
            </span>
          </div>
        ))}
        {data.length > 5 && (
          <p className="text-[10px] text-muted-foreground/60 pl-3">
            …还有 {data.length - 5} 条
          </p>
        )}
      </div>
    );
  }

  const priorityKeys = [
    "status",
    "message",
    "total",
    "id",
    "name",
    "title",
    "type",
    "count",
  ];
  const displayEntries = Object.entries(obj)
    .sort((a, b) => {
      const ai = priorityKeys.indexOf(a[0]);
      const bi = priorityKeys.indexOf(b[0]);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return 0;
    })
    .filter(([, value]) => !(typeof value === "string" && value.length > 300));

  return (
    <div className="space-y-0.5">
      {displayEntries.slice(0, 8).map(([key, value]) => (
        <div key={key} className="flex items-baseline gap-2 text-xs">
          <span className="text-muted-foreground/70 shrink-0">
            {getFieldLabel(key)}:
          </span>
          <span className="text-muted-foreground">
            {formatResultValue(value)}
          </span>
        </div>
      ))}
      {displayEntries.length > 8 && (
        <p className="text-[10px] text-muted-foreground/60">
          …还有 {displayEntries.length - 8} 个字段
        </p>
      )}
    </div>
  );
}