"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  formatToolArgumentScalar,
  formatToolArgumentsJson,
  listToolArgumentEntries,
  parseEmbeddedJsonArgument,
  parseToolArguments,
  type ToolArgumentEntry,
  type ToolArguments,
  type ToolArgumentValue,
} from "./tool-call-argument-model";
import { ToolCallJson } from "./tool-call-json";

const LONG_TEXT_LENGTH = 240;

function isObject(value: ToolArgumentValue): value is ToolArguments {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isUrl(value: string): boolean {
  return value.startsWith("https://") || value.startsWith("http://");
}

function LongTextValue({ value }: { value: string }) {
  const [expanded, setExpanded] = useState(false);
  const visibleValue = expanded ? value : `${value.slice(0, LONG_TEXT_LENGTH)}…`;

  return (
    <div className="min-w-0">
      <p className="whitespace-pre-wrap break-words leading-relaxed">{visibleValue}</p>
      <Button
        type="button"
        variant="ghost"
        size="xs"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        {expanded ? "收起全文" : "展开全文"}
      </Button>
    </div>
  );
}

function ScalarValue({
  fieldKey,
  toolName,
  value,
}: {
  fieldKey: string;
  toolName: string;
  value: null | boolean | number | string;
}) {
  const formatted = formatToolArgumentScalar(fieldKey, value, toolName);
  if (typeof value === "string" && isUrl(value)) {
    return (
      <a
        href={value}
        target="_blank"
        rel="noreferrer"
        className="break-all text-primary underline-offset-4 hover:underline"
      >
        {value}
      </a>
    );
  }
  if (typeof value === "string" && formatted.length > LONG_TEXT_LENGTH) {
    return <LongTextValue value={formatted} />;
  }
  return <span className="whitespace-pre-wrap break-words leading-relaxed">{formatted}</span>;
}

function ObjectValue({
  value,
  depth,
  toolName,
}: {
  value: ToolArguments;
  depth: number;
  toolName: string;
}) {
  const entries = listToolArgumentEntries(value, toolName);
  if (entries.length === 0) {
    return <p className="text-xs text-muted-foreground">无内容</p>;
  }

  return (
    <div className={depth === 0 ? "divide-y divide-border/20" : "divide-y divide-border/20 rounded-lg border border-border/20 px-2"}>
      {entries.map((entry) => (
        <ArgumentEntryView
          key={entry.key}
          entry={entry}
          depth={depth}
          toolName={toolName}
        />
      ))}
    </div>
  );
}

function ArrayValue({
  fieldKey,
  toolName,
  value,
  depth,
}: {
  fieldKey: string;
  toolName: string;
  value: ToolArgumentValue[];
  depth: number;
}) {
  if (value.length === 0) {
    return <span className="text-muted-foreground">无</span>;
  }
  const primitiveOnly = value.every((item) => !Array.isArray(item) && !isObject(item));
  if (primitiveOnly) {
    return (
      <div className="flex flex-wrap gap-1.5">
        {value.map((item, index) => (
          <span
            key={`${fieldKey}-${index}`}
            className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground"
          >
            <ScalarValue
              fieldKey={fieldKey}
              toolName={toolName}
              value={item as null | boolean | number | string}
            />
          </span>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {value.map((item, index) => (
        <div key={`${fieldKey}-${index}`} className="rounded-lg border border-border/20 p-2">
          <p className="mb-1.5 text-[11px] font-medium text-muted-foreground">
            第 {index + 1} 项
          </p>
          <ArgumentValueView
            fieldKey={fieldKey}
            value={item}
            depth={depth + 1}
            toolName={toolName}
          />
        </div>
      ))}
    </div>
  );
}

function ArgumentValueView({
  fieldKey,
  toolName,
  value,
  depth,
}: {
  fieldKey: string;
  toolName: string;
  value: ToolArgumentValue;
  depth: number;
}) {
  if (Array.isArray(value)) {
    return (
      <ArrayValue
        fieldKey={fieldKey}
        value={value}
        depth={depth}
        toolName={toolName}
      />
    );
  }
  if (isObject(value)) {
    return <ObjectValue value={value} depth={depth} toolName={toolName} />;
  }
  if (typeof value === "string") {
    const embeddedJson = parseEmbeddedJsonArgument(fieldKey, value);
    if (embeddedJson !== undefined) {
      return (
        <ArgumentValueView
          fieldKey={fieldKey}
          value={embeddedJson}
          depth={depth + 1}
          toolName={toolName}
        />
      );
    }
  }
  return <ScalarValue fieldKey={fieldKey} value={value} toolName={toolName} />;
}

function ArgumentEntryView({
  entry,
  depth,
  toolName,
}: {
  entry: ToolArgumentEntry;
  depth: number;
  toolName: string;
}) {
  const collection = Array.isArray(entry.value) || isObject(entry.value);
  return (
    <div className={collection ? "py-2" : "grid grid-cols-[minmax(88px,0.3fr)_minmax(0,1fr)] gap-3 py-2 text-xs"}>
      <div className={collection ? "mb-1.5 flex items-center gap-2" : "text-muted-foreground"}>
        <span className="font-medium">{entry.label}</span>
        {Array.isArray(entry.value) ? (
          <span className="text-[11px] text-muted-foreground">{entry.value.length} 项</span>
        ) : null}
      </div>
      <div className="min-w-0 text-foreground/80">
        <ArgumentValueView
          fieldKey={entry.key}
          value={entry.value}
          depth={depth + 1}
          toolName={toolName}
        />
      </div>
    </div>
  );
}

export function ToolCallArguments({
  argumentsText,
  toolName,
  view,
}: {
  argumentsText: string;
  toolName: string;
  view: "friendly" | "json";
}) {
  const argumentsValue = useMemo(
    () => parseToolArguments(argumentsText),
    [argumentsText],
  );
  const entries = listToolArgumentEntries(argumentsValue, toolName);

  return (
    <div>
      {view === "friendly" ? (
        entries.length === 0 ? (
          <p className="py-2 text-xs text-muted-foreground">此工具无需调用参数</p>
        ) : (
          <ObjectValue value={argumentsValue} depth={0} toolName={toolName} />
        )
      ) : (
        <ToolCallJson
          code={formatToolArgumentsJson(argumentsValue)}
          label="调用参数 JSON"
        />
      )}
    </div>
  );
}
