"use client";

import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
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
import { getToolArgumentItemLabel } from "./tool-call-argument-metadata";
import { ToolCallJson } from "./tool-call-json";

const LONG_TEXT_LENGTH = 240;
const SUMMARY_EXCLUDED_FIELDS = new Set([
  "content",
  "description",
  "dialogue",
  "framePrompt",
  "message",
  "music",
  "prompt",
  "rawContent",
  "remark",
  "sceneDescription",
  "sceneExpectation",
  "sound",
  "soundEffect",
  "storySynopsis",
  "synopsis",
  "videoPrompt",
]);

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
  plain = false,
}: {
  value: ToolArguments;
  depth: number;
  toolName: string;
  plain?: boolean;
}) {
  const entries = listToolArgumentEntries(value, toolName);
  if (entries.length === 0) {
    return <p className="text-xs text-muted-foreground">无内容</p>;
  }

  return (
    <div className={depth === 0 || plain ? "divide-y divide-border/20" : "divide-y divide-border/20 rounded-lg border border-border/20 px-2"}>
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

function isPrimitiveArray(value: ToolArgumentValue[]): boolean {
  return value.every((item) => !Array.isArray(item) && !isObject(item));
}

function getItemSummary(
  value: ToolArgumentValue,
  toolName: string,
): string | undefined {
  if (!isObject(value)) return undefined;

  const values = listToolArgumentEntries(value, toolName)
    .flatMap((entry) => {
      if (
        SUMMARY_EXCLUDED_FIELDS.has(entry.key)
        || Array.isArray(entry.value)
        || isObject(entry.value)
        || entry.value === null
        || typeof entry.value === "boolean"
        || entry.label.includes("ID")
      ) {
        return [];
      }
      const formatted = formatToolArgumentScalar(entry.key, entry.value, toolName);
      if (formatted.length > 18 || isUrl(formatted)) return [];
      return [formatted];
    })
    .filter((value, index, allValues) => allValues.indexOf(value) === index)
    .slice(0, 3);

  return values.length > 0 ? values.join(" · ") : undefined;
}

function ArrayItemValue({
  fieldKey,
  toolName,
  value,
  depth,
  index,
  total,
}: {
  fieldKey: string;
  toolName: string;
  value: ToolArgumentValue;
  depth: number;
  index: number;
  total: number;
}) {
  const [expanded, setExpanded] = useState(total <= 3);
  const itemLabel = getToolArgumentItemLabel(fieldKey);
  const summary = getItemSummary(value, toolName);
  const itemNumber = String(index + 1).padStart(String(total).length, "0");
  const displayLabel = itemLabel ?? "项目";

  return (
    <div className="overflow-hidden rounded-lg border border-border/20 bg-background/70">
      <Button
        type="button"
        variant="ghost"
        className="w-full justify-start"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        <span className="flex size-5 shrink-0 items-center justify-center rounded-md bg-muted text-[11px] font-semibold tabular-nums text-foreground">
          {itemNumber}
        </span>
        <span className="shrink-0">{displayLabel}</span>
        {summary ? (
          <span className="min-w-0 flex-1 truncate text-left text-xs font-normal text-muted-foreground">
            {summary}
          </span>
        ) : (
          <span className="flex-1" />
        )}
        {expanded ? (
          <ChevronDown className="text-muted-foreground" aria-hidden="true" />
        ) : (
          <ChevronRight className="text-muted-foreground" aria-hidden="true" />
        )}
      </Button>
      {expanded ? (
        <div className="border-t border-border/20 px-3 pb-1">
          <ArgumentValueView
            fieldKey={fieldKey}
            value={value}
            depth={depth + 1}
            toolName={toolName}
            plainObject
          />
        </div>
      ) : null}
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
  if (isPrimitiveArray(value)) {
    return (
      <div className="flex min-w-0 flex-wrap items-baseline gap-y-0.5">
        {value.map((item, index) => (
          <div key={`${fieldKey}-${index}`} className="flex min-w-0 items-baseline">
            <div className="min-w-0">
              <ScalarValue
                fieldKey={fieldKey}
                toolName={toolName}
                value={item as null | boolean | number | string}
              />
            </div>
            {index < value.length - 1 ? (
              <span className="mr-1 text-muted-foreground" aria-hidden="true">、</span>
            ) : null}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {value.map((item, index) => (
        <ArrayItemValue
          key={`${fieldKey}-${index}`}
          fieldKey={fieldKey}
          value={item}
          depth={depth}
          index={index}
          total={value.length}
          toolName={toolName}
        />
      ))}
    </div>
  );
}

function ArgumentValueView({
  fieldKey,
  toolName,
  value,
  depth,
  plainObject = false,
}: {
  fieldKey: string;
  toolName: string;
  value: ToolArgumentValue;
  depth: number;
  plainObject?: boolean;
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
    return (
      <ObjectValue
        value={value}
        depth={depth}
        toolName={toolName}
        plain={plainObject}
      />
    );
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
  const primitiveArray = Array.isArray(entry.value) && isPrimitiveArray(entry.value);
  const collection = isObject(entry.value) || (Array.isArray(entry.value) && !primitiveArray);
  const label = primitiveArray ? entry.label.replace(/列表$/, "").trimEnd() : entry.label;
  return (
    <div className={collection ? "py-2" : "grid grid-cols-[minmax(88px,0.3fr)_minmax(0,1fr)] gap-3 py-2 text-xs"}>
      <div className={collection ? "mb-1.5 flex items-center gap-2" : "text-muted-foreground"}>
        <span className="font-medium">{label}</span>
        {Array.isArray(entry.value) && !primitiveArray ? (
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
