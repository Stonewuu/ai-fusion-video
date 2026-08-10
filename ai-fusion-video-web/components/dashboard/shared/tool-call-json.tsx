"use client";

import { useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Copy } from "lucide-react";
import { useTheme } from "next-themes";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneDark,
  oneLight,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";

const VIRTUALIZED_CHARACTER_THRESHOLD = 8_000;
const VIRTUALIZED_LINE_THRESHOLD = 120;
const MAX_HIGHLIGHTED_LINE_LENGTH = 4_000;
const JSON_LINE_HEIGHT = 18;
const JSON_VIEWPORT_HEIGHT = 288;

function CopyJsonButton({ code, label }: { code: string; label: string }) {
  const copy = async () => {
    if (!navigator.clipboard) {
      toast.error("当前浏览器不支持剪贴板写入");
      return;
    }
    try {
      await navigator.clipboard.writeText(code);
      toast.success(`${label}已复制`);
    } catch (error) {
      toast.error("复制 JSON 失败", {
        description: error instanceof Error ? error.message : "剪贴板写入被拒绝",
      });
    }
  };

  return (
    <span className="absolute right-2 top-2 z-10 rounded-xl border border-border/40 bg-popover/70 backdrop-blur-xl shadow-sm">
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        onClick={copy}
        aria-label={`复制${label}`}
        title={`复制${label}`}
      >
        <Copy />
      </Button>
    </span>
  );
}

function HighlightedJsonLine({
  line,
  dark,
}: {
  line: string;
  dark: boolean;
}) {
  if (line.length > MAX_HIGHLIGHTED_LINE_LENGTH) {
    return (
      <code className="block whitespace-pre pl-3 pr-11 font-mono text-[11px] text-foreground/80">
        {line || " "}
      </code>
    );
  }

  return (
    <SyntaxHighlighter
      language="json"
      style={dark ? oneDark : oneLight}
      customStyle={{
        background: "transparent",
        fontSize: "11px",
        height: `${JSON_LINE_HEIGHT}px`,
        lineHeight: `${JSON_LINE_HEIGHT}px`,
        margin: 0,
        overflow: "visible",
        padding: "0 44px 0 12px",
        whiteSpace: "pre",
      }}
      PreTag="div"
    >
      {line || " "}
    </SyntaxHighlighter>
  );
}

function VirtualizedToolCallJson({
  code,
  lines,
  dark,
  label,
}: {
  code: string;
  lines: string[];
  dark: boolean;
  label: string;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  // TanStack Virtual owns a mutable measurement instance and intentionally
  // remains outside React Compiler memoization.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: lines.length,
    estimateSize: () => JSON_LINE_HEIGHT,
    getScrollElement: () => scrollRef.current,
    overscan: 12,
  });

  return (
    <div
      className="relative overflow-hidden rounded-lg border border-border/20 bg-background/70"
      aria-label={label}
      data-json-virtualized="true"
    >
      <CopyJsonButton code={code} label={label} />
      <div
        ref={scrollRef}
        className="overflow-auto"
        style={{ height: `${JSON_VIEWPORT_HEIGHT}px` }}
      >
        <div
          className="relative min-w-full"
          style={{ height: `${virtualizer.getTotalSize()}px` }}
        >
          {virtualizer.getVirtualItems().map((virtualLine) => (
            <div
              key={virtualLine.key}
              className="absolute left-0 top-0 min-w-full"
              style={{
                height: `${virtualLine.size}px`,
                transform: `translateY(${virtualLine.start}px)`,
                width: "max-content",
              }}
            >
              <HighlightedJsonLine line={lines[virtualLine.index] ?? ""} dark={dark} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function ToolCallJson({
  code,
  label,
}: {
  code: string;
  label: string;
}) {
  const { resolvedTheme } = useTheme();
  const lines = useMemo(() => code.split("\n"), [code]);
  const dark = resolvedTheme === "dark";
  const virtualized = code.length > VIRTUALIZED_CHARACTER_THRESHOLD
    || lines.length > VIRTUALIZED_LINE_THRESHOLD;

  if (virtualized) {
    return (
      <VirtualizedToolCallJson
        code={code}
        lines={lines}
        dark={dark}
        label={label}
      />
    );
  }

  return (
    <div
      className="relative overflow-hidden rounded-lg border border-border/20 bg-background/70"
      aria-label={label}
      data-json-virtualized="false"
    >
      <CopyJsonButton code={code} label={label} />
      <div className="max-h-72 overflow-auto">
        <SyntaxHighlighter
          language="json"
          style={dark ? oneDark : oneLight}
          customStyle={{
            background: "transparent",
            fontSize: "11px",
            lineHeight: 1.6,
            margin: 0,
            padding: "12px 44px 12px 12px",
            whiteSpace: "pre",
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
