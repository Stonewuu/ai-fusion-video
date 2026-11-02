"use client";
import { nodeGuidance } from "@/lib/canvas/node-guidance";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Search, Plus, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { compatibleNodes, type CompatibleNode } from "@/lib/canvas/graph";
import { canvasCategories, canvasPortLabels } from "@/lib/canvas/labels";
import type { CanvasAddIntent, CanvasCapability } from "@/lib/canvas/types";
import { cn } from "@/lib/utils";
import { useCanvas } from "../canvas-provider";

export type CanvasNodeChoice = {
  capability: CanvasCapability;
  connection?: CompatibleNode;
};
const commonKinds = [
  "story",
  "text",
  "image",
  "video",
  "audio",
  "imageGeneration",
  "videoGeneration",
  "note",
];
export function CanvasNodeMenu({
  intent,
  onClose,
  onChoose,
}: {
  intent: CanvasAddIntent;
  onClose: (restoreFocus: boolean) => void;
  onChoose: (choice: CanvasNodeChoice) => void;
}) {
  const snapshot = useCanvas((state) => state.snapshot!);
  const catalog = useCanvas((state) => state.catalog!);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<string | null>(
    intent.source ? null : "common",
  );
  const [active, setActive] = useState(0);
  const menu = useRef<HTMLDivElement>(null);
  const [menuPosition, setMenuPosition] = useState(intent.screen);
  useLayoutEffect(() => {
    const element = menu.current;
    if (!element) {
      return;
    }
    const update = () => {
      const anchor = intent.anchor?.getBoundingClientRect();
      const width = element.offsetWidth;
      const height = element.offsetHeight;
      const left = Math.max(8, Math.min(anchor?.left ?? intent.screen.x, window.innerWidth - width - 8));
      let top = anchor ? anchor.bottom + 8 : intent.screen.y;
      if (anchor && top + height > window.innerHeight - 8 && anchor.top - height - 8 >= 8) {
        top = anchor.top - height - 8;
      }
      top = Math.max(8, Math.min(top, window.innerHeight - height - 8));
      setMenuPosition((current) => current.x === left && current.y === top ? current : { x: left, y: top });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    if (intent.anchor) {
      observer.observe(intent.anchor);
    }
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [intent]);
  const source = snapshot.graph.nodes.find(
    (node) => node.id === intent.source?.nodeId,
  );
  const output = catalog.nodes
    .find((item) => item.kind === source?.kind)
    ?.outputs.find((port) => port.id === intent.source?.portId);
  const outputLabel = output ? canvasPortLabels[output.id] ?? output.label : null;
  const stale =
    !!intent.source &&
    (!source || source.configVersion !== intent.source.configVersion);
  const choices = useMemo<CanvasNodeChoice[]>(() => {
    if (intent.source) {
      return source && output
        ? compatibleNodes(
            source,
            output,
            catalog.nodes,
            snapshot.document.projectId,
          )
            .map((connection) => ({
              capability: connection.capability,
              connection,
            }))
        : [];
    }
    return catalog.nodes
      .filter(
        (item) => !item.requiresProject || snapshot.document.projectId !== null,
      )
      .map((capability) => ({ capability }));
  }, [
    intent.source,
    source,
    output,
    catalog,
    snapshot.document.projectId,
  ]);
  const categories = [
    ...new Set(choices.map((choice) => choice.capability.category)),
  ];
  const filtered = choices.filter(
    (choice) =>
      (!!search.trim() ||
        !category ||
        (category === "common"
          ? commonKinds.includes(choice.capability.kind)
          : choice.capability.category === category)) &&
      `${choice.capability.label} ${canvasCategories[choice.capability.category]} ${choice.connection ? canvasPortLabels[choice.connection.port.id] : ""}`
        .toLowerCase()
        .includes(search.trim().toLowerCase()),
  );
  if (category === "common" && !search.trim()) {
    filtered.sort((left, right) => commonKinds.indexOf(left.capability.kind) - commonKinds.indexOf(right.capability.kind));
  }
  const index = Math.min(active, Math.max(0, filtered.length - 1));
  useEffect(() => {
    menu.current?.querySelector("input")?.focus();
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && !menu.current?.contains(event.target)) {
        onClose(false);
      }
    };
    document.addEventListener("pointerdown", outside);
    return () => {
      document.removeEventListener("pointerdown", outside);
    };
  }, [onClose]);
  useEffect(() => {
    menu.current
      ?.querySelector(`[data-option-index="${index}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [index]);
  return createPortal(
    <div
      ref={menu}
      className={cn(
        "canvas-node-menu fixed z-70 flex max-w-[calc(100vw-1rem)] flex-col overflow-hidden rounded-2xl border border-border/40 bg-popover/80 shadow-xl backdrop-blur-xl",
        intent.source ? "w-88" : "w-80",
      )}
      style={{ left: menuPosition.x, top: menuPosition.y, maxHeight: Math.min(intent.source ? 520 : 420, window.innerHeight - 16) }}
      role="dialog"
      aria-label={intent.source ? "添加后续节点" : "添加节点"}
      onContextMenu={(event) => event.preventDefault()}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) {
          return;
        }
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          onClose(true);
        }
        if (
          event.target instanceof Element &&
          event.target.closest("[data-categories]")
        ) {
          return;
        }
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          setActive((current) =>
            Math.max(
              0,
              Math.min(
                filtered.length - 1,
                current + (event.key === "ArrowDown" ? 1 : -1),
              ),
            ),
          );
        }
        if (
          event.key === "Enter" &&
          event.target instanceof HTMLInputElement &&
          filtered[index] &&
          !stale
        ) {
          event.preventDefault();
          onChoose(filtered[index]);
        }
      }}
    >
      <div className="flex shrink-0 flex-col gap-2 border-b border-border/20 p-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm font-medium">
            {intent.source ? "添加后续节点" : "添加节点"}
          </p>
          <Button
            variant="ghost"
            size="icon-xs"
            title="关闭添加菜单"
            aria-label="关闭添加菜单"
            onClick={() => onClose(true)}
          >
            <X />
          </Button>
        </div>
        {source && output && (
          <p
            className="truncate text-xs text-muted-foreground"
            title={`来自「${source.title}」的${outputLabel}输出`}
          >
            来自「{source.title}」的{outputLabel}输出
          </p>
        )}
        {!intent.source && (
          <p className="text-xs text-muted-foreground">
            选择类型直接添加，随后在右侧填写内容或配置模型。
          </p>
        )}
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted-foreground" />
          <Input
            className="pl-9"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setActive(0);
            }}
            placeholder="搜索节点"
            aria-label="搜索节点"
            role="combobox"
            aria-controls="canvas-node-options"
            aria-expanded
            aria-activedescendant={
              filtered[index] ? `canvas-node-option-${index}` : undefined
            }
          />
        </div>
        <div
          className={cn(
            "gap-2",
            intent.source ? "grid grid-cols-3" : "flex flex-wrap",
          )}
          data-categories
          aria-label="节点分类"
        >
          {[...(!intent.source ? ["common"] : []), null, ...categories].map(
            (value) => {
              const label = value === "common"
                ? "常用"
                : value
                  ? canvasCategories[value]
                  : "全部";
              return (
                <Button
                  key={value ?? "all"}
                  size="xs"
                  className={intent.source ? "min-w-0 w-full" : undefined}
                  variant={
                    !search.trim() && category === value ? "secondary" : "ghost"
                  }
                  title={label}
                  aria-pressed={!search.trim() && category === value}
                  onClick={() => {
                    setSearch("");
                    setCategory(value);
                    setActive(0);
                  }}
                >
                  <span className="truncate">{label}</span>
                </Button>
              );
            },
          )}
        </div>
      </div>
      {stale ? (
        <p role="alert" className="p-3 text-sm text-destructive">
          源节点已发生变化，请重新选择输出端口。
        </p>
      ) : (
        <div
          id="canvas-node-options"
          role="listbox"
          className="flex min-h-0 flex-col gap-2 overflow-y-auto p-2"
        >
          {filtered.map((choice, optionIndex) => {
            const inputLabel = choice.connection
              ? canvasPortLabels[choice.connection.port.id] ?? choice.connection.port.label
              : null;
            return (
              <Button
                type="button"
                key={`${choice.capability.kind}:${choice.connection?.port.id ?? ""}`}
                id={`canvas-node-option-${optionIndex}`}
                data-option-index={optionIndex}
                variant={optionIndex === index ? "secondary" : "ghost"}
                size={choice.connection ? "default" : "sm"}
                className="w-full justify-start"
                role="option"
                tabIndex={-1}
                title={choice.connection
                  ? `添加「${choice.capability.label}」，连接到「${inputLabel}」输入\n${nodeGuidance[choice.capability.kind]}`
                  : nodeGuidance[choice.capability.kind]}
                aria-selected={optionIndex === index}
                onPointerMove={() => setActive(optionIndex)}
                onClick={() => onChoose(choice)}
              >
                <CanvasNodeTypeIcon capability={choice.capability} aria-hidden style={{ color: canvasNodeColor(choice.capability.kind) }} />
                <span className="flex min-w-0 flex-1 items-center gap-2 text-left">
                  <span className="min-w-0 truncate">
                    {choice.capability.label}
                  </span>
                  {choice.connection && (
                    <Badge
                      variant="secondary"
                      className="min-w-0 max-w-40 font-normal text-muted-foreground"
                    >
                      <span className="truncate">接入：{inputLabel}</span>
                    </Badge>
                  )}
                </span>
                {choice.connection ? (
                  <Plus aria-hidden />
                ) : (
                  <span className="text-xs text-muted-foreground">
                    {canvasCategories[choice.capability.category]}
                  </span>
                )}
              </Button>
            );
          })}
          {!filtered.length && (
            <p className="p-3 text-sm text-muted-foreground">
              {intent.source ? "没有找到匹配的后续步骤。请换一个关键词，或选择其他输出连接点。" : "没有找到匹配的节点。试试“图片”“分镜”或切换分类。"}
            </p>
          )}
        </div>
      )}
    </div>,
    document.body,
  );
}
