"use client";
import { nodeGuidance } from "@/lib/canvas/node-guidance";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Search, ArrowRight, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { compatibleNodes, type CompatibleNode } from "@/lib/canvas/graph";
import { canvasCategories, canvasPortLabels } from "@/lib/canvas/labels";
import type { CanvasAddIntent, CanvasCapability } from "@/lib/canvas/types";
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
    if (!element) return;
    const update = () => {
      const anchor = intent.anchor?.getBoundingClientRect();
      const width = element.offsetWidth;
      const height = element.offsetHeight;
      const left = Math.max(8, Math.min(anchor?.left ?? intent.screen.x, window.innerWidth - width - 8));
      let top = anchor ? anchor.bottom + 8 : intent.screen.y;
      if (anchor && top + height > window.innerHeight - 8 && anchor.top - height - 8 >= 8)
        top = anchor.top - height - 8;
      top = Math.max(8, Math.min(top, window.innerHeight - height - 8));
      setMenuPosition((current) => current.x === left && current.y === top ? current : { x: left, y: top });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    if (intent.anchor) observer.observe(intent.anchor);
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
  const stale =
    !!intent.source &&
    (!source || source.configVersion !== intent.source.configVersion);
  const choices = useMemo<CanvasNodeChoice[]>(() => {
    if (intent.source)
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
      if (event.target instanceof Node && !menu.current?.contains(event.target))
        onClose(false);
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
      className="canvas-node-menu fixed z-70 flex w-80 max-w-[calc(100vw-1rem)] flex-col overflow-hidden rounded-2xl border border-border/40 bg-popover/80 shadow-xl backdrop-blur-xl"
      style={{ left: menuPosition.x, top: menuPosition.y, maxHeight: Math.min(420, window.innerHeight - 16) }}
      role="dialog"
      aria-label={intent.source ? "添加后续节点" : "添加节点"}
      onContextMenu={(event) => event.preventDefault()}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return;
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          onClose(true);
        }
        if (
          event.target instanceof Element &&
          event.target.closest("[data-categories]")
        )
          return;
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
      <div className="space-y-2 border-b border-border/20 p-3">
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
          <p className="truncate text-xs text-muted-foreground">
            {source.title} · {canvasPortLabels[output.id] ?? output.label}
          </p>
        )}
        <p className="text-xs text-muted-foreground">
          {intent.source
            ? "按内容类型匹配，添加后自动连线，再到右侧配置。"
            : "选择类型直接添加，随后在右侧填写内容或配置模型。"}
        </p>
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
          className="flex flex-wrap gap-2"
          data-categories
          aria-label="节点分类"
        >
          {[...(!intent.source ? ["common"] : []), null, ...categories].map(
            (value) => (
              <Button
                key={value ?? "all"}
                size="xs"
                variant={
                  !search.trim() && category === value ? "secondary" : "ghost"
                }
                aria-pressed={!search.trim() && category === value}
                onClick={() => {
                  setSearch("");
                  setCategory(value);
                  setActive(0);
                }}
              >
                {value === "common"
                  ? "常用"
                  : value
                    ? canvasCategories[value]
                    : "全部"}
              </Button>
            ),
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
          className="min-h-0 overflow-y-auto p-2"
        >
          {filtered.map((choice, optionIndex) => (
            <Button
              key={`${choice.capability.kind}:${choice.connection?.port.id ?? ""}`}
              id={`canvas-node-option-${optionIndex}`}
              data-option-index={optionIndex}
              variant={optionIndex === index ? "secondary" : "ghost"}
              className="w-full justify-start"
              size="sm"
              role="option"
              title={nodeGuidance[choice.capability.kind]}
              aria-selected={optionIndex === index}
              onPointerMove={() => setActive(optionIndex)}
              onClick={() => onChoose(choice)}
            >
              <CanvasNodeTypeIcon capability={choice.capability} aria-hidden style={{ color: canvasNodeColor(choice.capability.kind) }} />
              <span className="min-w-0 flex-1 truncate text-left">
                {choice.capability.label}
                {choice.connection &&
                  ` · ${canvasPortLabels[choice.connection.port.id] ?? choice.connection.port.label}`}
              </span>
              {choice.connection ? (
                <ArrowRight />
              ) : (
                <span className="text-xs text-muted-foreground">
                  {canvasCategories[choice.capability.category]}
                </span>
              )}
            </Button>
          ))}
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
