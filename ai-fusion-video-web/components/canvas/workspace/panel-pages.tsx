"use client";

import type { ReactNode } from "react";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import type { CanvasPanel } from "./panel-navigation";

export function CanvasPanelPages({
  tabs,
  active,
  visited,
  content,
}: {
  tabs: readonly { key: CanvasPanel; title: string }[];
  active: CanvasPanel | null;
  visited: CanvasPanel[];
  content: Record<CanvasPanel, ReactNode>;
}) {
  return tabs.map(
    (tab) =>
      visited.includes(tab.key) && (
        <OverlayScrollArea
          key={tab.key}
          id={`canvas-panel-${tab.key}`}
          role="region"
          aria-label={tab.title}
          hidden={active !== tab.key}
          className={active === tab.key ? "min-h-0 flex-1" : "hidden"}
        >
          {content[tab.key]}
        </OverlayScrollArea>
      ),
  );
}
