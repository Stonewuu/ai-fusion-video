"use client";

import { useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { CanvasPanelPages } from "./panel-pages";
import type { CanvasPanel } from "./panel-navigation";

export function AnimatedCanvasPanelBody({ active, side, tabs, visited, content }: {
  active: CanvasPanel | null;
  side: "left" | "right";
  tabs: readonly { key: CanvasPanel; title: string }[];
  visited: CanvasPanel[];
  content: Record<CanvasPanel, ReactNode>;
}) {
  const [displayed, setDisplayed] = useState(active);
  if (active !== null && active !== displayed) {
    setDisplayed(active);
  }

  return (
    <div
      inert={active === null}
      aria-hidden={active === null}
      className={cn(
        "flex min-h-0 flex-1 flex-col transition-[opacity,translate] motion-reduce:transition-none",
        side === "left" ? "w-full md:w-64" : "w-full xl:w-88",
        active ? "translate-x-0 opacity-100 duration-200 ease-out" : "pointer-events-none opacity-0 duration-150 ease-in",
        !active && (side === "left" ? "-translate-x-2" : "translate-x-2"),
      )}
    >
      <CanvasPanelPages tabs={tabs} active={displayed} visited={visited} content={content} />
    </div>
  );
}
