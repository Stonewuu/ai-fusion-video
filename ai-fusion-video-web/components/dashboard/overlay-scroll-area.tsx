"use client";

import type {
  ComponentPropsWithoutRef,
  CSSProperties,
  KeyboardEventHandler,
  PointerEventHandler,
  ReactNode,
  Ref,
  TouchEventHandler,
  UIEventHandler,
  WheelEventHandler,
} from "react";
import { ScrollArea } from "@base-ui/react/scroll-area";
import { cn } from "@/lib/utils";

interface OverlayScrollAreaProps
  extends Omit<ComponentPropsWithoutRef<typeof ScrollArea.Root>, "children"> {
  children: ReactNode;
  viewportClassName?: string;
  viewportStyle?: CSSProperties;
  viewportRef?: Ref<HTMLDivElement>;
  onViewportScroll?: UIEventHandler<HTMLDivElement>;
  onViewportWheel?: WheelEventHandler<HTMLDivElement>;
  onViewportKeyDown?: KeyboardEventHandler<HTMLDivElement>;
  onViewportTouchStart?: TouchEventHandler<HTMLDivElement>;
  onViewportTouchMove?: TouchEventHandler<HTMLDivElement>;
  onScrollbarPointerDown?: PointerEventHandler<HTMLDivElement>;
  scrollbarClassName?: string;
}

export function OverlayScrollArea({
  children,
  className,
  viewportClassName,
  viewportStyle,
  viewportRef,
  onViewportScroll,
  onViewportWheel,
  onViewportKeyDown,
  onViewportTouchStart,
  onViewportTouchMove,
  onScrollbarPointerDown,
  scrollbarClassName,
  ...rootProps
}: OverlayScrollAreaProps) {
  return (
    <ScrollArea.Root
      data-slot="overlay-scroll-area"
      className={cn("relative min-h-0 min-w-0 overflow-hidden", className)}
      {...rootProps}
    >
      <ScrollArea.Viewport
        data-slot="overlay-scroll-viewport"
        ref={viewportRef}
        suppressHydrationWarning
        className={cn("h-full w-full overscroll-contain", viewportClassName)}
        style={viewportStyle}
        onScroll={onViewportScroll}
        onWheel={onViewportWheel}
        onKeyDown={onViewportKeyDown}
        onTouchStart={onViewportTouchStart}
        onTouchMove={onViewportTouchMove}
      >
        {children}
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar
        data-slot="overlay-scrollbar"
        orientation="vertical"
        onPointerDown={onScrollbarPointerDown}
        className={cn(
          "z-40 w-2 bg-transparent p-0.5 opacity-60 transition-opacity duration-150 hover:opacity-100 data-scrolling:opacity-100 motion-reduce:transition-none",
          scrollbarClassName,
        )}
      >
        <ScrollArea.Thumb
          data-slot="overlay-scroll-thumb"
          className="w-full rounded-full bg-foreground/20 transition-colors duration-150 hover:bg-foreground/35 motion-reduce:transition-none"
        />
      </ScrollArea.Scrollbar>
    </ScrollArea.Root>
  );
}
