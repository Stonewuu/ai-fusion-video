"use client";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import {
  useVirtualizer,
  defaultRangeExtractor,
  type Range,
} from "@tanstack/react-virtual";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { retainVirtualItems } from "@/lib/virtual-list";

export function VirtualDirectory<T extends { id: number }>({
  items,
  activeId,
  expandedIds,
  draggingId = null,
  estimateSize,
  onToggle,
  children,
  label,
}: {
  items: T[];
  activeId: number | null;
  expandedIds: Set<number>;
  draggingId?: number | null;
  estimateSize: (index: number) => number;
  onToggle: (id: number) => void;
  children: (item: T, index: number) => ReactNode;
  label: string;
}) {
  "use no memo";
  const viewportRef = useRef<HTMLDivElement>(null);
  const pendingFocus = useRef<number | null>(null);
  const lastNavigatedId = useRef<number | null>(null);
  const [focusedId, setFocusedId] = useState<number | null>(null);
  const indexes = useMemo(
    () => new Map(items.map((episode, index) => [episode.id, index])),
    [items],
  );
  const getItemKey = useCallback((index: number) => items[index].id, [items]);
  const rangeExtractor = useCallback(
    (range: Range) => {
      const pinned: number[] = [];
      // 焦点和拖动所属分集离开视口后继续挂载，操作完成后再回收。
      for (const id of [focusedId, draggingId]) {
        const index = id === null ? undefined : indexes.get(id);
        if (index !== undefined) {
          pinned.push(index);
        }
      }
      return retainVirtualItems(
        defaultRangeExtractor(range),
        pinned,
        items.length,
      );
    },
    [focusedId, draggingId, indexes, items.length],
  );
  // eslint-disable-next-line react-hooks/incompatible-library -- 容器已用 use no memo 排除自动编译。
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => viewportRef.current,
    getItemKey,
    estimateSize,
    rangeExtractor,
    overscan: 5,
    paddingStart: 8,
    paddingEnd: 8,
    scrollPaddingStart: 8,
    scrollPaddingEnd: 8,
    directDomUpdates: true,
    useAnimationFrameWithResizeObserver: true,
  });

  const viewportHeight = virtualizer.scrollRect?.height ?? 0;
  const viewportWidth = virtualizer.scrollRect?.width ?? 0;
  useEffect(() => {
    if (
      !viewportHeight ||
      !viewportWidth ||
      lastNavigatedId.current === activeId
    ) {
      return;
    }
    lastNavigatedId.current = activeId;
    const index = activeId === null ? undefined : indexes.get(activeId);
    if (index !== undefined) {
      virtualizer.scrollToIndex(index, { align: "auto" });
    }
  }, [activeId, indexes, virtualizer, viewportHeight, viewportWidth]);

  useLayoutEffect(() => {
    if (pendingFocus.current !== null) {
      const button = viewportRef.current?.querySelector<HTMLButtonElement>(
        '[data-directory-select="' + pendingFocus.current + '"]',
      );
      if (button) {
        button.focus({ preventScroll: true });
        pendingFocus.current = null;
      }
    }
  }, [focusedId]);

  const navigate = (event: KeyboardEvent<HTMLDivElement>) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>(
      "button[data-directory-select]",
    );
    if (!button) {
      return;
    }
    const id = Number(button.dataset.directorySelect);
    const current = indexes.get(id)!;
    let next: number;
    switch (event.key) {
      case "ArrowDown":
        next = Math.min(items.length - 1, current + 1);
        break;
      case "ArrowUp":
        next = Math.max(0, current - 1);
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = items.length - 1;
        break;
      case "ArrowRight":
      case "ArrowLeft":
        event.preventDefault();
        if (expandedIds.has(id) !== (event.key === "ArrowRight")) {
          onToggle(id);
        }
        return;
      default:
        return;
    }
    event.preventDefault();
    pendingFocus.current = items[next].id;
    setFocusedId(items[next].id);
    virtualizer.scrollToIndex(next, { align: "auto" });
  };

  return (
    <OverlayScrollArea
      className="min-h-0 flex-1"
      viewportRef={viewportRef}
      viewportClassName="scroll-p-2 [overflow-anchor:none]"
    >
      <div
        ref={virtualizer.containerRef}
        role="list"
        aria-label={label}
        className="relative w-full"
        onKeyDown={navigate}
      >
        {virtualizer.getVirtualItems().map((row) => {
          const item = items[row.index];
          return (
            <div
              key={row.key}
              data-index={row.index}
              data-directory-item={item.id}
              ref={virtualizer.measureElement}
              role="listitem"
              aria-posinset={row.index + 1}
              aria-setsize={items.length}
              className="absolute left-0 top-0 w-full px-2"
              onFocusCapture={() => setFocusedId(item.id)}
              onBlurCapture={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget)) {
                  setFocusedId(null);
                }
              }}
            >
              {children(item, row.index)}
            </div>
          );
        })}
      </div>
    </OverlayScrollArea>
  );
}
