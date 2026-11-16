"use client";
import {
  createContext,
  useCallback,
  useContext,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  defaultRangeExtractor,
  useVirtualizer,
  type ReactVirtualizer,
} from "@tanstack/react-virtual";
import { retainVirtualItems } from "@/lib/virtual-list";
export const StoryboardScrollContext = createContext<{
  scrollRef: RefObject<HTMLDivElement | null>;
  navigatingRef: RefObject<boolean>;
} | null>(null);
export function useStoryboardVirtualRows({
  count,
  getItemKey,
  estimateSize,
  pinnedIndexes,
  lanes = 1,
  gap = 0,
}: {
  count: number;
  getItemKey: (index: number) => string | number;
  estimateSize: (index: number) => number;
  pinnedIndexes: number[];
  lanes?: number | ((width: number) => number);
  gap?: number;
}) {
  "use no memo";
  const { scrollRef, navigatingRef } = useContext(StoryboardScrollContext)!;
  const contentRef = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState({ top: 0, width: 0 });
  const hasRows = count > 0;
  useLayoutEffect(() => {
    const content = contentRef.current;
    const viewport = scrollRef.current;
    if (!content || !viewport) {
      return;
    }
    let frame = 0;
    const measure = () => {
      const top =
        viewport.scrollTop +
        content.getBoundingClientRect().top -
        viewport.getBoundingClientRect().top;
      const width = content.clientWidth;
      setLayout((previous) =>
        previous.top === top && previous.width === width
          ? previous
          : { top, width },
      );
    };
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    });
    observer.observe(content);
    observer.observe(
      content.closest<HTMLElement>("[data-storyboard-list-content]")!,
    );
    measure();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [scrollRef, hasRows]);
  const columnCount = typeof lanes === "function" ? lanes(layout.width) : lanes;
  const virtualizer: ReactVirtualizer<HTMLDivElement, HTMLDivElement> =
    // eslint-disable-next-line react-hooks/incompatible-library -- 此 Hook 已排除自动编译，虚拟列表负责位置更新。
    useVirtualizer({
      count,
      getScrollElement: () => scrollRef.current,
      getItemKey,
      estimateSize,
      scrollMargin: layout.top,
      overscan: 3,
      lanes: columnCount,
      gap,
      directDomUpdates: true,
      useAnimationFrameWithResizeObserver: true,
      rangeExtractor: (range): number[] => {
        const top = virtualizer.scrollOffset ?? 0;
        const bottom = top + (virtualizer.scrollRect?.height ?? 0);
        const outside =
          bottom < layout.top || top > layout.top + virtualizer.getTotalSize();
        return retainVirtualItems(
          outside ? [] : defaultRangeExtractor(range),
          pinnedIndexes,
          count,
        );
      },
    });
  virtualizer.shouldAdjustScrollPositionOnItemSizeChange = (
    item,
    _delta,
    instance,
  ) => !navigatingRef.current && item.start < (instance.scrollOffset ?? 0);
  const setContentRef = useCallback(
    (node: HTMLDivElement | null) => {
      contentRef.current = node;
      virtualizer.containerRef(node);
    },
    [virtualizer],
  );
  return {
    virtualizer,
    setContentRef,
    width: layout.width,
    lanes: columnCount,
  };
}
