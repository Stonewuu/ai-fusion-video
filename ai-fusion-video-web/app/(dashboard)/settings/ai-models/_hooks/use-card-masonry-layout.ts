"use client";

import { useLayoutEffect, type RefObject } from "react";
import type { AiModelViewMode } from "../_components/ai-models-toolbar";

export function useCardMasonryLayout(
  gridRef: RefObject<HTMLDivElement | null>,
  options: {
    viewMode: AiModelViewMode;
    disabled: boolean;
    layoutSignature: string;
    shouldReduceMotion: boolean;
  },
) {
  const { viewMode, disabled, layoutSignature, shouldReduceMotion } = options;

  useLayoutEffect(() => {
    const grid = gridRef.current;
    if (!grid) {
      return;
    }

    const cards = Array.from(
      grid.querySelectorAll<HTMLElement>("[data-api-config-card]"),
    );
    const desktopColumns = window.matchMedia("(min-width: 1024px)");
    let layoutFrame: number | undefined;

    const resetLayout = () => {
      if (layoutFrame !== undefined) {
        window.cancelAnimationFrame(layoutFrame);
      }
      grid.style.gridAutoRows = "";
      cards.forEach((card) => {
        card.getAnimations().forEach((animation) => animation.cancel());
        card.style.gridRowEnd = "";
      });
    };

    const layoutCard = (card: HTMLElement) => {
      const gridStyle = window.getComputedStyle(grid);
      const rowHeight = Number.parseFloat(gridStyle.gridAutoRows);
      const rowGap = Number.parseFloat(gridStyle.rowGap);
      if (!Number.isFinite(rowHeight) || !Number.isFinite(rowGap)) {
        return;
      }
      const cardHeight = card.getBoundingClientRect().height;
      const rowSpan = Math.ceil((cardHeight + rowGap) / (rowHeight + rowGap));
      card.style.gridRowEnd = `span ${rowSpan}`;
    };

    const layoutCards = (animate: boolean) => {
      if (
        viewMode !== "card"
        || disabled
        || cards.length === 0
        || !desktopColumns.matches
      ) {
        resetLayout();
        return;
      }

      const previousRects =
        animate && !shouldReduceMotion
          ? new Map(cards.map((card) => [card, card.getBoundingClientRect()]))
          : null;

      cards.forEach((card) => {
        card.getAnimations().forEach((animation) => animation.cancel());
      });
      grid.style.gridAutoRows = "1px";
      cards.forEach(layoutCard);

      if (!previousRects) {
        return;
      }
      cards.forEach((card) => {
        const previousRect = previousRects.get(card);
        if (!previousRect) {
          return;
        }
        const nextRect = card.getBoundingClientRect();
        const deltaX = previousRect.left - nextRect.left;
        const deltaY = previousRect.top - nextRect.top;
        if (Math.abs(deltaX) < 1 && Math.abs(deltaY) < 1) {
          return;
        }
        card.animate(
          [
            { transform: `translate(${deltaX}px, ${deltaY}px)` },
            { transform: "translate(0, 0)" },
          ],
          {
            duration: 220,
            easing: "cubic-bezier(0.25, 0.46, 0.45, 0.94)",
          },
        );
      });
    };

    const scheduleAnimatedLayout = () => {
      if (layoutFrame !== undefined) {
        window.cancelAnimationFrame(layoutFrame);
      }
      layoutFrame = window.requestAnimationFrame(() => {
        layoutFrame = undefined;
        layoutCards(true);
      });
    };

    const resizeObserver = new ResizeObserver(() => {
      if (viewMode !== "card" || !desktopColumns.matches) {
        return;
      }
      scheduleAnimatedLayout();
    });

    cards.forEach((card) => resizeObserver.observe(card));
    const handleColumnsChange = () => {
      layoutCards(false);
    };
    desktopColumns.addEventListener("change", handleColumnsChange);
    layoutCards(false);

    return () => {
      resizeObserver.disconnect();
      desktopColumns.removeEventListener("change", handleColumnsChange);
      resetLayout();
    };
  }, [disabled, gridRef, layoutSignature, shouldReduceMotion, viewMode]);
}
