"use client";

import type { ReactNode, Ref } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { OverlayScrollArea } from "../overlay-scroll-area";
import { MODE_TRANSITION_EASE } from "./generation-motion";

interface GenerationWorkspaceLayoutProps {
  advancedMode: boolean;
  reduceMotion: boolean | null;
  composerRef: Ref<HTMLDivElement>;
  historyBottomInset: number;
  composer: ReactNode;
  children: ReactNode;
}

export function GenerationWorkspaceLayout({
  advancedMode,
  reduceMotion,
  composerRef,
  historyBottomInset,
  composer,
  children,
}: GenerationWorkspaceLayoutProps) {
  return (
    <OverlayScrollArea
      data-slot="generation-workspace"
      className={cn("min-h-0 flex-1", advancedMode && "px-5 lg:px-8")}
    >
      <div
        className={cn(
          "relative h-full min-h-0",
          advancedMode &&
            "flex w-full flex-col gap-4 xl:grid xl:grid-cols-[minmax(400px,460px)_minmax(0,1fr)]",
        )}
      >
        <AnimatePresence initial={false} mode="popLayout">
          {advancedMode && (
            <motion.div
              key="advanced-composer"
              data-slot="generation-composer-layer"
              initial={reduceMotion ? false : { opacity: 0, x: -14 }}
              animate={{
                opacity: 1,
                x: 0,
                transition: reduceMotion
                  ? { duration: 0 }
                  : { duration: 0.24, ease: MODE_TRANSITION_EASE },
              }}
              exit={{
                opacity: 0,
                x: reduceMotion ? 0 : -10,
                transition: reduceMotion
                  ? { duration: 0 }
                  : { duration: 0.14, ease: "easeIn" },
              }}
              className="h-full min-h-0 shrink-0"
            >
              <div
                data-slot="generation-composer"
                ref={composerRef}
                className="h-full min-h-0"
              >
                {composer}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.div
          layout
          transition={{
            layout: reduceMotion
              ? { duration: 0 }
              : { duration: 0.26, ease: MODE_TRANSITION_EASE },
          }}
          className={cn(
            "h-full min-h-0 w-full",
            advancedMode && "max-h-[42vh] shrink-0 xl:max-h-none",
          )}
        >
          <OverlayScrollArea
            data-slot="generation-history-scroll"
            className="h-full min-h-0 w-full"
            viewportStyle={{
              paddingBottom: historyBottomInset,
              scrollPaddingBottom: historyBottomInset,
            }}
          >
            <div
              className={cn("w-full", advancedMode ? "pr-3" : "px-5 lg:px-8")}
            >
              <div
                className={cn(
                  "mx-auto w-full",
                  !advancedMode && "max-w-[1180px]",
                )}
              >
                {children}
              </div>
            </div>
          </OverlayScrollArea>
        </motion.div>
      </div>
    </OverlayScrollArea>
  );
}
