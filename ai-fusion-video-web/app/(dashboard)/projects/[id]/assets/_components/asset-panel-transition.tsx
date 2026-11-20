"use client";

import type { ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

export function AssetPanelTransition({
  panelKey,
  className,
  children,
}: {
  panelKey: string | number;
  className?: string;
  children: ReactNode;
}) {
  const reduceMotion = useReducedMotion();
  return (
    <AnimatePresence initial={false} mode="wait">
      <motion.div
        key={panelKey}
        data-asset-panel={panelKey}
        className={className}
        initial={reduceMotion ? false : { opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0, pointerEvents: "auto" }}
        exit={{
          opacity: 0,
          y: reduceMotion ? 0 : -4,
          pointerEvents: "none",
          transition: { duration: reduceMotion ? 0 : 0.12, ease: "easeIn" },
        }}
        transition={{
          duration: reduceMotion ? 0 : 0.2,
          ease: [0.25, 0.46, 0.45, 0.94],
        }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
