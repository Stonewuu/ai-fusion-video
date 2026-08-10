"use client";

import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { dialogueTypeConfig } from "./constants";

export function InsertButton({
  onInsert,
  visible,
  isOpen,
  onToggle,
}: {
  onInsert: (type: number) => void;
  visible: boolean;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <div
      className={cn(
        "absolute left-1/2 -translate-x-1/2 -top-[9px] z-10 transition-opacity duration-150",
        visible || isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
      )}
    >
      <div className="relative">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
          className="size-[18px] rounded-full bg-primary/80 text-primary-foreground hover:bg-primary hover:scale-125 active:scale-95 transition-all shadow-sm cursor-pointer flex items-center justify-center"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <line x1="5" y1="1" x2="5" y2="9" />
            <line x1="1" y1="5" x2="9" y2="5" />
          </svg>
        </button>
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.1 }}
              className="absolute left-1/2 -translate-x-1/2 top-full mt-1 z-30 bg-popover border border-border/30 rounded-lg shadow-lg p-1 flex gap-0.5"
            >
              {Object.entries(dialogueTypeConfig).map(([typeKey, cfg]) => {
                const TypeIcon = cfg.icon;
                return (
                  <button
                    key={typeKey}
                    onClick={(e) => {
                      e.stopPropagation();
                      onInsert(Number(typeKey));
                      onToggle();
                    }}
                    className="flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs hover:bg-muted/50 transition-colors whitespace-nowrap"
                    title={cfg.label}
                  >
                    <TypeIcon className="h-3.5 w-3.5" />
                    {cfg.label}
                  </button>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
