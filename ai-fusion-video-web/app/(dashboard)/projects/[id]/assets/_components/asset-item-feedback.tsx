"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";

export function AssetItemMessage({
  message,
  id,
}: {
  message: string;
  id?: string;
}) {
  const reduceMotion = useReducedMotion();
  return (
    <AnimatePresence initial={false}>
      {message && (
        <motion.div
          key="message"
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{
            height: 0,
            opacity: 0,
            transition: { duration: reduceMotion ? 0 : 0.12 },
          }}
          transition={{ duration: reduceMotion ? 0 : 0.2, ease: "easeOut" }}
          className="overflow-hidden"
        >
          <p
            id={id}
            role="alert"
            className="py-1 text-xs leading-5 text-destructive"
          >
            {message}
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function AssetItemSaveStatus({
  creating,
  dirty,
  saving,
}: {
  creating: boolean;
  dirty: boolean;
  saving: boolean;
}) {
  const reduceMotion = useReducedMotion();
  const state = saving
    ? "saving"
    : dirty
      ? "dirty"
      : creating
        ? "draft"
        : "saved";
  const text = saving
    ? "正在保存"
    : dirty
      ? "有未保存修改"
      : creating
        ? "尚未创建"
        : "已保存";
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex min-h-5 min-w-0 items-center text-xs text-muted-foreground"
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={state}
          className="inline-flex items-center gap-1.5"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reduceMotion ? 0 : 0.12 }}
        >
          {saving ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : state === "saved" ? (
            <Check className="size-3.5" />
          ) : null}
          {text}
        </motion.span>
      </AnimatePresence>
    </div>
  );
}
