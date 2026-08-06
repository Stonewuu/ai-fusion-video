"use client";

import { useEffect } from "react";
import { notifyClientError } from "@/lib/client-error";

export function GlobalClientErrorNotifier() {
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      const message = notifyClientError(
        event.error ?? event.message,
        "页面运行异常"
      );
      if (message === null) {
        event.preventDefault();
      }
    };
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      const message = notifyClientError(event.reason, "异步任务发生异常");
      if (message === null) {
        event.preventDefault();
      }
    };

    window.addEventListener("error", handleError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);
    return () => {
      window.removeEventListener("error", handleError);
      window.removeEventListener("unhandledrejection", handleUnhandledRejection);
    };
  }, []);

  return null;
}
