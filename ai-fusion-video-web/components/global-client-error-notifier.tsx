"use client";

import { useEffect } from "react";
import { notifyClientError } from "@/lib/client-error";

export function GlobalClientErrorNotifier() {
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      // 浏览器把未完成的尺寸通知延后到下一帧；保留控制台诊断，不将其展示为操作失败。
      if (!event.error && event.message === "ResizeObserver loop completed with undelivered notifications.") {
        return;
      }
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
