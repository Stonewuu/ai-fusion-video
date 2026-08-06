"use client";

import { useEffect } from "react";
import { ErrorRecoveryPanel } from "@/components/error-recovery-panel";
import {
  getClientErrorMessage,
  notifyClientError,
} from "@/lib/client-error";

export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    notifyClientError(error, "页面加载失败");
  }, [error]);

  return (
    <ErrorRecoveryPanel
      className="min-h-screen"
      title="页面暂时无法显示"
      description="页面中的某个模块发生异常，应用没有退出。你可以重试或返回仪表盘。"
      details={getClientErrorMessage(error)}
      onRetry={reset}
      onGoHome={() => window.location.assign("/dashboard")}
    />
  );
}
