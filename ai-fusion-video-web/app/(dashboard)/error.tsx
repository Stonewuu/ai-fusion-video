"use client";

import { useEffect } from "react";
import { ErrorRecoveryPanel } from "@/components/error-recovery-panel";
import {
  getClientErrorMessage,
  notifyClientError,
} from "@/lib/client-error";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    notifyClientError(error, "当前页面加载失败");
  }, [error]);

  return (
    <ErrorRecoveryPanel
      className="min-h-[60vh]"
      title="当前页面发生异常"
      description="顶部导航和侧边栏仍可使用；重试只会重新渲染当前页面。"
      details={getClientErrorMessage(error)}
      onRetry={reset}
      onGoHome={() => window.location.assign("/dashboard")}
    />
  );
}
