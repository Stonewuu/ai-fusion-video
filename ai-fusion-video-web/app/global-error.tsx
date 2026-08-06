"use client";

import { useEffect } from "react";
import { ErrorRecoveryPanel } from "@/components/error-recovery-panel";
import { Toaster } from "@/components/ui/sonner";
import {
  getClientErrorMessage,
  notifyClientError,
} from "@/lib/client-error";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    notifyClientError(error, "应用运行异常");
  }, [error]);

  return (
    <html lang="zh-CN">
      <body className="min-h-screen bg-background text-foreground antialiased">
        <Toaster richColors position="top-center" />
        <main className="flex min-h-screen">
          <ErrorRecoveryPanel
            title="应用遇到异常"
            description="已启用全局错误保护，不再显示不可操作的空白页。"
            details={getClientErrorMessage(error)}
            onRetry={reset}
            onGoHome={() => window.location.assign("/dashboard")}
          />
        </main>
      </body>
    </html>
  );
}
