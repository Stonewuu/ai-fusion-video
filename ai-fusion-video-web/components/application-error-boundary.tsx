"use client";

import { usePathname, useRouter } from "next/navigation";
import { ClientErrorBoundary } from "@/components/client-error-boundary";
import { ErrorRecoveryPanel } from "@/components/error-recovery-panel";
import { getClientErrorMessage } from "@/lib/client-error";

export function ApplicationErrorBoundary({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <ClientErrorBoundary
      key={pathname}
      context="应用页面运行异常"
      fallback={(error, reset) => (
        <ErrorRecoveryPanel
          className="min-h-screen"
          title="页面暂时无法显示"
          description="应用外壳仍在运行，你可以重试当前页面或返回仪表盘。"
          details={getClientErrorMessage(error)}
          onRetry={reset}
          onGoHome={() => router.push("/dashboard")}
        />
      )}
    >
      {children}
    </ClientErrorBoundary>
  );
}
