"use client";

import { useEffect, useMemo, useState } from "react";

function formatApprovalCountdown(remainingMs: number): string {
  const totalSeconds = Math.max(0, Math.ceil(remainingMs / 1000));
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (days > 0) return `${days} 天 ${hours} 小时`;
  if (hours > 0) return `${hours} 小时 ${minutes} 分`;
  return `${minutes} 分 ${seconds} 秒`;
}

export function useToolConfirmationCountdown(
  expiresAt: string | undefined,
  updateEverySecond = true,
) {
  const expiresAtMs = useMemo(() => {
    if (expiresAt === undefined) return undefined;
    const parsed = Date.parse(expiresAt);
    if (!Number.isFinite(parsed)) {
      throw new Error("Tool confirmation expiry is invalid");
    }
    return parsed;
  }, [expiresAt]);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (expiresAtMs === undefined) return;
    const update = () => setNow(Date.now());
    update();
    const timer = updateEverySecond
      ? window.setInterval(update, 1000)
      : window.setTimeout(update, Math.max(0, expiresAtMs - Date.now()));
    window.addEventListener("focus", update);
    document.addEventListener("visibilitychange", update);
    return () => {
      if (updateEverySecond) {
        window.clearInterval(timer);
      } else {
        window.clearTimeout(timer);
      }
      window.removeEventListener("focus", update);
      document.removeEventListener("visibilitychange", update);
    };
  }, [expiresAtMs, updateEverySecond]);

  if (expiresAtMs === undefined) return undefined;
  const remainingMs = Math.max(0, expiresAtMs - now);
  return {
    expired: remainingMs === 0,
    label: formatApprovalCountdown(remainingMs),
  };
}
