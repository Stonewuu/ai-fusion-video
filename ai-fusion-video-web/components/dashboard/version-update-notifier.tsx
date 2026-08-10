"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { getSystemVersion } from "@/lib/api/system";
import {
  getIgnoredVersion,
  setIgnoredVersion,
  markVersionToastShown,
} from "@/lib/version-update";

const TOAST_ID = "system-version-update";

export function VersionUpdateNotifier() {
  const router = useRouter();

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const versionInfo = await getSystemVersion();
        if (cancelled || !versionInfo.updateAvailable || !versionInfo.latestVersion) {
          return;
        }
        if (getIgnoredVersion() === versionInfo.latestVersion) {
          return;
        }
        if (!markVersionToastShown(versionInfo.latestVersion)) {
          return;
        }

        const description = versionInfo.message
          || `当前版本 ${versionInfo.currentVersionDisplay}，可升级到 ${versionInfo.latestVersionDisplay}。`;

        toast.info(`发现可升级版本 ${versionInfo.latestVersionDisplay}`, {
          id: TOAST_ID,
          description,
          duration: 12000,
          action: {
            label: "查看",
            onClick: () => router.push("/settings/general"),
          },
          cancel: {
            label: "忽略",
            onClick: () => setIgnoredVersion(versionInfo.latestVersion!),
          },
        });
      } catch {
        // 版本检查失败不阻塞页面交互
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [router]);

  return null;
}