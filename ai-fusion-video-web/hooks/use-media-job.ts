"use client";

import { useEffect, useState } from "react";
import { mediaJobsApi, mediaJobRunning, type MediaJob } from "@/lib/api/media-jobs";

export function useMediaJob(id: string | null | undefined, refresh = 0) {
  const [state, setState] = useState<{ id: string; job?: MediaJob; error?: string }>();
  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const job = await mediaJobsApi.get(id!, controller.signal);
        if (controller.signal.aborted) return;
        setState({ id: id!, job });
        if (mediaJobRunning(job)) timer = setTimeout(refresh, 2000);
      } catch (error) {
        if (controller.signal.aborted) return;
        setState({ id: id!, error: error instanceof Error ? error.message : "任务读取失败" });
        timer = setTimeout(refresh, 5000);
      }
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [id, refresh]);
  return state?.id === id ? state : undefined;
}
