"use client";

import type { MediaJob } from "@/lib/api/media-jobs";
import { ExecutionRecordButton } from "@/components/dashboard/notification-panel/execution-record-button";

export function EpisodeAudioReviewForm({ job }: { job: MediaJob; onChanged: () => void }) {
  return <section className="flex flex-col gap-3 border-t border-border/20 pt-4">
    <h3 className="text-sm font-medium">{job.status === "UNKNOWN_SUBMISSION" ? "远端结果未知" : "音频任务记录"}</h3>
    <p className="text-sm text-muted-foreground">原请求不会自动重发，处理记录可在运行与通知中查看。</p>
    <div><ExecutionRecordButton target={{ kind: "EPISODE_AUDIO", id: job.id }} /></div>
  </section>;
}
