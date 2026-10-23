"use client";

import { useState } from "react";
import { Clapperboard, Loader2, PlayCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import type { StoryboardEpisode } from "@/lib/api/storyboard";
import { mediaJobsApi, mediaJobRunning } from "@/lib/api/media-jobs";
import { useMediaJob } from "@/hooks/use-media-job";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { EpisodeAudioReviewForm } from "./episode-audio-review";

interface StoryboardComposeActionsProps {
  episode: StoryboardEpisode | null;
  submitting: boolean;
  running: boolean;
  onCompose: (includeBgm: boolean) => Promise<void>;
  onPreview: (url: string) => void;
}

export function StoryboardComposeActions({ episode, submitting, running, onCompose, onPreview }: StoryboardComposeActionsProps) {
  const [details, setDetails] = useState(false);
  const [includeBgm, setIncludeBgm] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [cancelState, setCancelState] = useState<{ id: string; pending: boolean; message?: string }>();
  const state = useMediaJob(episode?.composeJobId, refresh);
  if (!episode) return null;
  const job = state?.job;
  const busy = submitting || (job ? mediaJobRunning(job) : running || episode.composeStatus === 1);
  const review = job?.status === "NEEDS_REVIEW";
  const partial = job?.results.video || job?.results.baseVideo;
  const cancel = async () => {
    if (!job) return;
    setCancelState({ id: job.id, pending: true });
    try {
      await mediaJobsApi.cancel(job.id);
      setRefresh(value => value + 1);
      setCancelState({ id: job.id, pending: false, message: "已请求取消，等待当前处理停止" });
    } catch (error) {
      setCancelState({ id: job.id, pending: false, message: error instanceof Error ? error.message : "取消失败" });
    }
  };
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-2">
      {episode.composedVideoUrl && (
        <Button variant="outline" size="sm" onClick={() => onPreview(episode.composedVideoUrl!)} title="预览最近一次写回的成片">
          <PlayCircle />预览
        </Button>
      )}
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <Checkbox checked={includeBgm} onCheckedChange={(checked) => setIncludeBgm(checked === true)} disabled={busy} />
        配乐
      </label>
      <Button variant="outline" size="sm" disabled={busy || review} onClick={() => onCompose(includeBgm)}
        title={includeBgm ? "按镜头顺序合成；启用 Sonilo 配置时生成配乐" : "按镜头顺序合成并保留原声"}>
        {busy ? <Loader2 className="animate-spin" /> : <Clapperboard />}
        {submitting ? "提交中" : review ? "待核实" : job?.cancelRequested && busy ? "取消中" : busy ? "合成中" : episode.composeStatus === 3 ? "重试" : episode.composedVideoUrl ? "重新合成" : "合成"}
      </Button>
      {episode.composeJobId && <Button variant="ghost" size="sm" onClick={() => setDetails(true)}>任务</Button>}
      <Dialog open={details} onOpenChange={setDetails}>
        <DialogContent className="max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
          <DialogHeader className="shrink-0">
            <DialogTitle>合成任务</DialogTitle>
            <DialogDescription>输入按提交时的镜头版本固定；分镜有新修改时，成片保留在任务中供检查。</DialogDescription>
          </DialogHeader>
          <DialogScrollBody className="flex flex-col gap-4">
            <p aria-live="polite">{state?.error || job?.phase || "正在读取任务"}</p>
            {job?.error && <p className="text-sm text-destructive">{job.error}</p>}
            <div className="flex gap-2">
              {partial && <Button variant="outline" onClick={() => onPreview(partial)}>查看结果</Button>}
              {job && mediaJobRunning(job) && <Button variant="destructive-ghost" disabled={job.cancelRequested || (cancelState?.id === job.id && cancelState.pending)} onClick={cancel}>{job.cancelRequested ? "正在取消" : "取消处理"}</Button>}
            </div>
            {job && mediaJobRunning(job) && cancelState?.id === job.id && cancelState.message && <p className="text-sm text-muted-foreground">{cancelState.message}</p>}
            {job && mediaJobRunning(job) && <p className="text-xs text-muted-foreground">关闭页面后继续处理。取消后等待供应商已受理的请求结束，再释放并发名额。</p>}
            {job && review && <EpisodeAudioReviewForm key={job.id} job={job} onChanged={() => setRefresh(value => value + 1)} />}
          </DialogScrollBody>
        </DialogContent>
      </Dialog>
    </div>
  );
}
