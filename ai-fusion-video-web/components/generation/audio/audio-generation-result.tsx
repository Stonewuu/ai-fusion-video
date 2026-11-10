"use client";

import { useEffect, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { audioGenerationApi, audioGenerationActive, AUDIO_GENERATION_STATES, type AudioGenerationTask } from "@/lib/api/audio-generation";
import { downloadMedia, mediaAccessUrl } from "@/lib/api/media";
import { Button } from "@/components/ui/button";

function AudioResult({ reference }: { reference: string }) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let current = true;
    mediaAccessUrl(reference).then(value => {
      if (current) {
        setUrl(value);
        setError("");
      }
    }).catch(failure => {
      if (current) {
        setError(failure instanceof Error ? failure.message : "音频暂时无法加载，请重试");
      }
    });
    return () => {
      current = false;
    };
  }, [reference, retry]);
  return <div className="flex flex-col gap-2">
    {url && <audio controls preload="metadata" src={url} className="w-full" />}
    {error ? <div className="flex items-center gap-2"><p className="text-sm text-destructive">{error}</p>
      <Button variant="outline" onClick={() => setRetry(value => value + 1)}>重试</Button></div>
      : <Button variant="outline" onClick={() => void downloadMedia(reference).catch(() => toast.error("下载失败，请重试"))}><Download />下载音频</Button>}
  </div>;
}

export function AudioGenerationResult({ task, onRetry }: { task: AudioGenerationTask | null; onRetry: () => void }) {
  const [saving, setSaving] = useState(false);
  if (!task) {
    return <p className="py-12 text-center text-sm text-muted-foreground">生成后可在这里试听</p>;
  }
  return <div className="flex min-w-0 flex-col gap-4 p-1">
    <div className="flex items-center gap-2">
      {audioGenerationActive(task) && <Loader2 className="size-4 animate-spin text-primary" />}
      <h3 className="font-medium">{AUDIO_GENERATION_STATES[task.state]}</h3>
    </div>
    {task.errorMessage && <p role="alert" className="text-sm text-destructive">{task.errorMessage}</p>}
    {task.canRetrySave && <Button variant="outline" disabled={saving} onClick={() => {
      setSaving(true);
      audioGenerationApi.retrySave(task.id).then(onRetry).catch(() => toast.error("暂时无法重试，请稍后再试"))
        .finally(() => setSaving(false));
    }}>重试保存</Button>}
    {task.result?.text && <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">{task.result.text}</div>}
    {task.result?.audio.map(reference => <AudioResult key={reference} reference={reference} />)}

  </div>;
}
