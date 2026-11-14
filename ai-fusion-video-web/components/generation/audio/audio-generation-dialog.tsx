"use client";

import { useEffect, useState } from "react";
import type { AiModel } from "@/lib/api/ai-model";
import { audioGenerationApi, audioGenerationActive, AUDIO_GENERATION_STATES, type AudioGenerationTask } from "@/lib/api/audio-generation";
import { Button } from "@/components/ui/button";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { cn } from "@/lib/utils";
import { AudioGenerationForm } from "./audio-generation-form";
import { AudioGenerationResult } from "./audio-generation-result";

export function AudioGenerationDialog({ model, onClose }: { model: AiModel; onClose: () => void }) {
  const [tasks, setTasks] = useState<AudioGenerationTask[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [error, setError] = useState("");
  const selected = tasks.find(task => task.id === selectedId) ?? null;

  useEffect(() => {
    let current = true;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const history = (await audioGenerationApi.list()).filter(task => task.modelId === model.id);
        if (current) {
          setTasks(history);
          setError("");
          if (history.some(audioGenerationActive)) {
            timer = setTimeout(() => void refresh(), 2000);
          }
        }
      } catch {
        if (current) {
          setError("生成记录暂时无法加载，请重试");
        }
      }
    };
    void refresh();
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [model.id, reload]);

  return <Dialog open onOpenChange={open => {
    if (!open) {
      onClose();
    }
  }}>
    <DialogContent className="sm:max-w-4xl max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
      <DialogHeader className="shrink-0">
        <DialogTitle>{model.name}</DialogTitle>
        <DialogDescription>{model.code.startsWith("lyria-") ? "输入音乐描述，生成后可试听。" : "输入朗读内容，生成后可试听。"}</DialogDescription>
      </DialogHeader>
      <DialogScrollBody>
        <div className="flex min-w-0 flex-col gap-6">
          <div className="grid min-w-0 gap-6 lg:grid-cols-2">
            <AudioGenerationForm model={model} onCreated={task => {
              setTasks(current => [task, ...current.filter(item => item.id !== task.id)]);
              setSelectedId(task.id);
            }} onRefresh={() => setReload(value => value + 1)} />
            <section aria-label="生成结果" className="min-w-0 border-t border-border/20 pt-4 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-6">
              <AudioGenerationResult task={selected} onRetry={() => setReload(value => value + 1)} />
            </section>
          </div>
          {error && <div className="flex items-center gap-2" role="alert"><p className="text-sm text-destructive">{error}</p>
            <Button variant="outline" onClick={() => setReload(value => value + 1)}>重试</Button></div>}
          {tasks.length > 0 && <section className="flex flex-col gap-3 border-t border-border/20 pt-4" aria-label="最近生成">
            <h3 className="text-sm font-medium">最近生成</h3>
            <div className="flex flex-col gap-1 p-1">
              {tasks.slice(0, 5).map(task => <button key={task.id} type="button" aria-pressed={selectedId === task.id}
                onClick={() => setSelectedId(task.id)} className={cn("flex items-center justify-between gap-3 rounded-xl border border-transparent p-3 text-left text-sm outline-none transition-colors motion-reduce:transition-none hover:bg-muted/40 focus-visible:ring-2 focus-visible:ring-ring",
                  selectedId === task.id && "border-primary/30 bg-primary/5")}>
                <span>{new Date(task.createTime + "Z").toLocaleString("zh-CN")}</span>
                <span className="text-xs text-muted-foreground">{AUDIO_GENERATION_STATES[task.state]}</span>
              </button>)}
            </div>
          </section>}
        </div>
      </DialogScrollBody>
      <DialogFooter className="shrink-0"><Button variant="outline" onClick={onClose}>关闭</Button></DialogFooter>
    </DialogContent>
  </Dialog>;
}
