"use client";

import { useState } from "react";
import { Copy, FileAudio, Video } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/media/managed-image";
import { resolveMediaUrl } from "@/lib/api/client";
import type { GenerationExecution } from "@/lib/api/generation-execution";
import { GenerationMediaPreviewDialog, type GenerationMediaPreview } from "./generation-media-preview-dialog";
import { executionParameters, executionReferenceLabels } from "./generation-execution-presentation";

export function GenerationExecutionInputSection({ data }: { data: GenerationExecution }) {
  const [preview, setPreview] = useState<GenerationMediaPreview | null>(null);
  const prompt = data.input.prompt;
  const copyPrompt = async () => {
    if (!prompt) {
      return;
    }
    try {
      await navigator.clipboard.writeText(prompt);
      toast.success("提示词已复制");
    } catch {
      toast.error("复制失败，请选中提示词复制");
    }
  };

  return (
    <section className="flex min-w-0 flex-col gap-4 border-t border-border/20 pt-4">
      <h3 className="text-sm font-medium">生成内容</h3>
      <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        {executionParameters(data).map((parameter) => (
          <div key={parameter.label} className="min-w-0">
            <dt className="text-xs text-muted-foreground">{parameter.label}</dt>
            <dd className="mt-1 break-words">{parameter.value}</dd>
          </div>
        ))}
      </dl>
      <div className="min-w-0 rounded-lg border border-border/20 bg-background/70 p-3">
        <div className="flex items-center justify-between gap-2">
          <h4 className="text-sm font-medium">提示词</h4>
          <Button variant="ghost" size="icon-sm" disabled={!prompt} onClick={() => void copyPrompt()} aria-label="复制提示词">
            <Copy />
          </Button>
        </div>
        {prompt ? (
          <details className="peer">
            <summary className="cursor-pointer rounded-lg py-1 text-xs text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring">
              查看完整提示词（{[...prompt].length} 字）
            </summary>
            <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6">{prompt}</p>
          </details>
        ) : <p className="text-sm text-muted-foreground">未填写提示词</p>}
        {prompt && <p className="mt-2 line-clamp-2 whitespace-pre-wrap break-words text-sm leading-6 peer-open:hidden">{prompt}</p>}
      </div>
      {data.input.references.length > 0 && (
        <div className="flex min-w-0 flex-col gap-2">
          <h4 className="text-xs text-muted-foreground">参考素材（{data.input.references.length}）</h4>
          <div className="flex flex-wrap gap-3 p-1">
            {data.input.references.map((reference, index) => {
              const label = `${executionReferenceLabels[reference.kind]} ${index + 1}`;
              const image = ["firstFrame", "lastFrame", "image"].includes(reference.kind);
              if (reference.kind === "audio") {
                return <span key={`${reference.source}-${index}`} className="flex items-center gap-2 text-sm"><FileAudio className="size-4" />{label}</span>;
              }
              return (
                <button key={`${reference.source}-${index}`} type="button" aria-label={`预览${label}`}
                  onClick={() => setPreview({ type: image ? "image" : "video", url: reference.source, title: label })}
                  className="flex flex-col gap-1 rounded-lg outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">
                  <span className="grid size-16 place-items-center overflow-hidden rounded-lg border border-border/40 bg-muted/30">
                    {image ? <SafeImage src={resolveMediaUrl(reference.source)} alt={label} className="size-full object-cover" fallbackType="image" /> : <Video className="size-6 text-muted-foreground" />}
                  </span>
                  <span className="text-xs text-muted-foreground">{label}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
      <GenerationMediaPreviewDialog preview={preview} onOpenChange={(open) => {
        if (!open) {
          setPreview(null);
        }
      }} />
    </section>
  );
}
