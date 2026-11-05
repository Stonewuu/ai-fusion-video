"use client";

import { CircleAlert, Clock3, CheckCircle2 } from "lucide-react";
import { SafeImage } from "@/components/media/managed-image";
import { ManagedVideo } from "@/components/media/managed-video";
import { generationPhaseLabels, generationStateLabels } from "@/lib/generation-execution-view";
import type { GenerationExecution } from "@/lib/api/generation-execution";
import { userFacingError } from "@/lib/user-facing-error";
import { generationTaskSummary } from "./generation-task-summary";

type GenerationOutput = { id: number; status: number; imageUrl?: string; videoUrl?: string; coverUrl?: string; errorMsg?: string };

export function GenerationTaskResult({ data }: { data: unknown }) {
  const result = data as { state: string; status?: string; execution?: GenerationExecution; outputs?: GenerationOutput[] };
  const summary = generationTaskSummary(result.state, result.status === "pending");
  const Icon = summary.tone === "success" ? CheckCircle2 : summary.tone === "warning" ? CircleAlert : Clock3;
  const outputs = result.outputs ?? [];

  return (
    <div className="flex flex-col gap-3 text-xs">
      <p className="flex items-center gap-2 font-medium text-foreground"><Icon className="size-4 shrink-0" />{summary.label}</p>
      {summary.description && <p className="text-muted-foreground">{summary.description}</p>}
      {result.execution && (
        <div className="flex flex-col gap-1 text-muted-foreground">
          <p>{result.execution.modelName}</p>
          {result.execution.items.map((item) => (
            <p key={item.id}>
              第 {item.index + 1} 项 · {item.state === "NEEDS_REVIEW" ? "需要核实" : generationStateLabels[item.state] ?? item.state}
              {item.phase !== "FINISHED" && ` · ${generationPhaseLabels[item.phase] ?? item.phase}`}
              {item.errorMessage && ` · ${userFacingError(item.errorMessage)}`}
            </p>
          ))}
        </div>
      )}
      {outputs.filter((output) => output.status === 1).map((output) => (
        <div key={output.id} className="flex flex-col gap-2">
          {output.imageUrl && <SafeImage src={output.imageUrl} alt="生成图片" className="max-h-64 w-full rounded-lg object-contain" />}
          {output.videoUrl && <ManagedVideo src={output.videoUrl} poster={output.coverUrl} controls className="max-h-64 w-full rounded-lg" />}
        </div>
      ))}
    </div>
  );
}
