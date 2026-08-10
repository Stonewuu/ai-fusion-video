"use client";

import { Brain, Loader2, Plus, Settings2 } from "lucide-react";
import { useRouter } from "next/navigation";
import type { RefObject } from "react";
import { Button } from "@/components/ui/button";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { AiModel } from "@/lib/api/ai-model";
import { getModelDisplayParts } from "@/lib/model-display";

interface AssistantModelControlsProps {
  models: AiModel[];
  modelsLoading: boolean;
  selectedModel: AiModel | null;
  selectedModelId: number | null;
  selectedReasoningEffort: string | null;
  tooltipsEnabled: boolean;
  fileInputRef: RefObject<HTMLInputElement | null>;
  fileAccept: string;
  addingFiles: boolean;
  canAddFiles: boolean;
  capabilitySummary: string;
  disabled: boolean;
  onModelChange: (modelId: number | null) => void;
  onReasoningEffortChange: (effort: string | null) => void;
  onFilesSelected: (files: File[]) => void;
}

export function AssistantModelControls({
  models,
  modelsLoading,
  selectedModel,
  selectedModelId,
  selectedReasoningEffort,
  tooltipsEnabled,
  fileInputRef,
  fileAccept,
  addingFiles,
  canAddFiles,
  capabilitySummary,
  disabled,
  onModelChange,
  onReasoningEffortChange,
  onFilesSelected,
}: AssistantModelControlsProps) {
  const router = useRouter();
  const selectItems = models.map((model) => ({ value: String(model.id), label: model.name }));
  const reasoningEffortLevels = selectedModel?.supportReasoning
    ? selectedModel.reasoningEffortLevels ?? []
    : [];
  const reasoningItems = reasoningEffortLevels.map((level) => ({ value: level, label: level }));

  return (
    <div className="flex min-w-0 flex-1 flex-nowrap items-center gap-1">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={fileAccept}
        className="hidden"
        onChange={(event) => onFilesSelected(Array.from(event.target.files ?? []))}
      />
      <Tooltip disabled={!tooltipsEnabled}>
        <TooltipTrigger
          render={
            <span className="inline-flex" data-assistant-interactive="true">
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                disabled={!canAddFiles || addingFiles || disabled}
                aria-label={capabilitySummary}
                title={capabilitySummary}
                onClick={() => fileInputRef.current?.click()}
              >
                {addingFiles ? <Loader2 className="animate-spin motion-reduce:animate-none" /> : <Plus />}
              </Button>
            </span>
          }
        />
        <TooltipContent>{capabilitySummary}</TooltipContent>
      </Tooltip>

      {models.length ? (
        <Select
          items={selectItems}
          value={selectedModelId ? String(selectedModelId) : null}
          onValueChange={(value) => onModelChange(value ? Number(value) : null)}
        >
          <SelectTrigger
            data-assistant-interactive="true"
            className="max-w-56 border-0 bg-transparent px-2 text-xs shadow-none focus-visible:ring-0"
            aria-label="选择对话模型"
          >
            <SelectValue className="sr-only" placeholder="选择模型" />
            {selectedModel ? (
              <>
                <ModelVendorIcon source={selectedModel} className="size-4" />
                <span className="min-w-0 flex-1 truncate text-left text-[11px] font-medium">
                  {getModelDisplayParts(selectedModel).name}
                </span>
              </>
            ) : null}
          </SelectTrigger>
          <SelectContent className="min-w-64 text-xs">
            <SelectGroup>
              {models.map((model) => (
                <SelectItem key={model.id} value={String(model.id)} className="min-h-11 py-2 text-xs">
                  <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-muted/70">
                    <ModelVendorIcon source={model} className="size-4" />
                  </span>
                  <span className="min-w-0 flex-1 leading-tight">
                    <span className="block truncate text-xs font-medium">
                      {getModelDisplayParts(model).name}
                    </span>
                    <span className="mt-0.5 block truncate font-mono text-[10px] text-muted-foreground">
                      {model.code}
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      ) : modelsLoading ? (
        <div className="flex h-9 w-48 items-center gap-2 px-2 text-xs text-muted-foreground" aria-label="加载对话模型">
          <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none" />
          <span>加载对话模型</span>
        </div>
      ) : (
        <Button type="button" variant="link" size="xs" onClick={() => router.push("/settings/ai-models")}>
          <Settings2 /> 模型设置
        </Button>
      )}

      {reasoningEffortLevels.length ? (
        <Select
          items={reasoningItems}
          value={selectedReasoningEffort}
          onValueChange={(value) => onReasoningEffortChange(value ? String(value) : null)}
        >
          <SelectTrigger
            data-assistant-interactive="true"
            className="max-w-40 border-0 bg-transparent px-2 text-xs shadow-none focus-visible:ring-0"
            aria-label={`思考强度：${selectedReasoningEffort ?? "未选择"}`}
            title="调整模型回答前的思考强度；强度越高，通常响应越慢"
          >
            <SelectValue className="sr-only" placeholder="选择思考强度" />
            <Brain className="size-4 text-muted-foreground" />
            <span className="truncate text-[11px] font-medium">
              思考 · <span className="font-mono">{selectedReasoningEffort}</span>
            </span>
          </SelectTrigger>
          <SelectContent className="text-xs">
            <SelectGroup>
              <SelectLabel className="px-2 pt-2 pb-1 font-medium  text-muted-foreground/70">
                思考强度
              </SelectLabel>
              {reasoningEffortLevels.map((level) => (
                <SelectItem key={level} value={level} className="text-xs">
                  {level}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      ) : null}
    </div>
  );
}
