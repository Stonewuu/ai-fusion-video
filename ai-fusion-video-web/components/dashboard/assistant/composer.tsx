"use client";

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import { useRouter } from "next/navigation";
import { Loader2, Send, Settings2, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { aiModelApi, type AiModel } from "@/lib/api/ai-model";
import { getModelDisplayParts } from "@/lib/model-display";
import { useAssistantStore } from "@/lib/store/assistant-store";

interface AssistantComposerProps {
  active: boolean;
  projectId?: number | null;
  inputRef?: RefObject<HTMLTextAreaElement | null>;
  onHeightChange: (height: number) => void;
}

export function AssistantComposer({ active, projectId, inputRef, onHeightChange }: AssistantComposerProps) {
  const router = useRouter();
  const selectedConversationId = useAssistantStore((state) => state.selectedConversationId);
  const text = useAssistantStore((state) =>
    selectedConversationId
      ? state.conversationStates[selectedConversationId]?.draft ?? ""
      : state.newDraft,
  );
  const runtimeStatus = useAssistantStore((state) =>
    selectedConversationId ? state.conversationStates[selectedConversationId]?.status : undefined,
  );
  const runtimeMessagesError = useAssistantStore((state) =>
    selectedConversationId ? state.conversationStates[selectedConversationId]?.messagesError : undefined,
  );
  const conversationProjectId = useAssistantStore((state) =>
    selectedConversationId ? state.conversationStates[selectedConversationId]?.conversation.projectId : undefined,
  );
  const selectedModelId = useAssistantStore((state) => state.selectedModelId);
  const setDraft = useAssistantStore((state) => state.setDraft);
  const setSelectedModelId = useAssistantStore((state) => state.setSelectedModelId);
  const sendMessage = useAssistantStore((state) => state.sendMessage);
  const stopGeneration = useAssistantStore((state) => state.stopGeneration);
  const connectionConversationId = useAssistantStore((state) => state.connection?.conversationId);
  const [models, setModels] = useState<AiModel[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [composing, setComposing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const localInputRef = useRef<HTMLTextAreaElement>(null);
  const cancelling = runtimeStatus === "CANCEL_REQUESTED";
  const running = !!runtimeStatus && ["running", "pending", "RUNNING", "WAITING_CONFIRMATION", "WAITING_EXTERNAL", "CANCEL_REQUESTED"].includes(runtimeStatus);
  const currentConnection = connectionConversationId === selectedConversationId;
  const effectiveProjectId = selectedConversationId ? conversationProjectId : projectId;
  const selectedModel = models.find((model) => model.id === selectedModelId) ?? null;
  const sendDisabled = !text.trim() || !selectedModelId || !models.length || submitting;

  useEffect(() => {
    if (!active || modelsLoaded || modelsLoading) return;
    setModelsLoading(true);
    setModelsError(null);
    void aiModelApi.listByType(1)
      .then((result) => {
        setModels(result);
        const preferred = selectedModelId && result.some((model) => model.id === selectedModelId)
          ? selectedModelId
          : result.find((model) => model.defaultModel)?.id ?? result[0]?.id ?? null;
        if (preferred !== selectedModelId) setSelectedModelId(preferred);
      })
      .catch((requestError: unknown) => setModelsError(requestError instanceof Error ? requestError.message : "模型加载失败"))
      .finally(() => {
        setModelsLoading(false);
        setModelsLoaded(true);
      });
  }, [active, modelsLoaded, modelsLoading, selectedModelId, setSelectedModelId]);

  useLayoutEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => onHeightChange(element.getBoundingClientRect().height));
    observer.observe(element);
    onHeightChange(element.getBoundingClientRect().height);
    return () => observer.disconnect();
  }, [onHeightChange]);

  useEffect(() => {
    const element = inputRef?.current ?? localInputRef.current;
    if (active && element) {
      const frame = requestAnimationFrame(() => element.focus({ preventScroll: true }));
      return () => cancelAnimationFrame(frame);
    }
    return undefined;
  }, [active, inputRef]);

  const updateText = (value: string) => setDraft(selectedConversationId, value);

  const submit = async () => {
    if (!text.trim() || submitting || running || !selectedModelId || !models.length) return;
    setSubmitting(true);
    setError(null);
    try {
      await sendMessage(text, selectedModelId, projectId);
      updateText("");
    } catch (submitError: unknown) {
      setError(submitError instanceof Error ? submitError.message : "发送失败，请重试");
    } finally {
      setSubmitting(false);
    }
  };

  const selectItems = models.map((model) => ({ value: String(model.id), label: model.name }));

  return (
    <div ref={containerRef} className="pointer-events-none absolute inset-x-0 bottom-0 z-10 px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
      <div className="pointer-events-auto mx-auto w-full max-w-3xl rounded-2xl border border-border/20 bg-card/90 p-2 shadow-xl backdrop-blur-xl backdrop-saturate-150">
        {error || modelsError || runtimeMessagesError ? (
          <div className="mb-2 flex items-center justify-between gap-2 rounded-xl border border-destructive/20 bg-destructive/5 px-2.5 py-1.5 text-[11px] text-destructive">
            <span className="min-w-0 flex-1 truncate">{error || modelsError || runtimeMessagesError}</span>
            {modelsError ? <Button type="button" variant="destructive-ghost" size="xs" onClick={() => { setModels([]); setModelsError(null); setModelsLoaded(false); }}>重试</Button> : null}
          </div>
        ) : null}

        <Textarea
          ref={(element) => {
            localInputRef.current = element;
            if (inputRef) inputRef.current = element;
          }}
          value={text}
          rows={3}
          placeholder={models.length ? "发挥想象…" : "请先配置一个对话模型"}
          disabled={!models.length || modelsLoading || submitting || running}
          data-assistant-interactive="true"
          className="min-h-20 max-h-40 border-0 bg-transparent px-2 py-2 text-sm shadow-none focus-visible:ring-0"
          onChange={(event) => updateText(event.target.value)}
          onCompositionStart={() => setComposing(true)}
          onCompositionEnd={() => setComposing(false)}
          onKeyDown={(event) => {
            if (event.key !== "Enter" || !event.ctrlKey || composing) return;
            event.preventDefault();
            void submit();
          }}
        />

        <div className="flex flex-wrap items-center gap-2 px-1 pt-1">
          <div className="flex min-w-0 items-center gap-2">
            {modelsLoading ? <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none text-muted-foreground" /> : null}
            {models.length ? (
              <Select
                items={selectItems}
                value={selectedModelId ? String(selectedModelId) : null}
                onValueChange={(value) => setSelectedModelId(value ? Number(value) : null)}
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
                      <span className="min-w-0 flex-1 text-left leading-tight">
                        <span className="block truncate text-[11px] font-medium">
                          {getModelDisplayParts(selectedModel).name}
                        </span>
                        <span className="block truncate font-mono text-[9px] text-muted-foreground">
                          {selectedModel.code}
                        </span>
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
            ) : (
              <Button type="button" variant="link" size="xs" onClick={() => router.push("/settings/ai-models")}>
                <Settings2 /> 模型设置
              </Button>
            )}
            {effectiveProjectId ? <span className="max-w-32 truncate rounded-full bg-muted px-2 py-1 text-[10px] text-muted-foreground" title={`当前项目 #${effectiveProjectId}`}>项目上下文</span> : null}
          </div>

          <div className="ml-auto flex items-center gap-2">
            {cancelling ? (
              <Button
                type="button"
                variant="destructive"
                size="sm"
                data-assistant-interactive="true"
                disabled
              >
                <Loader2 className="animate-spin motion-reduce:animate-none" /> 取消中…
              </Button>
            ) : running || currentConnection ? (
              <Button
                type="button"
                variant="destructive"
                size="sm"
                data-assistant-interactive="true"
                onClick={() => {
                  setError(null);
                  void stopGeneration().catch((stopError: unknown) => setError(stopError instanceof Error ? stopError.message : "停止失败"));
                }}
              >
                <Square /> 停止
              </Button>
            ) : (
              <Tooltip>
                <TooltipTrigger
                  render={
                    <span
                      className="inline-flex"
                      tabIndex={sendDisabled ? 0 : undefined}
                      aria-label={sendDisabled ? "发送快捷键" : undefined}
                      data-assistant-interactive="true"
                    >
                      <Button
                        type="button"
                        variant="default"
                        size="sm"
                        data-assistant-interactive="true"
                        disabled={sendDisabled}
                        onClick={() => void submit()}
                      >
                        {submitting ? <Loader2 className="animate-spin" /> : <Send />}
                        发送
                      </Button>
                    </span>
                  }
                />
                <TooltipContent>Enter 换行 · Ctrl+Enter 发送</TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
