"use client";

import {
  startTransition,
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
import { AssistantReferenceContext } from "./reference-context";
import { AssistantReferencePicker } from "./reference-picker";
import { useAssistantReferences } from "./use-assistant-references";

interface AssistantComposerProps {
  active: boolean;
  tooltipsEnabled: boolean;
  projectId?: number | null;
  inputRef?: RefObject<HTMLTextAreaElement | null>;
  onHeightChange: (height: number) => void;
}

let cachedAssistantModels: AiModel[] | null = null;
let cachedAssistantModelsAt = 0;
let assistantModelsRequest: Promise<AiModel[]> | null = null;
const ASSISTANT_MODELS_CACHE_TTL_MS = 30_000;

function loadAssistantModels(): Promise<AiModel[]> {
  const cached = cachedAssistantModels;
  if (cached && Date.now() - cachedAssistantModelsAt < ASSISTANT_MODELS_CACHE_TTL_MS) {
    return Promise.resolve(cached);
  }
  if (!assistantModelsRequest) {
    assistantModelsRequest = aiModelApi.listByType(1)
      .then((result) => {
        cachedAssistantModels = result;
        cachedAssistantModelsAt = Date.now();
        assistantModelsRequest = null;
        return result;
      })
      .catch((error: unknown) => {
        assistantModelsRequest = null;
        throw error;
      });
  }
  return assistantModelsRequest;
}

export function AssistantComposer({ active, tooltipsEnabled, projectId, inputRef, onHeightChange }: AssistantComposerProps) {
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
  const startNewConversation = useAssistantStore((state) => state.startNewConversation);
  const sendMessage = useAssistantStore((state) => state.sendMessage);
  const stopGeneration = useAssistantStore((state) => state.stopGeneration);
  const connectionConversationId = useAssistantStore((state) => state.connection?.conversationId);
  const [models, setModels] = useState<AiModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelLoadAttempt, setModelLoadAttempt] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [composing, setComposing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const composerSurfaceRef = useRef<HTMLDivElement>(null);
  const localInputRef = useRef<HTMLTextAreaElement>(null);
  const cancelling = runtimeStatus === "CANCEL_REQUESTED";
  const running = !!runtimeStatus && ["running", "pending", "RUNNING", "WAITING_CONFIRMATION", "WAITING_EXTERNAL", "CANCEL_REQUESTED"].includes(runtimeStatus);
  const currentConnection = connectionConversationId === selectedConversationId;
  const effectiveProjectId = selectedConversationId ? conversationProjectId : projectId;
  const selectedModel = models.find((model) => model.id === selectedModelId) ?? null;
  const sendDisabled = !text.trim() || !selectedModelId || !models.length || submitting;
  const updateText = (value: string) => setDraft(selectedConversationId, value);
  const references = useAssistantReferences({
    active,
    text,
    selectedConversationId,
    effectiveProjectId,
    inputRef: localInputRef,
    updateText,
  });

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let idleHandle: number | null = null;
    let fallbackTimer: ReturnType<typeof setTimeout> | null = null;

    const beginLoad = () => {
      if (cancelled) return;
      void loadAssistantModels()
      .then((result) => {
        if (cancelled) return;
        startTransition(() => {
          setModels(result);
          const currentSelectedModelId = useAssistantStore.getState().selectedModelId;
          const preferred = currentSelectedModelId && result.some((model) => model.id === currentSelectedModelId)
            ? currentSelectedModelId
            : result.find((model) => model.defaultModel)?.id ?? result[0]?.id ?? null;
          if (preferred !== currentSelectedModelId) setSelectedModelId(preferred);
          setModelsLoading(false);
        });
      })
      .catch((requestError: unknown) => {
        if (cancelled) return;
        setModelsError(requestError instanceof Error ? requestError.message : "模型加载失败");
        setModelsLoading(false);
      });
    };

    if (modelLoadAttempt > 0 || cachedAssistantModels) {
      beginLoad();
    } else if ("requestIdleCallback" in window) {
      idleHandle = window.requestIdleCallback(beginLoad, { timeout: 1200 });
    } else {
      fallbackTimer = setTimeout(beginLoad, 250);
    }

    return () => {
      cancelled = true;
      if (idleHandle !== null) window.cancelIdleCallback(idleHandle);
      if (fallbackTimer !== null) clearTimeout(fallbackTimer);
    };
  }, [active, modelLoadAttempt, setSelectedModelId]);

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

  const submit = async () => {
    if (!text.trim() || submitting || running || !selectedModelId || !models.length) return;
    setSubmitting(true);
    setError(null);
    try {
      const referencedProjectId = references.selectedProject?.id ?? null;
      if (selectedConversationId
        && referencedProjectId !== (conversationProjectId ?? null)) {
        setDraft(null, text);
        startNewConversation();
      }
      await sendMessage(
        text,
        selectedModelId,
        referencedProjectId,
        {
          skills: references.selectedSkills,
          mcpTools: references.selectedMcpTools,
        },
      );
      updateText("");
      references.clearTransientReferences();
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
            {modelsError ? <Button type="button" variant="destructive-ghost" size="xs" onClick={() => { setModels([]); setModelsLoading(true); setModelsError(null); setModelLoadAttempt((attempt) => attempt + 1); }}>重试</Button> : null}
          </div>
        ) : null}

        <div
          ref={composerSurfaceRef}
          className="min-w-0"
          data-assistant-interactive="true"
        >
          <AssistantReferenceContext
            project={references.selectedProject}
            skills={references.selectedSkills}
            mcpTools={references.selectedMcpTools}
            onRemoveProject={references.removeProject}
            onRemoveSkill={references.removeSkill}
            onRemoveMcpTool={references.removeMcpTool}
          />
          <Textarea
            ref={(element) => {
              localInputRef.current = element;
              if (inputRef) inputRef.current = element;
            }}
            value={text}
            rows={3}
            placeholder="发挥想象…"
            disabled={!models.length || modelsLoading || submitting || running}
            data-assistant-interactive="true"
            aria-expanded={!!references.picker}
            aria-label="助手消息；输入 @ 引用项目，输入 / 引用 Skill 或 MCP"
            className="min-h-20 max-h-40 border-0 bg-transparent px-2 py-2 text-sm shadow-none focus-visible:ring-0"
            onChange={(event) => references.updateTextWithTrigger(
              event.target.value,
              event.target.selectionStart ?? event.target.value.length,
            )}
            onCompositionStart={() => setComposing(true)}
            onCompositionEnd={() => setComposing(false)}
            onBlur={references.closePicker}
            onKeyDown={(event) => {
              if (references.handlePickerKeyDown(event)) return;
              if (event.key !== "Enter" || !event.ctrlKey || composing) return;
              event.preventDefault();
              void submit();
            }}
          />
        </div>

        <div className="flex flex-wrap items-center gap-2 px-1 pt-1">
          <div className="flex min-w-0 items-center gap-2">
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
              <Tooltip disabled={!tooltipsEnabled}>
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
      {references.picker ? (
        <AssistantReferencePicker
          anchorRef={composerSurfaceRef}
          mode={references.picker.mode}
          query={references.picker.query}
          items={references.pickerItems}
          activeIndex={references.activeIndex}
          loading={references.pickerLoading}
          error={references.pickerError}
          selectedKeys={references.selectedKeys}
          onActiveIndexChange={references.setActiveIndex}
          onSelect={references.selectPickerItem}
        />
      ) : null}
    </div>
  );
}
