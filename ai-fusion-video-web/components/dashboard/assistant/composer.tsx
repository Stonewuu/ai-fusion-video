"use client";

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import { Loader2, Send, Square, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useAssistantStore } from "@/lib/store/assistant-store";
import { AssistantReferenceContext } from "./reference-context";
import { AssistantReferencePicker } from "./reference-picker";
import { useAssistantReferences } from "./use-assistant-references";
import { AssistantAttachmentStrip } from "./attachment-strip";
import { useAssistantAttachments } from "./use-assistant-attachments";
import { AssistantModelControls } from "./model-controls";
import { AssistantToolPermissionControl } from "./tool-permission-control";
import { useCanvasAssistantSession } from "@/lib/canvas/assistant-session";
import { AssistantCanvasContext } from "./canvas-context";
import { useAssistantModels } from "./use-assistant-models";

interface AssistantComposerProps {
  active: boolean;
  tooltipsEnabled: boolean;
  projectId?: number | null;
  inputRef?: RefObject<HTMLTextAreaElement | null>;
  onHeightChange: (height: number) => void;
}

export function AssistantComposer({ active, tooltipsEnabled, projectId, inputRef, onHeightChange }: AssistantComposerProps) {
  const userId = useAssistantStore((state) => state.hydratedUserId);
  const canvasBinding = useCanvasAssistantSession((state) => state.binding);
  const selectedConversationId = useAssistantStore((state) => state.selectedConversationId);
  const text = useAssistantStore((state) => {
    const conversationId = state.selectedConversationId;
    if (!conversationId) return state.newDraft;
    const runtime = state.conversationStates[conversationId];
    if (!runtime) throw new Error(`Missing assistant runtime for ${conversationId}`);
    return runtime.draft;
  });
  const runtimeStatus = useAssistantStore((state) => {
    const conversationId = state.selectedConversationId;
    if (!conversationId) return undefined;
    const runtime = state.conversationStates[conversationId];
    if (!runtime) throw new Error(`Missing assistant runtime for ${conversationId}`);
    return runtime.status;
  });
  const runtimeMessagesError = useAssistantStore((state) => {
    const conversationId = state.selectedConversationId;
    if (!conversationId) return undefined;
    const runtime = state.conversationStates[conversationId];
    if (!runtime) throw new Error(`Missing assistant runtime for ${conversationId}`);
    return runtime.messagesError;
  });
  const conversationProjectId = useAssistantStore((state) => {
    const conversationId = state.selectedConversationId;
    if (!conversationId) return undefined;
    const runtime = state.conversationStates[conversationId];
    if (!runtime) throw new Error(`Missing assistant runtime for ${conversationId}`);
    return runtime.conversation.projectId;
  });
  const selectedModelId = useAssistantStore((state) => state.selectedModelId);
  const toolExecutionMode = useAssistantStore((state) => {
    const conversationId = state.selectedConversationId;
    if (!conversationId) return state.newToolExecutionMode;
    const runtime = state.conversationStates[conversationId];
    if (!runtime) {
      throw new Error(`Missing assistant runtime for ${conversationId}`);
    }
    return runtime.toolExecutionMode;
  });
  const setDraft = useAssistantStore((state) => state.setDraft);
  const setSelectedModelId = useAssistantStore((state) => state.setSelectedModelId);
  const setToolExecutionMode = useAssistantStore((state) => state.setToolExecutionMode);
  const startNewConversation = useAssistantStore((state) => state.startNewConversation);
  const sendMessage = useAssistantStore((state) => state.sendMessage);
  const stopGeneration = useAssistantStore((state) => state.stopGeneration);
  const connectionConversationId = useAssistantStore((state) => state.connection?.conversationId);
  const [modelLoadAttempt, setModelLoadAttempt] = useState(0);
  const { models, modelsLoading, modelsError } = useAssistantModels(userId, active, modelLoadAttempt);
  const [selectedReasoningEffort, setSelectedReasoningEffort] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dismissedAlert, setDismissedAlert] = useState<string | null>(null);
  const [composing, setComposing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const composerSurfaceRef = useRef<HTMLDivElement>(null);
  const localInputRef = useRef<HTMLTextAreaElement>(null);
  const cancelling = runtimeStatus === "CANCEL_REQUESTED";
  const running = !!runtimeStatus && ["running", "pending", "RUNNING", "WAITING_CONFIRMATION", "WAITING_EXTERNAL", "CANCEL_REQUESTED"].includes(runtimeStatus);
  const currentConnection = connectionConversationId === selectedConversationId;
  const effectiveProjectId = canvasBinding ? canvasBinding.projectId : selectedConversationId ? conversationProjectId : projectId;
  const selectedModel = models.find((model) => model.id === selectedModelId) ?? null;
  const updateText = (value: string) => setDraft(selectedConversationId, value);
  const references = useAssistantReferences({
    userId,
    agentType: canvasBinding ? "canvas_assistant" : "ai_assistant_agent",
    active,
    text,
    selectedConversationId,
    effectiveProjectId,
    inputRef: localInputRef,
    updateText,
  });
  const attachments = useAssistantAttachments({
    conversationId: selectedConversationId,
    model: selectedModel,
    onError: (attachmentError) => {
      setDismissedAlert(null);
      setError(attachmentError instanceof Error ? attachmentError.message : "添加附件失败");
    },
  });
  const alertMessage = error || modelsError || runtimeMessagesError || attachments.compatibilityError;
  const visibleAlertMessage = alertMessage !== dismissedAlert ? alertMessage : null;
  const sendDisabled = (!text.trim() && !attachments.attachments.length)
    || !selectedModelId
    || !models.length
    || submitting
    || attachments.uploading
    || !!attachments.compatibilityError;

  useEffect(() => {
    setDismissedAlert(null);
  }, [alertMessage]);

  useEffect(() => {
    const firstLevel = selectedModel?.supportReasoning
      ? selectedModel.reasoningEffortLevels?.[0] ?? null
      : null;
    setSelectedReasoningEffort(firstLevel);
  }, [selectedModel?.id, selectedModel?.supportReasoning, selectedModel?.reasoningEffortLevels]);

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
    if ((!text.trim() && !attachments.attachments.length)
      || submitting
      || running
      || !selectedModelId
      || !models.length
      || attachments.uploading
      || attachments.compatibilityError) return;
    setSubmitting(true);
    setError(null);
    try {
      const canvas = canvasBinding ? await canvasBinding.prepare() : undefined;
      if (canvas && references.selectedProject && references.selectedProject.id !== canvas.projectId)
        throw new Error("引用项目与当前画布不一致，请移除该项目引用或切换到对应画布。");
      const referencedProjectId = canvas ? canvas.projectId : references.selectedProject?.id ?? null;
      if (selectedConversationId
        && referencedProjectId !== (conversationProjectId ?? null)) {
        setDraft(null, text);
        startNewConversation();
      }
      await sendMessage(
        text,
        selectedModelId,
        selectedReasoningEffort,
        referencedProjectId,
        {
          project: references.selectedProject,
          skills: references.selectedSkills,
          mcpTools: references.selectedMcpTools,
          canvas,
          multimodalInputs: attachments.attachments.map((attachment) => ({
            id: attachment.id,
            name: attachment.name,
            inputType: attachment.inputType,
            mimeType: attachment.mimeType,
            transport: attachment.transport,
            url: attachment.url,
            data: attachment.data,
            resourceUrl: attachment.resourceUrl,
            size: attachment.size,
          })),
        },
      );
      updateText("");
      attachments.clearAttachments();
      references.clearTransientReferences();
    } catch (submitError: unknown) {
      setError(submitError instanceof Error ? submitError.message : "发送失败，请重试");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div ref={containerRef} className="pointer-events-none absolute inset-x-0 bottom-0 z-10 px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
      <div className="pointer-events-auto mx-auto w-full max-w-3xl rounded-2xl border border-border/20 bg-card/90 p-2 shadow-xl backdrop-blur-xl backdrop-saturate-150">
        {visibleAlertMessage ? (
          <div className="mb-2 flex items-center justify-between gap-2 rounded-xl border border-destructive/20 bg-destructive/5 px-2.5 py-1.5 text-[11px] text-destructive">
            <span className="min-w-0 flex-1 truncate">{visibleAlertMessage}</span>
            <div className="flex shrink-0 items-center gap-2">
              {modelsError ? <Button type="button" variant="destructive-ghost" size="xs" onClick={() => setModelLoadAttempt((attempt) => attempt + 1)}>重试</Button> : null}
              <Button
                type="button"
                variant="destructive-ghost"
                size="icon-xs"
                aria-label="关闭错误提示"
                title="关闭"
                onClick={() => setDismissedAlert(visibleAlertMessage)}
              >
                <X />
              </Button>
            </div>
          </div>
        ) : null}

        <div
          ref={composerSurfaceRef}
          className="min-w-0"
          data-assistant-interactive="true"
        >
          <AssistantReferenceContext
            project={canvasBinding && references.selectedProject?.id === canvasBinding.projectId ? null : references.selectedProject}
            skills={references.selectedSkills}
            mcpTools={references.selectedMcpTools}
            onRemoveProject={references.removeProject}
            onRemoveSkill={references.removeSkill}
            onRemoveMcpTool={references.removeMcpTool}
          >
            {canvasBinding ? <AssistantCanvasContext /> : null}
          </AssistantReferenceContext>
          <AssistantAttachmentStrip
            attachments={attachments.attachments}
            disabled={submitting || running}
            onRemove={attachments.removeAttachment}
          />
          <Textarea
            ref={(element) => {
              localInputRef.current = element;
              if (inputRef) inputRef.current = element;
            }}
            value={text}
            rows={3}
            placeholder={canvasBinding ? "例如：整理选中的镜头，或把这段故事改成分镜…" : "发挥想象…\n使用 '@' 引用项目、'/' 引用 Skill"}
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
            onPaste={attachments.handlePaste}
            onBlur={references.closePicker}
            onKeyDown={(event) => {
              if (references.handlePickerKeyDown(event)) return;
              if (event.key !== "Enter" || !event.ctrlKey || composing) return;
              event.preventDefault();
              void submit();
            }}
          />
        </div>

        <div className="flex items-center gap-1 px-1 pt-1">
          <AssistantModelControls
            models={models}
            modelsLoading={modelsLoading}
            selectedModel={selectedModel}
            selectedModelId={selectedModelId}
            selectedReasoningEffort={selectedReasoningEffort}
            tooltipsEnabled={tooltipsEnabled}
            fileInputRef={attachments.fileInputRef}
            fileAccept={attachments.accept}
            addingFiles={attachments.uploading}
            canAddFiles={attachments.canAdd}
            capabilitySummary={attachments.capabilitySummary}
            disabled={submitting || running}
            onModelChange={setSelectedModelId}
            onReasoningEffortChange={setSelectedReasoningEffort}
            onFilesSelected={(files) => {
              setError(null);
              void attachments.addFiles(files).catch((uploadError: unknown) => setError(
                uploadError instanceof Error ? uploadError.message : "添加附件失败",
              ));
            }}
          />

          <AssistantToolPermissionControl
            value={toolExecutionMode}
            disabled={submitting || running}
            onChange={setToolExecutionMode}
          />

          <div className="ml-auto flex shrink-0 items-center gap-2">
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
                        variant="secondary"
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
