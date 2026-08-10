"use client";

import { useState, type RefObject } from "react";
import {
  Ban,
  CheckCircle2,
  CircleAlert,
  Clapperboard,
  Images,
  Lightbulb,
  Loader2,
  MessageSquarePlus,
  MoreHorizontal,
  Trash2,
  WandSparkles,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { AssistantBrandIcon } from "./assistant-brand-icon";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { AgentConversation } from "@/lib/api/ai-assistant";
import { useAssistantStore } from "@/lib/store/assistant-store";
import { cn } from "@/lib/utils";

function isRunningStatus(status?: string) {
  return !!status && ["running", "pending", "RUNNING", "WAITING_CONFIRMATION", "WAITING_EXTERNAL", "CANCEL_REQUESTED"].includes(status);
}

function relativeTime(value?: string) {
  if (!value) return "刚刚";
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return "刚刚";
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)} 天前`;
  return new Date(timestamp).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function statusLabel(status?: string) {
  if (status === "CANCEL_REQUESTED") return "取消中";
  if (status === "WAITING_CONFIRMATION") return "待确认";
  if (status === "WAITING_EXTERNAL") return "等待执行";
  if (isRunningStatus(status)) return "运行中";
  if (status === "failed" || status === "error") return "失败";
  if (status === "cancelled") return "已取消";
  return "已完成";
}

function StatusIcon({ status, unread }: { status?: string; unread: boolean }) {
  if (status === "CANCEL_REQUESTED") return <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none text-muted-foreground" />;
  if (status === "WAITING_CONFIRMATION" || status === "WAITING_EXTERNAL") {
    return <CircleAlert className="size-3.5 text-primary" />;
  }
  if (isRunningStatus(status)) return <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none text-primary" />;
  if (status === "failed" || status === "error") return <XCircle className="size-3.5 text-destructive" />;
  if (status === "cancelled") return <Ban className="size-3.5 text-muted-foreground" />;
  if (unread) return <CircleAlert className="size-3.5 text-primary" />;
  return <CheckCircle2 className="size-3.5 text-emerald-500" />;
}

interface DeleteTarget {
  conversationId: string;
  id: number;
  title: string;
}

function ConversationRow({
  conversation,
  selected,
  onSelected,
  onDelete,
}: {
  conversation: AgentConversation;
  selected: boolean;
  onSelected?: () => void;
  onDelete: (target: DeleteTarget) => void;
}) {
  const status = useAssistantStore((state) =>
    state.conversationStates[conversation.conversationId]?.status,
  );
  const unread = useAssistantStore((state) =>
    state.conversationStates[conversation.conversationId]?.unread ?? false,
  );
  const selectConversation = useAssistantStore((state) => state.selectConversation);
  const running = isRunningStatus(status);

  return (
    <div
      role="button"
      tabIndex={0}
      className={cn(
        "group mb-1 flex cursor-pointer select-none items-start gap-2 rounded-xl border border-transparent px-2.5 py-2 text-left transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
        selected
          ? "border-primary/30 bg-primary/10"
          : "hover:border-border/30 hover:bg-muted/60",
      )}
      onClick={() => {
        selectConversation(conversation.conversationId);
        onSelected?.();
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectConversation(conversation.conversationId);
          onSelected?.();
        }
      }}
    >
      <div className="flex min-w-0 flex-1 items-start gap-2">
        <span className="mt-0.5 shrink-0"><StatusIcon status={status} unread={unread} /></span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs font-medium" title={conversation.title}>
            {conversation.title || "新对话"}
          </span>
          <span className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <span>{statusLabel(status)}</span>
            <span aria-hidden="true">·</span>
            <span>{relativeTime(conversation.lastMessageTime ?? conversation.createTime)}</span>
            {unread ? <span className="size-1.5 rounded-full bg-primary" aria-label="未读" /> : null}
          </span>
        </span>
      </div>
      <Button
        type="button"
        variant="destructive-ghost"
        size="icon-xs"
        aria-label={`删除会话 ${conversation.title}`}
        title={running ? "运行中的会话不能删除" : "删除会话"}
        disabled={running}
        data-assistant-interactive="true"
        className="mt-0.5 shrink-0 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
        onClick={(event) => {
          event.stopPropagation();
          onDelete({
            conversationId: conversation.conversationId,
            id: conversation.id,
            title: conversation.title,
          });
        }}
      >
        <Trash2 />
      </Button>
    </div>
  );
}

function ConversationList({ onSelected }: { onSelected?: () => void }) {
  const conversations = useAssistantStore((state) => state.conversations);
  const selectedId = useAssistantStore((state) => state.selectedConversationId);
  const loading = useAssistantStore((state) => state.conversationsLoading);
  const hasMore = useAssistantStore((state) => state.hasMoreConversations);
  const loadMore = useAssistantStore((state) => state.loadMoreConversations);
  const startNewConversation = useAssistantStore((state) => state.startNewConversation);
  const deleteConversation = useAssistantStore((state) => state.deleteConversation);
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  return (
    <>
      <div className="flex shrink-0 items-center justify-between border-b border-border/20 px-3 py-3">
        <div>
          <p className="text-xs font-semibold">会话</p>
          <p className="text-[10px] text-muted-foreground">仅显示助手对话</p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="新建对话"
          title="新建对话"
          data-assistant-interactive="true"
          onClick={() => {
            startNewConversation();
            onSelected?.();
          }}
        >
          <MessageSquarePlus />
        </Button>
      </div>

      <OverlayScrollArea
        className="min-h-0 flex-1"
        viewportClassName="p-2"
        onViewportScroll={(event) => {
          const element = event.currentTarget;
          if (hasMore && !loading && element.scrollHeight - element.scrollTop - element.clientHeight < 80) loadMore();
        }}
      >
        {conversations.length === 0 && !loading ? (
          <div className="flex flex-col items-center gap-2 px-4 py-10 text-center text-muted-foreground">
            <MessageSquarePlus className="size-7 opacity-40" />
            <p className="text-xs">还没有助手会话</p>
            <Button type="button" variant="outline" size="sm" onClick={() => startNewConversation()}>
              开始新对话
            </Button>
          </div>
        ) : null}

        {conversations.map((conversation) => (
          <ConversationRow
            key={conversation.conversationId}
            conversation={conversation}
            selected={selectedId === conversation.conversationId}
            onSelected={onSelected}
            onDelete={(target) => {
              setDeleteError(null);
              setDeleteTarget(target);
            }}
          />
        ))}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" /> 加载中
          </div>
        ) : null}
        {!loading && hasMore ? (
          <button type="button" className="flex w-full items-center justify-center gap-1 py-3 text-[10px] text-muted-foreground hover:text-foreground" onClick={loadMore}>
            <MoreHorizontal className="size-3" /> 加载更多
          </button>
        ) : null}
      </OverlayScrollArea>

      <Dialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <DialogContent className="max-h-[calc(100vh-2rem)] overflow-hidden rounded-2xl">
          <DialogHeader>
            <DialogTitle>删除这条会话？</DialogTitle>
            <DialogDescription>
              “{deleteTarget?.title || "新对话"}”的消息记录会从当前账户移除，此操作无法撤销。
            </DialogDescription>
          </DialogHeader>
          {deleteError ? <p className="text-xs text-destructive">{deleteError}</p> : null}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setDeleteTarget(null)}>取消</Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => {
                if (!deleteTarget) return;
                void deleteConversation(deleteTarget.conversationId, deleteTarget.id)
                  .then(() => setDeleteTarget(null))
                  .catch((error: unknown) => setDeleteError(error instanceof Error ? error.message : "删除失败"));
              }}
            >
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export function ConversationNavigation({ onSelected }: { onSelected?: () => void }) {
  const drawerOpen = useAssistantStore((state) => state.drawerOpen);
  const setDrawerOpen = useAssistantStore((state) => state.setDrawerOpen);
  return (
    <>
      <aside className={cn("assistant-navigation", drawerOpen && "assistant-navigation-open")}>
        <ConversationList onSelected={onSelected} />
      </aside>
      {drawerOpen ? (
        <button
          type="button"
          aria-label="关闭会话列表"
          className="assistant-navigation-overlay assistant-narrow-only absolute inset-0 z-10 bg-background/25"
          onClick={() => setDrawerOpen(false)}
        />
      ) : null}
    </>
  );
}

const STARTER_PROMPTS = [
  {
    prompt: "怎么把创意整理成视频脚本？",
    icon: Clapperboard,
    iconClassName: "text-sky-500",
  },
  {
    prompt: "能帮我设计一组连贯分镜吗？",
    icon: Images,
    iconClassName: "text-violet-500",
  },
  {
    prompt: "如何统一画面提示词的风格？",
    icon: WandSparkles,
    iconClassName: "text-emerald-500",
  },
  {
    prompt: "有哪些适合短视频的创意方向？",
    icon: Lightbulb,
    iconClassName: "text-orange-500",
  },
] as const;

export function AssistantEmptyState({
  inputRef,
  bottomInset,
}: {
  inputRef: RefObject<HTMLTextAreaElement | null>;
  bottomInset: number;
}) {
  const setDraft = useAssistantStore((state) => state.setDraft);

  const selectPrompt = (prompt: string) => {
    setDraft(null, prompt);
    requestAnimationFrame(() => {
      inputRef.current?.focus({ preventScroll: true });
      inputRef.current?.setSelectionRange(prompt.length, prompt.length);
    });
  };

  return (
    <OverlayScrollArea
      className="min-h-0 flex-1"
      viewportStyle={{
        paddingBottom: bottomInset,
        scrollPaddingBottom: bottomInset,
      }}
    >
      <div className="flex min-h-full w-full flex-col p-6">
        <div className="my-auto flex w-full flex-col items-center gap-10">
          <div className="space-y-6 text-center">
            <AssistantBrandIcon className="mx-auto size-20" />
            <h2 className="text-2xl font-medium tracking-tight text-foreground">
              今天想创作什么？
            </h2>
          </div>
          <div className="assistant-starter-prompts grid w-full max-w-5xl gap-3">
            {STARTER_PROMPTS.map(({ prompt, icon: Icon, iconClassName }) => (
              <button
                key={prompt}
                type="button"
                className="flex min-h-32 flex-col justify-between rounded-xl border border-border/30 bg-card p-4 text-left shadow-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50"
                onClick={() => selectPrompt(prompt)}
              >
                <Icon className={cn("size-5", iconClassName)} strokeWidth={1.8} />
                <span className="text-sm font-medium leading-6 text-foreground">{prompt}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </OverlayScrollArea>
  );
}
