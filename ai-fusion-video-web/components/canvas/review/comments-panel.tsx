"use client";

import { useEffect, useState } from "react";
import {
  Check,
  LocateFixed,
  MessageSquare,
  Pencil,
  Reply,
  Trash2,
} from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasTextarea } from "../ui/canvas-field";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { canvasDate } from "@/lib/canvas/labels";
import type { CanvasComment } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { cn } from "@/lib/utils";

function mergeComments(current: CanvasComment[], incoming: CanvasComment[]) {
  const rows = new Map(current.map((item) => [item.id, item]));
  incoming.forEach((item) => {
    if (!rows.has(item.id) || rows.get(item.id)!.version <= item.version) {
      rows.set(item.id, item);
    }
  });
  return [...rows.values()].sort(
    (a, b) =>
      a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id),
  );
}

export function CanvasCommentsPanel() {
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const selected = useCanvas((state) => state.selection[0]);
  const userId = useAuthStore((state) => state.user?.id);
  const requests = useCommandRequestKey();
  const [items, setItems] = useState<CanvasComment[]>([]);
  const [offset, setOffset] = useState(0);
  const [more, setMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [commentId, setCommentId] = useState(() => crypto.randomUUID());
  const [anchor, setAnchor] = useState<string | null>(null);
  const [reply, setReply] = useState<CanvasComment | null>(null);
  const [editing, setEditing] = useState<CanvasComment | null>(null);
  const [busy, setBusy] = useState(false);
  const editable = snapshot.permissions.canEdit && !snapshot.document.archived;

  useEffect(() => {
    let active = true;
    void canvasApi
      .comments(snapshot.document.id)
      .then((result) => {
        if (active) {
          setItems((current) => mergeComments(current, result));
          setOffset((current) => Math.max(current, result.length));
          setMore(result.length === 50);
        }
      })
      .catch((failure) => {
        if (active) {
          setError(getApiErrorMessage(failure));
        }
      });
    return () => {
      active = false;
    };
  }, [snapshot.document.id, snapshot.document.revision]);

  const loadMore = async () => {
    setLoading(true);
    try {
      const result = await canvasApi.comments(snapshot.document.id, offset);
      setItems((current) => mergeComments(current, result));
      setOffset(offset + result.length);
      setMore(result.length === 50);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setLoading(false);
    }
  };

  const patch = async (
    comment: CanvasComment,
    change: { resolved?: boolean; deleted?: boolean; content?: string },
  ) => {
    const body = { expectedVersion: comment.version, ...change };
    setBusy(true);
    setError(null);
    try {
      await canvasApi.patchComment(snapshot.document.id, comment.id, {
        ...body,
        requestId: requests.forPayload({ commentId: comment.id, ...body }),
      });
      const updated = await canvasApi.commentDetail(
        snapshot.document.id,
        comment.id,
      );
      setItems((current) => mergeComments(current, [updated]));
      await store.getState().load();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
      throw failure;
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    if (!text.trim()) {
      return;
    }
    if (editing) {
      try {
        await patch(editing, { content: text });
        setEditing(null);
        setText("");
      } catch {
        // 错误已由 patch 提示，保留当前输入以便重试。
      }
      return;
    }
    const body = {
      id: commentId,
      nodeId: reply ? reply.nodeId : anchor,
      parentId: reply?.id ?? null,
      content: text,
    };
    setBusy(true);
    setError(null);
    try {
      await canvasApi.comment(snapshot.document.id, {
        ...body,
        requestId: requests.forPayload(body),
      });
      const created = await canvasApi.commentDetail(
        snapshot.document.id,
        commentId,
      );
      setItems((current) => mergeComments(current, [created]));
      setText("");
      setReply(null);
      setAnchor(null);
      setCommentId(crypto.randomUUID());
      requests.clear();
      await store.getState().load();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };

  const activeCount = items.filter((item) => !item.deleted).length;

  return (
    <section className="flex flex-col gap-3 p-3.5" aria-label="画布评论">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[13px] font-medium">评审评论</h3>
        {activeCount > 0 && (
          <span className="rounded-[6px] bg-foreground/[0.05] px-1.5 py-0.5 text-[11px] tabular-nums text-muted-foreground dark:bg-white/[0.07]">
            {activeCount} 条
          </span>
        )}
      </div>
      {error && (
        <p
          role="alert"
          className="rounded-[10px] border border-destructive/20 bg-destructive/5 px-2.5 py-2 text-xs text-destructive"
        >
          {error}
        </p>
      )}
      {items.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-[12px] border border-dashed border-foreground/[0.1] bg-foreground/[0.02] px-4 py-6 text-center dark:border-white/[0.1] dark:bg-white/[0.02]">
          <span className="grid size-8 place-items-center rounded-[10px] bg-foreground/[0.05] text-muted-foreground dark:bg-white/[0.07]">
            <MessageSquare className="size-4" aria-hidden />
          </span>
          <p className="text-xs text-muted-foreground">
            暂无评论，可针对画布或指定节点留下修改意见。
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {items.map((comment) => {
            const nodeTitle = comment.nodeId
              ? (snapshot.graph.nodes.find((node) => node.id === comment.nodeId)
                ?.title ?? "节点已移除")
              : null;
            return (
              <article
                key={comment.id}
                className={cn(
                  "group flex flex-col gap-1.5 rounded-[12px] border border-foreground/[0.07] bg-background/70 p-3 dark:border-white/[0.08] dark:bg-white/[0.03]",
                  comment.parentId && "ml-4 border-l-2 border-l-primary/35",
                  comment.resolved && "opacity-70",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="min-w-0 truncate text-[11px] text-muted-foreground">
                    <span className="font-medium text-foreground/85">
                      {comment.createdBy === userId ? "我" : comment.authorName}
                    </span>
                    {" · "}
                    <span className="tabular-nums">
                      {canvasDate(comment.createdAt)}
                    </span>
                    {comment.resolved && (
                      <span className="ml-1.5 rounded-[5px] bg-emerald-500/12 px-1.5 py-0.5 text-[10px] text-emerald-700 dark:text-emerald-300">
                        已解决
                      </span>
                    )}
                  </p>
                  {editable && !comment.deleted && (
                    <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 [@media(pointer:coarse)]:opacity-100 motion-reduce:transition-none">
                      {!comment.parentId && (
                        <>
                          <CanvasIconButton
                            size="icon-xs"
                            label="回复"
                            disabled={busy}
                            onClick={() => {
                              setReply(comment);
                              setEditing(null);
                            }}
                          >
                            <Reply />
                          </CanvasIconButton>
                          <CanvasIconButton
                            size="icon-xs"
                            label={comment.resolved ? "重新打开" : "标记解决"}
                            disabled={busy}
                            onClick={() =>
                              void patch(comment, {
                                resolved: !comment.resolved,
                              }).catch(() => undefined)
                            }
                          >
                            <Check />
                          </CanvasIconButton>
                        </>
                      )}
                      {(comment.createdBy === userId ||
                        snapshot.permissions.canManage) && (
                          <>
                            <CanvasIconButton
                              size="icon-xs"
                              label="编辑评论"
                              disabled={busy}
                              onClick={() => {
                                setEditing(comment);
                                setText(comment.content);
                                setReply(null);
                              }}
                            >
                              <Pencil />
                            </CanvasIconButton>
                            <CanvasIconButton
                              variant="danger"
                              size="icon-xs"
                              label="删除评论"
                              disabled={busy}
                              onClick={() =>
                                void patch(comment, { deleted: true }).catch(
                                  () => undefined,
                                )
                              }
                            >
                              <Trash2 />
                            </CanvasIconButton>
                          </>
                        )}
                    </div>
                  )}
                </div>
                {comment.nodeId && (
                  <div>
                    <button
                      type="button"
                      className="inline-flex max-w-full items-center gap-1 rounded-[6px] bg-primary/[0.08] px-1.5 py-0.5 text-[11px] font-medium text-primary outline-none transition-colors hover:bg-primary/[0.14] focus-visible:ring-2 focus-visible:ring-primary/35"
                      onClick={() => store.getState().focus([comment.nodeId!])}
                    >
                      <LocateFixed className="size-3 shrink-0" aria-hidden />
                      <span className="truncate">
                        {nodeTitle}
                        {comment.resourceVersion !== null
                          ? ` · v${comment.resourceVersion}`
                          : ""}
                      </span>
                    </button>
                  </div>
                )}
                <p className="whitespace-pre-wrap break-words text-[13px] leading-relaxed text-foreground/90">
                  {comment.deleted ? "评论已删除" : comment.content}
                </p>
              </article>
            );
          })}
        </div>
      )}
      {more && (
        <CanvasButton
          variant="secondary"
          size="sm"
          disabled={loading}
          onClick={() => void loadMore()}
        >
          加载更多评论
        </CanvasButton>
      )}
      {editable && (
        <div className="flex flex-col gap-2 border-t border-foreground/[0.07] pt-3 dark:border-white/[0.08]">
          <p className="text-xs text-muted-foreground">
            {editing
              ? "编辑评论"
              : reply
                ? "回复评论"
                : anchor
                  ? `评论节点：${snapshot.graph.nodes.find((node) => node.id === anchor)?.title ?? "已移除节点"}`
                  : "发表画布评论"}
          </p>
          <div className="p-0.5">
            <CanvasTextarea
              name="canvas-comment-content"
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={3}
              className="min-h-20"
              placeholder="写下评审意见或修改建议…"
              maxLength={16000}
              aria-label="评论内容"
              disabled={busy}
            />
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {selected && !editing && !reply && (
              <CanvasButton
                variant="secondary"
                size="sm"
                disabled={busy}
                onClick={() => setAnchor(selected)}
              >
                关联选中节点
              </CanvasButton>
            )}
            {(editing || reply || anchor) && (
              <CanvasButton
                variant="ghost"
                size="sm"
                disabled={busy}
                onClick={() => {
                  setEditing(null);
                  setReply(null);
                  setAnchor(null);
                  setText("");
                }}
              >
                取消
              </CanvasButton>
            )}
            <CanvasButton
              variant="generate"
              size="sm"
              className="ml-auto"
              disabled={busy || !text.trim()}
              onClick={() => void save()}
            >
              {editing ? "保存评论" : "发表评论"}
            </CanvasButton>
          </div>
        </div>
      )}
    </section>
  );
}
