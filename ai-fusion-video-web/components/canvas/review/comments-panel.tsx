"use client";

import { useEffect, useState } from "react";
import { MessageSquare, Reply, Check, Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { canvasDate } from "@/lib/canvas/labels";
import type { CanvasComment } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

function mergeComments(current: CanvasComment[], incoming: CanvasComment[]) {
  const rows = new Map(current.map((item) => [item.id, item]));
  incoming.forEach((item) => {
    if (!rows.has(item.id) || rows.get(item.id)!.version <= item.version)
      rows.set(item.id, item);
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
        if (active) setError(getApiErrorMessage(failure));
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
    if (!text.trim()) return;
    if (editing) {
      try {
        await patch(editing, { content: text });
        setEditing(null);
        setText("");
      } catch {}
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
  return (
    <section className="space-y-4 p-4" aria-label="画布评论">
      <div className="flex items-center gap-2">
        <MessageSquare className="size-4" />
        <h3 className="text-sm font-semibold">评审评论</h3>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <div className="space-y-4">
        {items.map((comment) => (
          <article
            key={comment.id}
            className={
              comment.parentId
                ? "space-y-2 border-l border-border/30 pl-4"
                : "space-y-2 border-b border-border/20 pb-4"
            }
          >
            <p className="text-xs text-muted-foreground">
              {comment.createdBy === userId ? "我" : comment.authorName} ·{" "}
              {canvasDate(comment.createdAt)}
              {comment.resolved ? " · 已解决" : ""}
            </p>
            {comment.nodeId && (
              <Button
                variant="link"
                size="xs"
                onClick={() => store.getState().focus([comment.nodeId!])}
              >
                {snapshot.graph.nodes.find((node) => node.id === comment.nodeId)
                  ?.title ?? "节点已移除"}
                {comment.resourceVersion !== null
                  ? ` · v${comment.resourceVersion}`
                  : ""}
              </Button>
            )}
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {comment.deleted ? "评论已删除" : comment.content}
            </p>
            {editable && !comment.deleted && (
              <div className="flex gap-2">
                {!comment.parentId && (
                  <>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      title="回复"
                      disabled={busy}
                      onClick={() => {
                        setReply(comment);
                        setEditing(null);
                      }}
                    >
                      <Reply />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      title={comment.resolved ? "重新打开" : "标记解决"}
                      disabled={busy}
                      onClick={() =>
                        void patch(comment, {
                          resolved: !comment.resolved,
                        }).catch(() => undefined)
                      }
                    >
                      <Check />
                    </Button>
                  </>
                )}
                {(comment.createdBy === userId ||
                  snapshot.permissions.canManage) && (
                  <>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      title="编辑评论"
                      disabled={busy}
                      onClick={() => {
                        setEditing(comment);
                        setText(comment.content);
                        setReply(null);
                      }}
                    >
                      <Pencil />
                    </Button>
                    <Button
                      variant="destructive-ghost"
                      size="icon-xs"
                      title="删除评论"
                      disabled={busy}
                      onClick={() =>
                        void patch(comment, { deleted: true }).catch(
                          () => undefined,
                        )
                      }
                    >
                      <Trash2 />
                    </Button>
                  </>
                )}
              </div>
            )}
          </article>
        ))}
      </div>
      {more && (
        <Button
          variant="outline"
          size="sm"
          disabled={loading}
          onClick={() => void loadMore()}
        >
          加载更多评论
        </Button>
      )}
      {editable && (
        <div className="space-y-3 border-t border-border/20 pt-4">
          <p className="text-xs text-muted-foreground">
            {editing
              ? "编辑评论"
              : reply
                ? "回复评论"
                : anchor
                  ? `评论节点：${snapshot.graph.nodes.find((node) => node.id === anchor)?.title ?? "已移除节点"}`
                  : "评论画布"}
          </p>
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={4}
            maxLength={16000}
            aria-label="评论内容"
            disabled={busy}
          />
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              disabled={busy || !text.trim()}
              onClick={() => void save()}
            >
              {editing ? "保存评论" : "发表评论"}
            </Button>
            {selected && !editing && !reply && (
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={() => setAnchor(selected)}
              >
                针对选中节点
              </Button>
            )}
            {(editing || reply || anchor) && (
              <Button
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
              </Button>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
