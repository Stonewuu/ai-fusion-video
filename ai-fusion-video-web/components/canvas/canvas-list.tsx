"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Plus, LayoutDashboard, ArrowRight, RefreshCw, FileUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { CanvasToolbarToggle } from "./workspace/toolbar-toggle";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { canvasLocation } from "@/lib/canvas/location";
import { canvasDate } from "@/lib/canvas/labels";
import type { CanvasDocument } from "@/lib/canvas/types";
import { CanvasCreateDialog } from "./canvas-create-dialog";
import { CanvasGuidance } from "./shared/canvas-guidance";
import { CanvasTemplateImportDialog } from "./templates/template-import-dialog";

export function CanvasList({ projectId }: { projectId?: number }) {
  const router = useRouter();
  const [documents, setDocuments] = useState<(CanvasDocument & { projectName: string | null })[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [archived, setArchived] = useState(false);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [importing, setImporting] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const visibleDocuments = documents;
  const searchFor = (value: string) => { setSearch(value); setPage(1); setLoading(true); setError(null); };
  const refresh = () => { setLoading(true); setError(null); setAttempt((value) => value + 1); };
  useEffect(() => {
    const controller = new AbortController();
    void canvasApi
      .list(
        { projectId, pageNo: page, pageSize: 30, archived, search: search.trim() },
        controller.signal,
      )
      .then((result) => {
        if (!controller.signal.aborted) {
          setDocuments(result.list);
          setTotal(result.total);
          setError(null);
        }
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setError(getApiErrorMessage(failure));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [projectId, page, archived, attempt, search]);
  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 px-4 pb-20 md:pb-3">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold">
            {projectId ? "项目画布" : "创作画布"}
          </h1>
          <p className="text-sm text-muted-foreground">
            组织创意、剧本、分镜与媒体，让创作过程清晰可见。
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
        <Button variant="outline" onClick={() => setImporting(true)}><FileUp />导入模板</Button>
        <Button onClick={() => setCreating(true)}>
          <Plus />
          新建画布
        </Button>
        </div>
      </header>
      <div className="flex flex-wrap items-center gap-2">
        <Input
          className="max-w-sm"
          placeholder="搜索画布或项目名称"
          aria-label="搜索画布或项目名称"
          maxLength={200}
          value={search}
          onChange={(event) => searchFor(event.target.value)}
        />
        <CanvasToolbarToggle
          label="已归档"
          checked={archived}
          onCheckedChange={(checked) => {
            setArchived(checked);
            setPage(1);
            setLoading(true);
          }}
        />
        <Button
          variant="ghost"
          size="icon"
          title="刷新画布列表"
          disabled={loading}
          onClick={refresh}
        >
          <RefreshCw />
        </Button>
      </div>
      {error ? (
        <CanvasGuidance title="暂时无法读取画布" tone="error" actions={<Button variant="outline" size="sm" onClick={refresh}><RefreshCw />重新加载</Button>}>
          {error} 加载失败不代表画布已被删除。
        </CanvasGuidance>
      ) : loading ? (
        <p role="status" className="py-12 text-center text-muted-foreground">
          正在读取画布…
        </p>
      ) : (
        <div className="overflow-hidden rounded-xl border border-border/30 bg-card/50">
          {visibleDocuments
            .map((item) => (
              <div
                key={item.id}
                className="flex items-center gap-4 border-b border-border/20 p-4 last:border-b-0"
              >
                <LayoutDashboard className="size-5 shrink-0 text-primary" />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">{item.title}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {item.projectId ? item.projectName : "个人画布"} ·
                    更新于 {canvasDate(item.updateTime)}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  aria-label={`打开${item.title}`}
                  onClick={() => router.push(canvasLocation(item))}
                >
                  打开
                  <ArrowRight />
                </Button>
              </div>
            ))}
          {!visibleDocuments.length && (
            <div className="space-y-3 p-12 text-center">
              <p className="text-sm text-muted-foreground">
                {search.trim() ? "没有找到匹配的画布" : archived ? "暂无归档画布" : "还没有画布"}
              </p>
              {search.trim() ? (
                <Button variant="outline" onClick={() => searchFor("")}>清空搜索</Button>
              ) : !archived && (
                <Button variant="outline" onClick={() => setCreating(true)}>
                  创建第一张画布
                </Button>
              )}
            </div>
          )}
        </div>
      )}
      {!error && total > 30 && (
        <div className="flex items-center justify-end gap-2">
          <span className="text-sm text-muted-foreground">
            第 {page} 页 · 共 {total} 张
          </span>
          <Button
            variant="outline"
            disabled={page === 1}
            onClick={() => {
              setLoading(true);
              setPage(page - 1);
            }}
          >
            上一页
          </Button>
          <Button
            variant="outline"
            disabled={page * 30 >= total}
            onClick={() => {
              setLoading(true);
              setPage(page + 1);
            }}
          >
            下一页
          </Button>
        </div>
      )}
      {creating && (
        <CanvasCreateDialog
          projectId={projectId}
          onClose={() => setCreating(false)}
          onCreated={(document) => router.push(canvasLocation(document))}
        />
      )}
      {importing && <CanvasTemplateImportDialog onClose={() => setImporting(false)} />}
    </div>
  );
}
