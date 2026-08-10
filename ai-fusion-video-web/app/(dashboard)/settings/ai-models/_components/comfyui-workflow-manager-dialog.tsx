"use client";

import { useCallback, useEffect, useState } from "react";
import { ImageIcon, Loader2, Plus, Trash2, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useConfirm } from "@/components/ui/confirm-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { ApiConfig } from "@/lib/api/ai-model";
import { toastApiError } from "@/lib/api/toast-api-error";
import { comfyUiWorkflowApi, type ComfyUiWorkflow } from "@/lib/api/comfyui-workflow";
import { cn } from "@/lib/utils";
import { ComfyUiWorkflowEditor } from "./comfyui-workflow-editor";

interface ComfyUiWorkflowManagerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  apiConfig: ApiConfig | null;
  onChanged: () => void;
}

export function ComfyUiWorkflowManagerDialog({
  open,
  onOpenChange,
  apiConfig,
  onChanged,
}: ComfyUiWorkflowManagerDialogProps) {
  const { confirm } = useConfirm();
  const [loading, setLoading] = useState(false);
  const [workflows, setWorkflows] = useState<ComfyUiWorkflow[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [creating, setCreating] = useState(false);
  const selectedWorkflow = workflows.find(workflow => workflow.id === selectedId) || null;

  const loadWorkflows = useCallback(async (preferredId?: number) => {
    if (!apiConfig) {
      setWorkflows([]);
      setSelectedId(null);
      return;
    }
    setLoading(true);
    try {
      const result = await comfyUiWorkflowApi.page(1, 100, apiConfig.id);
      setWorkflows(result.list);
      setSelectedId(current => {
        const target = preferredId ?? current;
        return result.list.some(item => item.id === target) ? target : result.list[0]?.id ?? null;
      });
    } catch (error) {
      toastApiError(error, "加载 ComfyUI 工作流失败");
    } finally {
      setLoading(false);
    }
  }, [apiConfig]);

  useEffect(() => {
    if (!open) return;
    setCreating(false);
    setSelectedId(null);
    void loadWorkflows();
  }, [apiConfig?.id, loadWorkflows, open]);

  const deleteWorkflow = async (workflow: ComfyUiWorkflow) => {
    const accepted = await confirm({
      title: "删除 ComfyUI 工作流",
      description: `确定删除“${workflow.name}”吗？已绑定模型或存在不可删除版本时服务端会拒绝该操作。`,
      confirmText: "删除",
      variant: "destructive",
    });
    if (!accepted) return;
    try {
      await comfyUiWorkflowApi.delete(workflow.id);
      await loadWorkflows();
      onChanged();
    } catch (error) {
      toastApiError(error, "删除 ComfyUI 工作流失败");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[min(96vw,1400px)] h-[min(900px,calc(100vh-2rem))] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>{apiConfig?.name || "ComfyUI"} · 工作流</DialogTitle>
          <DialogDescription>通过导入 ComfyUI 的 API-JSON，并且完成输入输出绑定、在线校验和试运行，即可将工作流配置到模型中，在添加模型的时候关联已发布的工作流即可。</DialogDescription>
        </DialogHeader>

        <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="flex min-h-0 flex-col rounded-xl border border-border/30 bg-card/50 p-3 backdrop-blur-sm">
            <div className="flex shrink-0 items-center justify-between gap-2">
              <div>
                <p className="text-sm font-semibold">工作流目录</p>
                <p className="mt-1 text-xs text-muted-foreground">{workflows.length} 个工作流</p>
              </div>
              <Button
                variant="outline"
                size="icon-sm"
                aria-label="新建 ComfyUI 工作流"
                onClick={() => { setCreating(true); setSelectedId(null); }}
                disabled={!apiConfig}
              >
                <Plus />
              </Button>
            </div>

            <div className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
              {loading ? (
                <div className="flex justify-center py-8"><Loader2 className="size-5 animate-spin text-muted-foreground" /></div>
              ) : workflows.length === 0 && !creating ? (
                <div className="rounded-lg border border-border/20 bg-background/70 p-3 text-xs leading-5 text-muted-foreground">还没有工作流。点击右上角加号开始导入。</div>
              ) : workflows.map(workflow => {
                const selected = workflow.id === selectedId && !creating;
                return (
                  <div key={workflow.id} className={cn("group flex items-center gap-1 rounded-lg border p-1", selected ? "border-primary/40 bg-primary/8" : "border-transparent hover:bg-muted/40")}>
                    <button
                      type="button"
                      onClick={() => { setCreating(false); setSelectedId(workflow.id); }}
                      className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-muted/60">
                        {workflow.modelType === 2 ? <ImageIcon className="size-4" /> : <Video className="size-4" />}
                      </span>
                      <span className="min-w-0">
                        <span className="block truncate text-xs font-medium">{workflow.name}</span>
                        <span className="mt-0.5 block truncate font-mono text-[10px] text-muted-foreground">{workflow.code}</span>
                      </span>
                    </button>
                    <Button variant="destructive-ghost" size="icon-xs" aria-label={`删除 ${workflow.name}`} onClick={() => void deleteWorkflow(workflow)}>
                      <Trash2 />
                    </Button>
                  </div>
                );
              })}
            </div>
          </aside>

          <section className="min-h-0 overflow-hidden rounded-xl border border-border/30 bg-card/50 p-4 backdrop-blur-sm">
            {apiConfig && (creating || selectedWorkflow) ? (
              <ComfyUiWorkflowEditor
                key={`${apiConfig.id}:${creating ? "new" : selectedWorkflow?.id}`}
                workflow={creating ? null : selectedWorkflow}
                apiConfig={apiConfig}
                onWorkflowCreated={async id => {
                  setCreating(false);
                  await loadWorkflows(id);
                  onChanged();
                }}
                onWorkflowChanged={async () => {
                  await loadWorkflows(selectedId ?? undefined);
                  onChanged();
                }}
              />
            ) : (
              <div className="grid h-full place-items-center text-center">
                <div><p className="text-sm font-medium">选择一个工作流</p><p className="mt-1 text-xs text-muted-foreground">或新建工作流，开始六步配置。</p></div>
              </div>
            )}
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
