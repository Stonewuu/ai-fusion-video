"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { canvasApi } from "@/lib/api/canvas";
import { projectApi, type Project } from "@/lib/api/project";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasDocument } from "@/lib/canvas/types";
import { CreateProjectDialog } from "@/components/dashboard/create-project-dialog";
import { CanvasSelect } from "./forms/canvas-select";
import { CanvasDialogContent as DialogContent } from "./shared/canvas-dialog";
import { CanvasGuidance } from "./shared/canvas-guidance";

export function CanvasCreateDialog({
  projectId,
  onClose,
  onCreated,
}: {
  projectId?: number;
  onClose: () => void;
  onCreated: (document: CanvasDocument) => void;
}) {
  const [command, setCommand] = useState<{
    id: string;
    title: string;
    projectId: number | null;
  } | null>(null);
  const [title, setTitle] = useState("");
  const [scope, setScope] = useState(
    projectId ? String(projectId) : "personal",
  );
  const [projects, setProjects] = useState<Project[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [projectCreating, setProjectCreating] = useState(false);
  const loadProjects = useCallback(() => {
    void projectApi
      .list()
      .then(setProjects)
      .catch((failure) => setError(getApiErrorMessage(failure)));
  }, []);
  useEffect(loadProjects, [loadProjects]);
  const create = async () => {
    const submitted = command ?? {
      id: crypto.randomUUID(),
      title: title.trim(),
      projectId: scope === "personal" ? null : Number(scope),
    };
    setCommand(submitted);
    setBusy(true);
    setError(null);
    try {
      onCreated(
        await canvasApi.create(
          submitted.id,
          submitted.title,
          submitted.projectId,
        ),
      );
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <Dialog
        open={!projectCreating}
        onOpenChange={(open) => {
          if (!open && !busy) onClose();
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>新建画布</DialogTitle>
            <DialogDescription>
              为这次创作起个名字，再选择是否关联项目。
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="flex flex-col gap-2">
              <label htmlFor="canvas-title" className="text-sm font-medium">
                画布名称
              </label>
              <Input
                id="canvas-title"
                autoFocus
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                maxLength={200}
                placeholder="例如：第一集视觉探索"
                disabled={busy || !!command}
              />
            </div>
            <div className="flex flex-col gap-2">
            <p className="text-sm font-medium">所属项目</p>
            <CanvasSelect
              label="所属项目"
              value={scope}
              placeholder="选择画布归属"
              disabled={busy || !!command || projectId !== undefined}
              onChange={setScope}
              items={[
                { value: "personal", label: "个人画布" },
                ...projects.map((project) => ({
                  value: String(project.id),
                  label: project.name,
                })),
              ]}
            />
            </div>
            <CanvasGuidance title={scope === "personal" ? "自由整理想法与素材" : "与项目内容保持同步"}>
              {scope === "personal" ? "个人画布可以编写文字、生成图片和视频。需要编辑剧本或分镜时，请选择一个项目。" : "可以直接添加此项目的剧本、分镜和资产；在画布中的内容修改会同步到项目。"}
            </CanvasGuidance>
            {projectId === undefined && (
              <Button
                variant="ghost"
                size="sm"
              disabled={busy || !!command}
                onClick={() => setProjectCreating(true)}
              >
                创建新项目
              </Button>
            )}
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" disabled={busy} onClick={onClose}>
              取消
            </Button>
            <Button
              disabled={busy || !title.trim()}
              onClick={() => void create()}
            >
              {busy ? "正在创建…" : command ? "查看创建结果" : "创建画布"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <CreateProjectDialog
        open={projectCreating}
        onClose={() => setProjectCreating(false)}
        onCreated={(project) => {
          setProjectCreating(false);
          setProjects((current) => [
            ...current.filter((item) => item.id !== project.id),
            project,
          ]);
          setScope(String(project.id));
        }}
      />
    </>
  );
}
