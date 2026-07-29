"use client";

import { useRef, useState } from "react";
import {
  AlertTriangle,
  Archive,
  FileText,
  FolderOpen,
  Loader2,
  PackageOpen,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  agentConfigApi,
  type AgentSkillImportCandidate,
  type AgentSkillImportPreview,
  type AgentSkillImportSelection,
} from "@/lib/api/agent-config";

interface SkillImportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: () => Promise<void>;
}

interface ImportSource {
  files: File[];
  paths: string[];
  label: string;
}

interface CandidateDraft {
  selected: boolean;
  displayName: string;
}

const DIRECTORY_INPUT_ATTRIBUTES = {
  webkitdirectory: "",
  directory: "",
} as const;

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function SkillImportDialog({
  open,
  onOpenChange,
  onImported,
}: SkillImportDialogProps) {
  const zipInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [source, setSource] = useState<ImportSource | null>(null);
  const [preview, setPreview] = useState<AgentSkillImportPreview | null>(null);
  const [drafts, setDrafts] = useState<Record<string, CandidateDraft>>({});
  const [previewing, setPreviewing] = useState(false);
  const [importing, setImporting] = useState(false);

  const reset = () => {
    setSource(null);
    setPreview(null);
    setDrafts({});
    setPreviewing(false);
    setImporting(false);
    if (zipInputRef.current) zipInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
  };

  const changeOpen = (nextOpen: boolean) => {
    if (!nextOpen && importing) return;
    if (!nextOpen && !importing) reset();
    onOpenChange(nextOpen);
  };

  const inspect = async (nextSource: ImportSource) => {
    setSource(nextSource);
    setPreview(null);
    setDrafts({});
    setPreviewing(true);
    try {
      const result = await agentConfigApi.previewSkillImport(
        nextSource.files,
        nextSource.paths,
      );
      setPreview(result);
      setDrafts(Object.fromEntries(result.skills.map((skill) => [
        skill.rootPath,
        {
          selected: skill.valid && !skill.exists,
          displayName: skill.suggestedDisplayName,
        },
      ])));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "预检 Skill 导入包失败");
      setSource(null);
    } finally {
      setPreviewing(false);
    }
  };

  const inspectZip = (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".zip")) {
      toast.error("请选择 .zip 文件");
      return;
    }
    void inspect({ files: [file], paths: [file.name], label: file.name });
  };

  const inspectFolder = (files: FileList | null) => {
    if (!files?.length) return;
    const selected = Array.from(files);
    const paths = selected.map((file) => file.webkitRelativePath || file.name);
    const root = paths[0]?.split("/")[0] || "Skill 目录";
    void inspect({ files: selected, paths, label: root });
  };

  const updateDraft = (rootPath: string, value: Partial<CandidateDraft>) => {
    setDrafts((current) => ({
      ...current,
      [rootPath]: { ...current[rootPath], ...value },
    }));
  };

  const selections = (): AgentSkillImportSelection[] => (
    preview?.skills.map((skill) => {
      const draft = drafts[skill.rootPath];
      return {
        rootPath: skill.rootPath,
        displayName: draft?.displayName.trim() || skill.suggestedDisplayName,
        action: !draft?.selected ? "SKIP" : skill.exists ? "REPLACE" : "CREATE",
      };
    }) ?? []
  );

  const canImport = Boolean(
    source
    && preview
    && preview.errors.length === 0
    && preview.skills.some((skill) => skill.valid && drafts[skill.rootPath]?.selected)
    && preview.skills.every((skill) => (
      !drafts[skill.rootPath]?.selected || drafts[skill.rootPath]?.displayName.trim()
    )),
  );

  const submit = async () => {
    if (!source || !preview || !canImport) return;
    setImporting(true);
    try {
      const result = await agentConfigApi.importSkills(
        source.files,
        source.paths,
        selections(),
      );
      const parts = [
        result.createdCount ? `新建 ${result.createdCount} 个` : "",
        result.replacedCount ? `覆盖 ${result.replacedCount} 个` : "",
        result.skippedCount ? `跳过 ${result.skippedCount} 个` : "",
      ].filter(Boolean);
      toast.success(`Skill 导入完成：${parts.join("，")}`);
      try {
        await onImported();
      } catch (refreshError) {
        toast.error(refreshError instanceof Error
          ? `Skill 已导入，但刷新列表失败：${refreshError.message}`
          : "Skill 已导入，但刷新列表失败");
      }
      reset();
      onOpenChange(false);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "导入 Skill 失败");
    } finally {
      setImporting(false);
    }
  };

  const renderCandidate = (skill: AgentSkillImportCandidate) => {
    const draft = drafts[skill.rootPath];
    const selected = draft?.selected ?? false;
    return (
      <div
        key={skill.rootPath}
        className="rounded-lg border border-border/20 bg-background/70 p-3"
      >
        <div className="flex items-start gap-3">
          <Checkbox
            id={`import-${skill.rootPath}`}
            checked={selected}
            disabled={!skill.valid || importing}
            onCheckedChange={(checked) => updateDraft(skill.rootPath, { selected: checked })}
          />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Label htmlFor={`import-${skill.rootPath}`} className="font-medium">
                {skill.name || skill.rootPath}
              </Label>
              {skill.exists && <Badge variant="destructive">已存在</Badge>}
              {!skill.valid && <Badge variant="destructive">校验失败</Badge>}
              {skill.hasScripts && <Badge variant="outline">含脚本</Badge>}
              {skill.hasReferences && <Badge variant="outline">引用</Badge>}
              {skill.hasAssets && <Badge variant="outline">资源</Badge>}
            </div>
            <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
              {skill.rootPath}
            </p>
            {skill.description && (
              <p className="mt-2 text-sm text-muted-foreground">{skill.description}</p>
            )}
            <p className="mt-2 text-xs text-muted-foreground">
              {skill.fileCount} 个文件 · {formatBytes(skill.totalBytes)}
            </p>

            {skill.valid && (
              <div className="mt-3 space-y-1.5">
                <Label htmlFor={`display-${skill.rootPath}`}>显示名称</Label>
                <Input
                  id={`display-${skill.rootPath}`}
                  value={draft?.displayName ?? skill.suggestedDisplayName}
                  disabled={!selected || importing}
                  onChange={(event) => updateDraft(skill.rootPath, {
                    displayName: event.target.value,
                  })}
                />
                {skill.exists && selected && (
                  <p className="text-xs text-destructive">将完整替换已有 Skill 目录。</p>
                )}
              </div>
            )}

            {skill.errors.length > 0 && (
              <ul className="mt-3 space-y-1 text-xs text-destructive">
                {skill.errors.map((error) => <li key={error}>· {error}</li>)}
              </ul>
            )}
            {skill.warnings.length > 0 && (
              <ul className="mt-3 space-y-1 text-xs text-amber-600 dark:text-amber-400">
                {skill.warnings.map((warning) => <li key={warning}>· {warning}</li>)}
              </ul>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogContent className="max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden rounded-2xl sm:max-w-3xl">
        <DialogHeader className="shrink-0 pr-8">
          <DialogTitle>导入 Skill</DialogTitle>
          <DialogDescription>
            支持一个或多个目录级 Agent Skill，每个 Skill 都必须包含 SKILL.md。
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 overflow-y-auto py-1">
          <div className="space-y-4">
            <div
              className="rounded-lg border-2 border-dashed border-border/40 bg-background/70 p-5 text-center"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                inspectZip(event.dataTransfer.files);
              }}
            >
              <PackageOpen className="mx-auto size-6 text-muted-foreground" />
              <p className="mt-2 text-sm font-medium">拖入 ZIP，或从本机选择</p>
              <p className="mt-1 text-xs text-muted-foreground">
                上传与解压后均不超过 30 MB，最多 1000 个文件。
              </p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                <Button variant="outline" onClick={() => zipInputRef.current?.click()} disabled={previewing || importing}>
                  <Archive />选择 ZIP
                </Button>
                <Button variant="outline" onClick={() => folderInputRef.current?.click()} disabled={previewing || importing}>
                  <FolderOpen />选择目录
                </Button>
              </div>
              <input
                ref={zipInputRef}
                type="file"
                accept=".zip,application/zip"
                className="hidden"
                onChange={(event) => inspectZip(event.target.files)}
              />
              <input
                ref={folderInputRef}
                type="file"
                multiple
                className="hidden"
                {...DIRECTORY_INPUT_ATTRIBUTES}
                onChange={(event) => inspectFolder(event.target.files)}
              />
            </div>

            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-muted-foreground">
              <div className="flex items-start gap-2">
                <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" />
                <p>Skill 暂不支持Shell脚本或授予额外工具权限。</p>
              </div>
            </div>

            {previewing && (
              <div className="flex min-h-32 items-center justify-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="animate-spin" />正在预检 {source?.label || "导入包"}…
              </div>
            )}

            {preview && !previewing && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <FileText className="size-4 text-muted-foreground" />
                    <span className="truncate text-sm font-medium">{source?.label}</span>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {preview.skills.length} 个 Skill · {preview.totalFiles} 个文件 · {formatBytes(preview.totalBytes)}
                  </span>
                </div>

                {preview.errors.length > 0 && (
                  <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3">
                    <ul className="space-y-1 text-sm text-destructive">
                      {preview.errors.map((error) => <li key={error}>· {error}</li>)}
                    </ul>
                  </div>
                )}
                {preview.warnings.length > 0 && (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
                    <ul className="space-y-1 text-xs text-amber-600 dark:text-amber-400">
                      {preview.warnings.map((warning) => <li key={warning}>· {warning}</li>)}
                    </ul>
                  </div>
                )}
                <div className="space-y-3">{preview.skills.map(renderCandidate)}</div>
              </div>
            )}
          </div>
        </div>

        <DialogFooter className="shrink-0 border-t border-border/20 pt-4">
          <Button variant="outline" onClick={() => changeOpen(false)} disabled={importing}>取消</Button>
          <Button onClick={submit} disabled={!canImport || importing}>
            {importing && <Loader2 className="animate-spin" />}
            导入所选 Skill
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
