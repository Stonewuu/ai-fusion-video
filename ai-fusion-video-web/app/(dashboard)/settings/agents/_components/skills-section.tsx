"use client";

import { useConfirm } from "@/components/ui/confirm-dialog";
import { useState } from "react";
import { BookOpen, Loader2, Pencil, Plus, Trash2, Upload } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import { Textarea } from "@/components/ui/textarea";
import { agentConfigApi, type AgentUserSkill } from "@/lib/api/agent-config";
import { SkillImportDialog } from "./skills/skill-import-dialog";
import { settingsTypography } from "../../_shared";

interface SkillsSectionProps {
  skills: AgentUserSkill[];
  onRefresh: () => Promise<void>;
}

const EMPTY_FORM = { name: "", displayName: "", description: "", content: "" };

export function SkillsSection({ skills, onRefresh }: SkillsSectionProps) {
  const { confirm } = useConfirm();
  const [open, setOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [editing, setEditing] = useState<AgentUserSkill | null>(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);

  const showEditor = (skill?: AgentUserSkill) => {
    setEditing(skill || null);
    setForm(skill ? {
      name: skill.name,
      displayName: skill.displayName === null ? "" : skill.displayName,
      description: skill.description,
      content: skill.content,
    } : EMPTY_FORM);
    setOpen(true);
  };

  const save = async () => {
    if (!form.name.trim() || !form.displayName.trim() || !form.description.trim() || !form.content.trim()) {
      toast.error("请完整填写 Skill 调用名称、显示名称、描述和内容");
      return;
    }
    setSaving(true);
    try {
      await agentConfigApi.saveSkill({
        originalName: editing?.name,
        name: form.name,
        displayName: form.displayName,
        description: form.description,
        content: form.content,
      });
      toast.success(editing ? "Skill 已更新" : "Skill 已创建");
      setOpen(false);
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "保存 Skill 失败");
    } finally {
      setSaving(false);
    }
  };

  const remove = async (skill: AgentUserSkill) => {
    const ok = await confirm({ title: "删除 Skill", description: `确认删除 Skill「${skill.name}」？`, variant: "destructive", confirmText: "确定删除" }); if (!ok) return;
    setDeleting(skill.name);
    try {
      await agentConfigApi.deleteSkill(skill.name);
      toast.success("Skill 已删除");
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "删除 Skill 失败");
    } finally {
      setDeleting(null);
    }
  };

  return (
    <section className="flex flex-col rounded-xl border border-border/30 bg-card/50 p-4 backdrop-blur-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <div className="rounded-xl bg-primary/10 p-2 text-primary">
            <BookOpen className="size-5" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className={settingsTypography.sectionTitle}>我的 Skills</h2>
              <Badge variant="secondary">{skills.length}</Badge>
            </div>
            <p className={settingsTypography.sectionDescription}>保存个人工作规范，在助手输入框中输入 / 即可主动引用。</p>
          </div>
        </div>
        <div className="flex shrink-0 gap-2">
          <Button variant="outline" onClick={() => setImportOpen(true)}><Upload />导入 Skill</Button>
          <Button onClick={() => showEditor()}><Plus />新建 Skill</Button>
        </div>
      </div>

      <div className="mt-4 flex-1 overflow-hidden rounded-lg border border-border/20 bg-background/70">
        {skills.length === 0 ? (
          <div className="flex min-h-32 flex-col items-center justify-center p-6 text-center">
            <BookOpen className="size-5 text-muted-foreground" />
            <p className="mt-2 text-sm font-medium">还没有自定义 Skill</p>
            <p className="mt-1 text-xs text-muted-foreground">创建后可通过 / 快速选择并加入当前对话。</p>
          </div>
        ) : skills.map((skill) => (
          <div
            key={skill.id}
            className="flex items-center justify-between gap-3 border-b border-border/20 p-3 transition-colors last:border-b-0 hover:bg-muted/40"
          >
            <div className="min-w-0">
              <div className="flex min-w-0 items-center gap-2">
                {skill.displayName === null ? (
                  <span className="truncate text-sm font-medium text-destructive">未设置显示名称</span>
                ) : (
                  <span className="truncate text-sm font-medium">{skill.displayName}</span>
                )}
                <Badge variant="outline">SKILL.md</Badge>
              </div>
              <div className="mt-1 truncate font-mono text-xs text-muted-foreground">{skill.name}</div>
              <div className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">{skill.description}</div>
            </div>
            <div className="flex shrink-0 gap-2">
              <Button variant="ghost" size="icon-sm" aria-label={`编辑 ${skill.name}`} title="编辑" onClick={() => showEditor(skill)}>
                <Pencil />
              </Button>
              <Button
                variant="destructive-ghost"
                size="icon-sm"
                aria-label={`删除 ${skill.name}`}
                title="删除"
                onClick={() => remove(skill)}
                disabled={deleting === skill.name}
              >
                {deleting === skill.name ? <Loader2 className="animate-spin" /> : <Trash2 />}
              </Button>
            </div>
          </div>
        ))}
      </div>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden rounded-2xl sm:max-w-2xl">
          <DialogHeader className="shrink-0 pr-8">
            <DialogTitle>{editing ? "编辑 Skill" : "新建 Skill"}</DialogTitle>
            <DialogDescription>显示名称面向用户，调用名称用于 AgentScope；正文写清触发条件、步骤、限制和输出要求。</DialogDescription>
          </DialogHeader>
          <div className="-mx-1 min-h-0 overflow-y-auto px-1 py-1">
            <div className="space-y-5">
              <div className="space-y-1.5">
                <Label htmlFor="skill-display-name" className={settingsTypography.fieldLabel}>显示名称</Label>
                <Input
                  id="skill-display-name"
                  value={form.displayName}
                  onChange={(event) => setForm((value) => ({ ...value, displayName: event.target.value }))}
                  placeholder="分镜连续性检查"
                />
                <p className="text-xs text-muted-foreground">显示在引用选择器、输入区标签和历史消息中。</p>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="skill-name" className={settingsTypography.fieldLabel}>调用名称</Label>
                <Input
                  id="skill-name"
                  value={form.name}
                  onChange={(event) => setForm((value) => ({ ...value, name: event.target.value }))}
                  placeholder="storyboard-review"
                />
                <p className="text-xs text-muted-foreground">使用小写字母、数字或短横线，不能包含连续短横线，最长 64 位。</p>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="skill-description" className={settingsTypography.fieldLabel}>描述</Label>
                <Input
                  id="skill-description"
                  value={form.description}
                  onChange={(event) => setForm((value) => ({ ...value, description: event.target.value }))}
                  placeholder="检查分镜连续性并给出修改建议"
                />
                <p className="text-xs text-muted-foreground">用一句话说明何时应该选择这个 Skill。</p>
              </div>
              <div className="space-y-1.5 border-t border-border/20 pt-5">
                <div>
                  <Label htmlFor="skill-content" className={settingsTypography.fieldLabel}>SKILL.md 正文</Label>
                  <p className="mt-1 text-xs text-muted-foreground">建议包含工作步骤、限制条件和明确的输出格式。</p>
                </div>
                <Textarea
                  id="skill-content"
                  value={form.content}
                  onChange={(event) => setForm((value) => ({ ...value, content: event.target.value }))}
                  className="min-h-72 font-mono text-sm"
                  placeholder={"# 工作方式\n\n1. 读取项目上下文\n2. 检查镜头衔接\n3. 输出问题与建议"}
                />
              </div>
            </div>
          </div>
          <DialogFooter className="shrink-0 border-t border-border/20 pt-4">
            <Button variant="outline" onClick={() => setOpen(false)}>取消</Button>
            <Button onClick={save} disabled={saving}>
              {saving && <Loader2 className="animate-spin" />}
              保存
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <SkillImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImported={onRefresh}
      />
    </section>
  );
}
