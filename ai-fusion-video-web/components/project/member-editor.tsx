"use client";

import { UserMinus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CanvasSegmentedControl } from "@/components/canvas/workspace/segmented-control";
import type { ProjectMember } from "@/lib/api/project-members";

export interface ProjectMemberEditorState {
  member: ProjectMember;
  revision: number;
  role: string;
  removing: boolean;
}
export function ProjectMemberEditor({ editor, onChange, busy, stale, onReload }: {
  editor: ProjectMemberEditorState;
  onChange: (editor: ProjectMemberEditorState) => void;
  busy: boolean;
  stale: boolean;
  onReload: () => void;
}) {
  return (
    <div className="space-y-4">
      <div>
        <p className="text-sm font-semibold">设置 {editor.member.person.name} 的项目权限</p>
        <p className="text-xs text-muted-foreground">@{editor.member.person.username}</p>
      </div>
      <CanvasSegmentedControl
        label="项目权限"
        value={editor.role}
        options={[
          { value: "4", label: "查看", disabled: busy || editor.removing },
          { value: "3", label: "编辑", disabled: busy || editor.removing },
          { value: "2", label: "管理", disabled: busy || editor.removing },
        ]}
        onChange={(role) => onChange({ ...editor, role })}
        className="w-full"
      />
      <p className="text-sm text-muted-foreground">
        {editor.role === "4" ? "可查看项目、剧本、分镜和项目画布。" : editor.role === "3" ? "可修改项目内容、编辑画布并生成媒体。" : "可编辑和运行，还可管理项目成员与项目画布。"}
        单独设置的画布权限仍然有效；私人素材须有原资源授权。
      </p>
      {stale && (
        <div role="alert" className="space-y-2 rounded-lg border border-border/30 bg-muted/30 p-3">
          <p className="text-sm">项目成员权限已变化，请读取最新权限后重新选择。</p>
          <Button variant="outline" size="sm" disabled={busy} onClick={onReload}>读取最新权限</Button>
        </div>
      )}
      {editor.member.role !== null && !editor.removing && (
        <Button variant="destructive-ghost" disabled={busy} onClick={() => onChange({ ...editor, removing: true })}>
          <UserMinus />移除成员
        </Button>
      )}
      {editor.removing && (
        <div className="space-y-3 rounded-lg border border-destructive/30 bg-destructive/5 p-3">
          <p className="text-sm">将移除此账号的项目授权和本项目所有画布的单独授权。通过团队继承的访问仍由团队管理。</p>
          <Button variant="outline" size="sm" disabled={busy} onClick={() => onChange({ ...editor, removing: false })}>取消移除</Button>
        </div>
      )}
    </div>
  );
}
