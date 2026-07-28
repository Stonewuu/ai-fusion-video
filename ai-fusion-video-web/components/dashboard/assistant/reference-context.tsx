"use client";

import { FolderKanban, PlugZap, Sparkles, X } from "lucide-react";
import type { ReactNode } from "react";
import type {
  AssistantMcpToolReferenceOption,
  AssistantSkillReferenceOption,
} from "@/lib/api/ai-assistant";
import type { AssistantProjectReference } from "./reference-types";

interface AssistantReferenceContextProps {
  project: AssistantProjectReference | null;
  skills: AssistantSkillReferenceOption[];
  mcpTools: AssistantMcpToolReferenceOption[];
  onRemoveProject: () => void;
  onRemoveSkill: (id: string) => void;
  onRemoveMcpTool: (serverName: string, toolName: string) => void;
}

interface ReferenceChipProps {
  label: string;
  title: string;
  icon: ReactNode;
  onRemove: () => void;
}

function ReferenceChip({ label, title, icon, onRemove }: ReferenceChipProps) {
  return (
    <span
      className="inline-flex h-7 max-w-52 items-center gap-1 rounded-full border border-border/30 bg-muted/70 pl-2 pr-1 text-[11px] text-foreground"
      title={title}
    >
      <span className="shrink-0 text-muted-foreground">{icon}</span>
      <span className="min-w-0 truncate">{label}</span>
      <button
        type="button"
        className="grid size-6 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-background/80 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
        aria-label={`移除 ${label}`}
        data-assistant-interactive="true"
        onClick={onRemove}
      >
        <X className="size-3" />
      </button>
    </span>
  );
}

export function AssistantReferenceContext({
  project,
  skills,
  mcpTools,
  onRemoveProject,
  onRemoveSkill,
  onRemoveMcpTool,
}: AssistantReferenceContextProps) {
  const hasReferences = !!project || skills.length > 0 || mcpTools.length > 0;
  if (!hasReferences) return null;

  return (
    <div className="flex min-h-9 flex-wrap items-center gap-1.5 border-b border-border/20 px-2 py-1.5">
      <span className="mr-0.5 text-[10px] font-medium text-muted-foreground">引用</span>
      {project ? (
        <ReferenceChip
          label={project.name}
          title={`项目 · ${project.name} · #${project.id}`}
          icon={<FolderKanban className="size-3.5" />}
          onRemove={onRemoveProject}
        />
      ) : null}
      {skills.map((skill) => (
        <ReferenceChip
          key={skill.id}
          label={skill.displayName}
          title={`Skill · ${skill.name} · ${skill.description}`}
          icon={<Sparkles className="size-3.5" />}
          onRemove={() => onRemoveSkill(skill.id)}
        />
      ))}
      {mcpTools.map((tool) => (
        <ReferenceChip
          key={`${tool.serverName}:${tool.toolName}`}
          label={tool.toolName}
          title={`MCP · ${tool.serverName} · ${tool.description || tool.toolName}`}
          icon={<PlugZap className="size-3.5" />}
          onRemove={() => onRemoveMcpTool(tool.serverName, tool.toolName)}
        />
      ))}
    </div>
  );
}
