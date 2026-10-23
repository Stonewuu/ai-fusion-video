"use client";
import { useState } from "react";
import {
  Archive,
  ArchiveRestore,
  Download,
  Ellipsis,
  Pencil,
  Users,
  FolderGit2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";

export function CanvasDocumentActions({
  editable,
  canManage,
  archived,
  busy,
  onRename,
  onArchive,
  onShare,
  onExport,
  onFork,
}: {
  editable: boolean;
  canManage: boolean;
  archived: boolean;
  busy: boolean;
  onRename: () => void;
  onArchive: () => void;
  onShare: () => void;
  onExport: () => void;
  onFork: () => void;
}) {
  const [open, setOpen] = useState(false);
  const actions = [
    {
      label: "重命名",
      title: "重命名画布",
      icon: Pencil,
      disabled: !editable || busy,
      run: onRename,
    },
    ...(canManage
      ? [
          {
            label: archived ? "恢复" : "归档",
            title: archived ? "恢复画布" : "归档后保留内容并停止编辑",
            icon: archived ? ArchiveRestore : Archive,
            disabled: busy,
            run: onArchive,
          },
        ]
      : []),
    {
      label: "独立副本",
      title: "复制为独立项目",
      icon: FolderGit2,
      disabled: busy,
      run: onFork,
    },
    {
      label: "分享",
      title: "分享画布",
      icon: Users,
      disabled: false,
      run: onShare,
    },
    {
      label: "导出",
      title: "导出与交付",
      icon: Download,
      disabled: false,
      run: onExport,
    },
  ];
  return (
    <>
      <div className="hidden items-center gap-2 lg:flex">
        {actions.map((action) => (
          <Button
            key={action.label}
            variant="ghost"
            size="default"
            title={action.title}
            aria-label={action.title}
            disabled={action.disabled}
            onClick={action.run}
          >
            <action.icon />
            {action.label}
          </Button>
        ))}
      </div>
      <div className="lg:hidden">
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger
            render={
              <Button
                variant="ghost"
                size="icon"
                aria-label="画布更多操作"
              />
            }
          >
            <Ellipsis />
          </PopoverTrigger>
          <PopoverContent
            align="end"
            className="w-48 gap-2 border-border/40 bg-popover/80 p-2 shadow-xl backdrop-blur-xl"
          >
            <PopoverTitle className="sr-only">画布操作</PopoverTitle>
            {actions.map((action) => (
              <Button
                key={action.label}
                variant="ghost"
                size="default"
                className="w-full justify-start"
                disabled={action.disabled}
                onClick={() => {
                  setOpen(false);
                  action.run();
                }}
              >
                <action.icon />
                {action.label}
              </Button>
            ))}
          </PopoverContent>
        </Popover>
      </div>
    </>
  );
}
