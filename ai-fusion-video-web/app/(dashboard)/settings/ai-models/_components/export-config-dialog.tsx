"use client";

import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { DialogContent } from "@/components/shared/dialog-content";

interface ExportConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  includeSecrets: boolean;
  onIncludeSecretsChange: (value: boolean) => void;
  onExport: (includeSecrets: boolean) => void;
}

export function ExportConfigDialog({
  open,
  onOpenChange,
  includeSecrets,
  onIncludeSecretsChange,
  onExport,
}: ExportConfigDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>导出配置</DialogTitle>
          <DialogDescription>导出当前全部 API 配置和模型为 JSON 文件。</DialogDescription>
        </DialogHeader>

        <div className="flex items-center justify-between gap-4 rounded-xl border border-border/30 bg-background/70 p-3">
          <div className="flex min-w-0 flex-col gap-1">
            <Label htmlFor="export-include-secrets" className="cursor-pointer text-sm font-medium">
              包含密钥与代理密码
            </Label>
            <p className="text-xs text-muted-foreground">
              仅在个人备份或可信环境中使用；关闭时会移除所有密钥字段。
            </p>
          </div>
          <Switch
            id="export-include-secrets"
            checked={includeSecrets}
            onCheckedChange={onIncludeSecretsChange}
          />
        </div>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>取消</DialogClose>
          <Button onClick={() => onExport(includeSecrets)}>
            <Download />
            导出 JSON
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
