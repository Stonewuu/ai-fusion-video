"use client";

import { MousePointer2, Hand, Plus, GitBranch, Play, Sparkles } from "lucide-react";
import {
  Dialog,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";

const steps = [
  {
    icon: Plus,
    title: "添加内容",
    detail: "点击“添加节点”或右键空白处；也可以从左侧项目与素材面板拖入内容。",
  },
  {
    icon: MousePointer2,
    title: "编辑与预览",
    detail:
      "点击节点即可编辑，键盘聚焦后也可按 Enter 打开；图片、视频可点预览按钮查看。按住 Shift 点击可多选。",
  },
  {
    icon: GitBranch,
    title: "连接下一步",
    detail:
      "从节点右侧的输出圆点拖线到另一个节点，或拖到空白处选择可连接的后续步骤。也可点击输出旁的加号。",
  },
  {
    icon: Hand,
    title: "移动画布",
    detail:
      "按住空格拖动，或切换“移动”工具；滚轮缩放。点击顶部“查看全部”可找回画布中的内容。",
  },
  { icon: Play, title: "运行与选择结果", detail: "点击右上角运行，先检查范围与生成数量。完成后到“运行”面板预览结果，挑选合适的内容放入画布。" },
  { icon: Sparkles, title: "让助手帮你创作", detail: "先选中要处理的节点，再打开“助手”描述目标。点击输入框上方的画布引用，可查看范围并设置是否自动定位到助手结果。" },
];
const shortcuts = [
  ["V / H", "选择 / 移动工具"],
  ["N", "添加节点"],
  ["/", "查找节点"],
  ["F", "查看全部节点"],
  ["Enter", "打开聚焦的节点"],
  ["Ctrl / ⌘ + Z", "撤销"],
  ["Ctrl / ⌘ + Shift + Z", "重做"],
  ["Ctrl / ⌘ + D", "复制所选节点"],
  ["Esc", "关闭菜单或收起侧栏"],
];

export function CanvasHelpDialog({ onClose }: { onClose: () => void }) {
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>画布使用指南</DialogTitle>
          <DialogDescription>
            左侧找内容，中间组织创作步骤，右侧编辑与预览。
          </DialogDescription>
        </DialogHeader>
        <div className="min-h-0 space-y-6 overflow-y-auto">
          <ol className="space-y-4">
            {steps.map((step) => (
              <li key={step.title} className="flex gap-3">
                <step.icon className="mt-1 size-4 shrink-0 text-primary" aria-hidden />
                <div className="space-y-1">
                  <h3 className="text-sm font-medium">{step.title}</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {step.detail}
                  </p>
                </div>
              </li>
            ))}
          </ol>
          <section className="space-y-3 border-t border-border/20 pt-4">
            <h3 className="text-sm font-medium">快捷键</h3>
            <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {shortcuts.map(([key, label]) => (
                <div
                  key={key}
                  className="flex items-center justify-between gap-3 text-xs"
                >
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd>
                    <kbd className="font-mono">{key}</kbd>
                  </dd>
                </div>
              ))}
            </dl>
            <p className="text-xs text-muted-foreground">
              输入文字时不会触发画布快捷键。切换侧栏会保留当前输入。
            </p>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
