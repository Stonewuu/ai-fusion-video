import { Check, ChevronRight } from "lucide-react";
import {
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

export function ImportStepHeader({ preview }: { preview: boolean }) {
  return (
    <DialogHeader className="shrink-0 gap-4 border-b border-border/20 px-4 py-5 sm:px-6">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-4 pr-8">
        <div className="flex flex-col gap-2">
          <DialogTitle className="text-xl font-semibold">
            {preview ? "检查分集" : "导入原文"}
          </DialogTitle>
          <DialogDescription>
            {preview
              ? "检查标题与分界，整理好后导入剧本。"
              : "将小说、故事或剧本整理为分集。"}
          </DialogDescription>
        </div>
        <ol
          aria-label="导入步骤"
          className="flex items-center gap-3 text-sm sm:ml-auto"
        >
          <li
            aria-current={!preview ? "step" : undefined}
            className="flex items-center gap-2"
          >
            <span
              className={cn(
                "flex size-6 items-center justify-center rounded-full text-xs font-semibold",
                preview
                  ? "bg-primary/10 text-primary"
                  : "bg-primary text-primary-foreground",
              )}
            >
              {preview ? <Check className="size-3.5" /> : "1"}
            </span>
            <span
              className={cn(
                !preview && "font-medium",
                preview && "text-muted-foreground",
              )}
            >
              导入原文
            </span>
          </li>
          <ChevronRight
            aria-hidden
            className="size-4 text-muted-foreground/60"
          />
          <li
            aria-current={preview ? "step" : undefined}
            className="flex items-center gap-2"
          >
            <span
              className={cn(
                "flex size-6 items-center justify-center rounded-full text-xs font-semibold",
                preview
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground",
              )}
            >
              2
            </span>
            <span
              className={cn(preview ? "font-medium" : "text-muted-foreground")}
            >
              检查分集
            </span>
          </li>
        </ol>
      </div>
    </DialogHeader>
  );
}
