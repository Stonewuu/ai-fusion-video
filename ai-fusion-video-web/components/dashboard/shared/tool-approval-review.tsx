"use client";

import { AlertTriangle, FileCheck2 } from "lucide-react";
import type { ToolApprovalReview } from "@/lib/ai/tool-approval-review";

export function ToolApprovalReviewContent({ review }: { review: ToolApprovalReview }) {
  return (
    <section className="flex flex-col gap-3 border-t border-border/20 p-3" aria-label={review.title}>
      <div className="flex items-start gap-2">
        <FileCheck2 className="mt-0.5 size-4 shrink-0 text-sky-700 dark:text-sky-300" />
        <div className="min-w-0">
          <h4 className="text-sm font-medium">{review.title}</h4>
          <p className="break-words text-xs text-muted-foreground">{review.summary}</p>
        </div>
      </div>
      {review.rows.length > 0 ? (
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-2 text-xs">
          {review.rows.map((row, index) => (
            <div key={index} className="contents">
              <dt className="text-muted-foreground">{row.label}</dt>
              <dd className="min-w-0 whitespace-pre-wrap break-words">
                {row.value.length > 280 ? (
                  <details>
                    <summary className="cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-ring">
                      {row.value.slice(0, 120)}…（展开全部，共 {row.value.length} 字）
                    </summary>
                    <p className="mt-2 max-h-64 overflow-y-auto p-1">{row.value}</p>
                  </details>
                ) : row.value}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
      {review.changes.length > 0 ? (
        <details open={review.changes.length <= 6}>
          <summary className="cursor-pointer text-xs font-medium outline-none focus-visible:ring-2 focus-visible:ring-ring">
            影响 {review.changes.length} 项
          </summary>
          <ul className="mt-2 flex max-h-64 flex-col gap-2 overflow-y-auto px-1 py-1">
            {review.changes.map((change, index) => (
              <li key={index} className="flex items-start gap-2 text-xs">
                <span className={change.action === "删除" ? "shrink-0 text-destructive" : "shrink-0 text-sky-700 dark:text-sky-300"}>{change.action}</span>
                <div className="min-w-0">
                  <p className="break-words font-medium">{change.object}</p>
                  {change.detail ? change.detail.length > 280 ? (
                    <details className="text-muted-foreground">
                      <summary className="cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-ring">{change.detail.slice(0, 120)}…（展开完整修改）</summary>
                      <p className="mt-2 whitespace-pre-wrap break-words">{change.detail}</p>
                    </details>
                  ) : <p className="whitespace-pre-wrap break-words text-muted-foreground">{change.detail}</p> : null}
                </div>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {review.warnings.map((warning, index) => (
        <p key={index} className="flex items-start gap-2 text-xs text-muted-foreground">
          <AlertTriangle className="size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
          <span>{warning}</span>
        </p>
      ))}
    </section>
  );
}
