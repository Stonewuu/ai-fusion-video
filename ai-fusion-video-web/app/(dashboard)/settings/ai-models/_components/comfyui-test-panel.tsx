"use client";

import { useMemo, useState } from "react";
import { FlaskConical, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import type { ComfyUiWorkflowVersion } from "@/lib/api/comfyui-workflow";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { comfyUiTestInput } from "@/lib/comfyui-test-request";
import { parseInputBindingsJson } from "./comfyui-workflow-support";
import { ComfyUiTestFields } from "./comfyui-test-fields";
import { ComfyUiTestHistory } from "./comfyui-test-history";
import { useComfyUiTests } from "./use-comfyui-tests";

export function ComfyUiTestPanel({
  version,
  disabled,
  onUpdated,
}: {
  version: ComfyUiWorkflowVersion;
  disabled: boolean;
  onUpdated: () => void | Promise<void>;
}) {
  const tests = useComfyUiTests(version.id, onUpdated);
  const [values, setValues] = useState<Record<string, string>>({
    prompt: "一张电影感的测试画面",
  });
  const [formError, setFormError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<"withdraw" | "clear" | null>(
    null,
  );
  const [uploads, setUploads] = useState<string[]>([]);
  const parsed = useMemo(() => {
    try {
      const bindings = parseInputBindingsJson(version.inputBindingsJson);
      const fields = Object.entries(bindings).map(([name, items]) => {
        if (
          !items[0] ||
          items.some((item) => item.valueType !== items[0].valueType)
        )
          throw new Error(`字段 ${name} 的绑定类型不一致，请先修复版本。`);
        if (
          items[0].valueType === "uploaded_video" ||
          items[0].valueType === "uploaded_audio"
        )
          throw new Error(
            "当前 ComfyUI Native API 尚不支持视频或音频上传绑定。",
          );
        return { name, type: items[0].valueType };
      });
      return { fields, error: null };
    } catch (error) {
      return { fields: [], error: getApiErrorMessage(error) };
    }
  }, [version.inputBindingsJson]);
  const blocked =
    disabled ||
    tests.loading ||
    tests.busy ||
    !!tests.pending ||
    tests.corrupt ||
    uploads.length > 0;
  const submit = async () => {
    try {
      const inputs: Record<string, unknown> = {};
      for (const field of parsed.fields) {
        const raw = values[field.name]?.trim();
        if (raw) inputs[field.name] = comfyUiTestInput(raw, field.type);
      }
      setFormError(null);
      await tests.submit(version.workflowHash, inputs);
    } catch (error) {
      setFormError(getApiErrorMessage(error));
    }
  };
  const error = formError ?? parsed.error ?? tests.error;
  return (
    <section
      className="space-y-4 border-t border-border/20 pt-4"
      aria-label="工作流试运行"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">试运行</h3>
          <p className="text-xs text-muted-foreground">
            提交后可关闭页面，重新打开仍可查看进度和结果。每个供应商一次执行一个试运行。
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          title="刷新试运行"
          disabled={tests.busy}
          onClick={tests.reload}
        >
          <RefreshCw />
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {tests.pending && (
        <div className="space-y-3 rounded-lg border border-border/40 bg-background/70 p-3">
          <p className="text-sm font-medium">还有一条请求需要确认</p>
          <p className="text-xs text-muted-foreground">
            原始输入已保存。确认会查询或受理同一请求；撤回会阻止迟到提交，并取消已经受理的任务。
          </p>
          {typeof tests.pending.request.inputs.prompt === "string" && (
            <p className="line-clamp-3 text-sm">
              {tests.pending.request.inputs.prompt}
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              disabled={tests.busy}
              onClick={() => void tests.confirm()}
            >
              确认原请求
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={tests.busy}
              onClick={() => setConfirmation("withdraw")}
            >
              撤回请求
            </Button>
          </div>
        </div>
      )}
      {tests.corrupt && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => setConfirmation("clear")}
        >
          清除损坏的本地记录
        </Button>
      )}
      <ComfyUiTestFields
        fields={parsed.fields}
        values={values}
        disabled={blocked}
        onUploading={(name, uploading) =>
          setUploads((previous) =>
            uploading
              ? [...new Set([...previous, name])]
              : previous.filter((item) => item !== name),
          )
        }
        onChange={(name, value) =>
          setValues((previous) => ({ ...previous, [name]: value }))
        }
        onAppend={(name, references) =>
          setValues((previous) => ({
            ...previous,
            [name]: [previous[name]?.trim(), ...references]
              .filter(Boolean)
              .join("\n"),
          }))
        }
      />
      {!tests.pending && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-xs text-muted-foreground">
            {version.validationStatus !== 1
              ? "先完成在线校验，再试运行当前保存的版本。"
              : "将按当前保存的版本和输入运行，可能产生供应商用量。"}
          </p>
          <Button
            disabled={
              blocked || !!parsed.error || version.validationStatus !== 1
            }
            onClick={() => void submit()}
          >
            <FlaskConical />
            {tests.busy ? "正在提交…" : "开始试运行"}
          </Button>
        </div>
      )}
      <ComfyUiTestHistory
        workflowHash={version.workflowHash}
        runs={tests.runs}
        busy={tests.busy}
        error={tests.error}
        onCancel={tests.cancel}
        onReview={tests.review}
      />
      {confirmation && (
        <Dialog
          open
          onOpenChange={(open) => {
            if (!open && !tests.busy) setConfirmation(null);
          }}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {confirmation === "withdraw" ? "撤回原请求" : "清除本地记录"}
              </DialogTitle>
              <DialogDescription>
                {confirmation === "withdraw"
                  ? "尚未受理的请求将被阻止；已经受理的任务会请求取消，供应商可能仍会完成或计费。"
                  : "只删除无法读取的浏览器记录，不会取消服务器任务。请先检查下方任务历史，避免重新提交相同工作。"}
              </DialogDescription>
            </DialogHeader>
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            <DialogFooter>
              <Button
                variant="outline"
                disabled={tests.busy}
                onClick={() => setConfirmation(null)}
              >
                返回
              </Button>
              <Button
                disabled={tests.busy}
                onClick={async () => {
                  try {
                    if (confirmation === "clear") {
                      await tests.clearCorrupt();
                      setConfirmation(null);
                    } else if (await tests.withdraw()) setConfirmation(null);
                  } catch (failure) {
                    setFormError(getApiErrorMessage(failure));
                  }
                }}
              >
                确认{confirmation === "withdraw" ? "撤回" : "清除"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </section>
  );
}
