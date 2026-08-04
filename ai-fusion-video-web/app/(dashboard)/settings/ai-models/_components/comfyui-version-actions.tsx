"use client";

import { useMemo, useState } from "react";
import { CheckCircle2, FlaskConical, Loader2, Rocket, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  comfyUiWorkflowApi,
  type ComfyUiBindingValueType,
  type ComfyUiWorkflow,
  type ComfyUiWorkflowVersion,
} from "@/lib/api/comfyui-workflow";
import { parseInputBindingsJson } from "./comfyui-workflow-support";

interface ComfyUiVersionActionsProps {
  workflow: ComfyUiWorkflow;
  version: ComfyUiWorkflowVersion | null;
  onUpdated: () => Promise<void> | void;
}

function statusLabel(status: number, passed: string, failed: string, pending: string) {
  if (status === 1) return passed;
  if (status === 2) return failed;
  return pending;
}

function inputKind(valueType: ComfyUiBindingValueType): "text" | "number" | "boolean" | "list" {
  if (valueType === "integer" || valueType === "number") return "number";
  if (valueType === "boolean") return "boolean";
  if (valueType === "string_list" || valueType.startsWith("uploaded_")) return "list";
  return "text";
}

export function ComfyUiVersionActions({ workflow, version, onUpdated }: ComfyUiVersionActionsProps) {
  const [running, setRunning] = useState<"validate" | "test" | "publish" | null>(null);
  const [testValues, setTestValues] = useState<Record<string, string>>({ prompt: "一张电影感的测试画面" });

  const testFields = useMemo(() => {
    if (!version) return [];
    try {
      const bindings = parseInputBindingsJson(version.inputBindingsJson);
      return Object.entries(bindings).map(([businessField, items]) => ({
        businessField,
        kind: inputKind(items[0]?.valueType || "string"),
      }));
    } catch {
      return [];
    }
  }, [version]);

  const buildInputs = () => testFields.reduce<Record<string, unknown>>((inputs, field) => {
    const rawValue = testValues[field.businessField]?.trim();
    if (!rawValue) return inputs;
    if (field.kind === "number") inputs[field.businessField] = Number(rawValue);
    else if (field.kind === "boolean") inputs[field.businessField] = rawValue === "true";
    if (field.kind === "list") {
      inputs[field.businessField] = rawValue.split(/[\n,]/).map(value => value.trim()).filter(Boolean);
    } else if (field.kind === "text") {
      inputs[field.businessField] = rawValue;
    }
    return inputs;
  }, {});

  const validate = async () => {
    if (!version) return;
    setRunning("validate");
    try {
      const result = await comfyUiWorkflowApi.validateVersion(version.id);
      if (result.valid) {
        toast.success("在线校验通过", { description: `已检查 ${result.checkedNodeCount} 个节点` });
      } else {
        toast.error("在线校验未通过", { description: result.message });
      }
      await onUpdated();
    } catch (error) {
      toastApiError(error, "在线校验失败");
    } finally {
      setRunning(null);
    }
  };

  const test = async () => {
    if (!version) return;
    setRunning("test");
    try {
      const result = await comfyUiWorkflowApi.testVersion(version.id, buildInputs());
      if (result.passed) {
        toast.success("真实试运行成功", { description: `生成 ${result.outputs.length} 个结果，耗时 ${result.durationMillis}ms` });
      } else {
        toast.error("真实试运行失败", { description: result.message });
      }
      await onUpdated();
    } catch (error) {
      toastApiError(error, "真实试运行失败");
      await onUpdated();
    } finally {
      setRunning(null);
    }
  };

  const publish = async () => {
    if (!version) return;
    setRunning("publish");
    try {
      await comfyUiWorkflowApi.publish(workflow.id, version.id);
      toast.success(workflow.activeVersionId ? "工作流版本已切换" : "工作流已发布");
      await onUpdated();
    } catch (error) {
      toastApiError(error, "发布工作流失败");
    } finally {
      setRunning(null);
    }
  };

  if (!version) {
    return (
      <div className="rounded-lg border border-border/20 bg-background/70 p-4 text-sm text-muted-foreground">
        先保存工作流版本，才能在线校验、真实试运行和发布。
      </div>
    );
  }

  const isActive = workflow.activeVersionId === version.id;
  const publishReady = version.validationStatus === 1 && version.testStatus === 1;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-lg border border-border/20 bg-background/70 p-3">
          <p className="text-xs text-muted-foreground">本地结构</p>
          <p className="mt-1 text-sm font-medium"><CheckCircle2 className="mr-1.5 inline size-4 text-emerald-500" />已保存并固化哈希</p>
        </div>
        <div className="rounded-lg border border-border/20 bg-background/70 p-3">
          <p className="text-xs text-muted-foreground">在线校验</p>
          <p className="mt-1 text-sm font-medium">{statusLabel(version.validationStatus, "已通过", "未通过", "未校验")}</p>
          {version.validationMessage && <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{version.validationMessage}</p>}
        </div>
        <div className="rounded-lg border border-border/20 bg-background/70 p-3">
          <p className="text-xs text-muted-foreground">真实试运行</p>
          <p className="mt-1 text-sm font-medium">{statusLabel(version.testStatus, "已成功", "失败", "未运行")}</p>
          {version.testMessage && <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{version.testMessage}</p>}
        </div>
      </div>

      <div className="rounded-lg border border-border/20 bg-background/70 p-3 space-y-3">
        <div>
          <p className="text-sm font-medium">试运行输入</p>
          <p className="mt-1 text-xs text-muted-foreground">引用资源每行填写一个 URL 或 Data URI；试运行结果会保存到平台存储。</p>
        </div>
        {testFields.length === 0 ? (
          <p className="text-xs text-muted-foreground">当前版本没有平台输入绑定。</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {testFields.map(field => (
              <div key={field.businessField} className="space-y-1.5">
                <Label className="text-[11px] text-muted-foreground">{field.businessField}</Label>
                {field.kind === "list" ? (
                  <Textarea
                    value={testValues[field.businessField] || ""}
                    onChange={event => setTestValues(previous => ({ ...previous, [field.businessField]: event.target.value }))}
                    placeholder="每行一个 URL 或 Data URI"
                    className="min-h-20 resize-none text-xs"
                  />
                ) : (
                  <Input
                    type={field.kind === "number" ? "number" : "text"}
                    value={testValues[field.businessField] || ""}
                    onChange={event => setTestValues(previous => ({ ...previous, [field.businessField]: event.target.value }))}
                    placeholder={field.kind === "boolean" ? "true 或 false" : "输入测试值"}
                    className="text-xs"
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex flex-wrap justify-end gap-2">
        <Button variant="outline" size="sm" onClick={validate} disabled={running !== null}>
          {running === "validate" ? <Loader2 className="animate-spin" /> : <ShieldCheck />}
          在线校验
        </Button>
        <Button variant="secondary" size="sm" onClick={test} disabled={running !== null || version.validationStatus !== 1}>
          {running === "test" ? <Loader2 className="animate-spin" /> : <FlaskConical />}
          真实试运行
        </Button>
        <Button size="sm" onClick={publish} disabled={running !== null || !publishReady || isActive}>
          {running === "publish" ? <Loader2 className="animate-spin" /> : <Rocket />}
          {isActive ? "当前已发布" : version.published ? "回滚到此版本" : "发布版本"}
        </Button>
      </div>
    </div>
  );
}
