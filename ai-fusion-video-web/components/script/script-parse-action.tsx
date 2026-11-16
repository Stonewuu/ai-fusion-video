"use client";

import { useState } from "react";
import { Loader2, SlidersHorizontal, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { scriptApi, type Script, type ScriptEpisode } from "@/lib/api/script";
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  parseEpisodesMessage,
  ScriptParseRangeError,
  selectParseEpisodes,
  type ScriptParseScope,
} from "@/lib/scripts/parse-scope";
import { usePipelineStore } from "@/lib/store/pipeline-store";

const scopeOptions = [
  { value: "pending", label: "全部待解析分集" },
  { value: "range", label: "指定集数范围" },
  { value: "all", label: "重新解析全部分集" },
];

interface Props {
  script: Script;
  onComplete: () => void;
  disabled?: boolean;
}

export function ScriptParseAction({
  script,
  onComplete,
  disabled = false,
}: Props) {
  const [loading, setLoading] = useState(false);
  const [episodes, setEpisodes] = useState<ScriptEpisode[] | null>(null);
  const { addPipeline, setPanelExpanded, setExpandedTaskId } =
    usePipelineStore();

  const start = (
    selected: ScriptEpisode[],
    replaceExisting: boolean,
    scope: ScriptParseScope,
  ) => {
    const id = addPipeline({
      label: `解析剧本 · ${selected.length} 集`,
      projectId: script.projectId,
      request: {
        agentType: "script_full_parse",
        toolExecutionMode: "FULL_ACCESS",
        category: "pipeline",
        title: `解析剧本 · ${selected.length} 集`,
        projectId: script.projectId,
        context: {
          scriptId: script.id,
          ...(scope === "range"
            ? {
                selectedScriptEpisodeIds: selected.map((episode) => episode.id),
              }
            : {}),
          replaceExisting,
        },
        message: parseEpisodesMessage(selected, replaceExisting, scope),
      },
      onComplete,
    });
    setEpisodes(null);
    setPanelExpanded(true);
    setExpandedTaskId(id);
  };

  const load = async (configure: boolean) => {
    setLoading(true);
    try {
      const current = await scriptApi.listEpisodes(script.id);
      if (configure) {
        setEpisodes(current);
      } else {
        const pending = selectParseEpisodes(current, "pending", 1, 1, false);
        if (!pending.length) {
          toast.info(
            current.some((episode) => episode.rawContentLength)
              ? "所有分集均已解析，可在“解析范围”中重新解析"
              : "暂无可解析的分集原文，请先导入原文",
          );
          return;
        }
        start(pending, false, "pending");
      }
    } catch (error) {
      toastApiError(error, "无法读取待解析分集，请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="flex items-center gap-2">
        <Button
          variant="ai"
          disabled={disabled || loading}
          onClick={() => void load(false)}
          title="解析全部待解析分集，跳过已完成分集"
        >
          {loading ? <Loader2 className="animate-spin" /> : <Sparkles />}
          {loading ? "正在读取" : "AI 解析"}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          disabled={disabled || loading}
          aria-label="设置解析范围"
          title="指定集数范围或重新解析"
          onClick={() => void load(true)}
        >
          <SlidersHorizontal />
        </Button>
      </div>
      {episodes && (
        <ParseScopeDialog
          episodes={episodes}
          onClose={() => setEpisodes(null)}
          onStart={start}
        />
      )}
    </>
  );
}

function ParseScopeDialog({
  episodes,
  onClose,
  onStart,
}: {
  episodes: ScriptEpisode[];
  onClose: () => void;
  onStart: (
    episodes: ScriptEpisode[],
    replaceExisting: boolean,
    scope: ScriptParseScope,
  ) => void;
}) {
  const [scope, setScope] = useState<ScriptParseScope>("pending");
  const [from, setFrom] = useState("1");
  const [to, setTo] = useState(
    String(Math.max(1, ...episodes.map((episode) => episode.episodeNumber))),
  );
  const [includeCompleted, setIncludeCompleted] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const replaceExisting =
    scope === "all" || (scope === "range" && includeCompleted);
  let selected: ScriptEpisode[] = [];
  let error = "";
  let invalidField: ScriptParseRangeError["field"] | null = null;
  try {
    selected = selectParseEpisodes(
      episodes,
      scope,
      Number(from),
      Number(to),
      replaceExisting,
    );
  } catch (failure) {
    const rangeError = failure as ScriptParseRangeError;
    error = rangeError.message;
    invalidField = rangeError.field;
  }
  const existingCount = selected.filter(
    (episode) => episode.totalScenes > 0,
  ).length;
  const needsConfirmation = replaceExisting && existingCount > 0;

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <DialogContent className="flex max-h-[calc(100dvh-2rem)] flex-col overflow-hidden sm:max-w-lg">
        <DialogHeader className="shrink-0">
          <DialogTitle>解析范围</DialogTitle>
          <DialogDescription>
            默认解析全部待解析分集，已完成的自动跳过。
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="flex flex-col gap-4">
          <div className="flex min-w-0 flex-col gap-2">
            <label htmlFor="script-parse-scope" className="text-sm font-medium">
              解析方式
            </label>
            <Select
              value={scope}
              items={scopeOptions}
              onValueChange={(value) => {
                setScope(value as ScriptParseScope);
                setConfirmed(false);
              }}
            >
              <SelectTrigger id="script-parse-scope" className="w-full">
                <SelectValue placeholder="选择解析方式" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {scopeOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
          {scope === "range" && (
            <>
              <div className="grid min-w-0 grid-cols-2 gap-4">
                <div className="flex min-w-0 flex-col gap-2">
                  <label
                    htmlFor="parse-range-from"
                    className="text-sm font-medium"
                  >
                    开始集数
                  </label>
                  <Input
                    id="parse-range-from"
                    type="number"
                    min={1}
                    step={1}
                    value={from}
                    aria-invalid={
                      invalidField === "from" || invalidField === "range"
                    }
                    aria-describedby={
                      invalidField === "from" || invalidField === "range"
                        ? "script-parse-range-error"
                        : undefined
                    }
                    onChange={(event) => {
                      setFrom(event.target.value);
                      setConfirmed(false);
                    }}
                  />
                </div>
                <div className="flex min-w-0 flex-col gap-2">
                  <label
                    htmlFor="parse-range-to"
                    className="text-sm font-medium"
                  >
                    结束集数
                  </label>
                  <Input
                    id="parse-range-to"
                    type="number"
                    min={1}
                    step={1}
                    value={to}
                    aria-invalid={
                      invalidField === "to" || invalidField === "range"
                    }
                    aria-describedby={
                      invalidField === "to" || invalidField === "range"
                        ? "script-parse-range-error"
                        : undefined
                    }
                    onChange={(event) => {
                      setTo(event.target.value);
                      setConfirmed(false);
                    }}
                  />
                </div>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <Checkbox
                  checked={includeCompleted}
                  onCheckedChange={(checked) => {
                    setIncludeCompleted(checked === true);
                    setConfirmed(false);
                  }}
                />
                包含已完成分集，重新解析已有场次
              </label>
            </>
          )}
          {!error && (
            <p className="text-sm text-muted-foreground">
              本次解析 {selected.length} 集
            </p>
          )}
          {!error && selected.length === 0 && (
            <p className="text-sm text-muted-foreground">
              当前范围没有待解析原文，可调整范围或先导入原文。
            </p>
          )}
          {needsConfirmation && (
            <label className="flex items-start gap-2 text-sm">
              <Checkbox
                checked={confirmed}
                onCheckedChange={(checked) => setConfirmed(checked === true)}
              />
              <span>
                确认替换 {existingCount} 集的已有场次。分集关联和已有分镜保留。
              </span>
            </label>
          )}
          {error && (
            <p
              id="script-parse-range-error"
              role="alert"
              className="text-sm text-destructive"
            >
              {error}
            </p>
          )}
        </DialogScrollBody>
        <DialogFooter className="shrink-0 gap-2">
          <Button variant="outline" onClick={onClose}>
            取消
          </Button>
          <Button
            variant="ai"
            disabled={
              !!error || !selected.length || (needsConfirmation && !confirmed)
            }
            onClick={() => {
              if (!error) {
                onStart(selected, replaceExisting, scope);
              }
            }}
          >
            <Sparkles />
            解析 {selected.length} 集
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
