"use client";

import { useEffect, useId, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { mediaJobsApi, type EpisodeAudioReview, type EpisodeAudioReviewAction, type MediaJob } from "@/lib/api/media-jobs";
import { generationExecutionApi, type GenerationScope } from "@/lib/api/generation-execution";
import { getApiErrorMessage } from "@/lib/api/api-error";

const actions: Record<EpisodeAudioReviewAction, { label: string; description: string }> = {
  RECHECK: { label: "继续查询原任务", description: "只查询已受理的原任务。取消中的任务结束后会停止，不会开始下一轨音频。" },
  RECORD_RECEIPT: { label: "补记原受理回执", description: "在供应商侧找到原请求后填写其任务编号；请勿填写新请求的编号或结果 URL。" },
  RETRY_INGEST: { label: "继续登记已有结果", description: "使用已保存的音频终态响应继续下载、登记和混音。业务版本变化时仍不会覆盖新内容。" },
  CONFIRM_FINISHED: { label: "确认远端结束", description: "确认所有待核实调用都已结束后，释放额度并停止后续处理，已登记的结果仍保留。" },
};

export function EpisodeAudioReviewForm({ job, onChanged }: { job: MediaJob; onChanged: () => void }) {
  const [loaded, setLoaded] = useState<{ detail: EpisodeAudioReview; scope: GenerationScope }>();
  const [action, setAction] = useState<EpisodeAudioReviewAction | null>(null);
  const [step, setStep] = useState<string | null>(null);
  const [receipt, setReceipt] = useState("");
  const [note, setNote] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const factsKey = useRef<string | null>(null);
  const id = useId();
  useEffect(() => {
    const controller = new AbortController();
    void Promise.all([mediaJobsApi.audio(job.id, controller.signal), generationExecutionApi.scope()])
      .then(([detail, scope]) => {
        if (!controller.signal.aborted) {
          const key = `${detail.jobId}:${detail.factsHash}:${scope.userId}:${scope.teamId}`;
          if (factsKey.current !== key) {
            setAction(null);
            setStep(null);
            setReceipt("");
            setConfirmed(false);
            factsKey.current = key;
          }
          setLoaded({ detail, scope });
          setError(null);
        }
      }).catch(failure => { if (!controller.signal.aborted) setError(getApiErrorMessage(failure)); });
    return () => controller.abort();
  }, [job.id, job.revision, refresh]);
  const detail = loaded?.detail;
  const current = detail?.jobId === job.id && detail.revision === job.revision;
  const active = current && action && detail?.actions.includes(action) ? action : null;
  const options = detail?.actions.map(value => ({ value, label: actions[value].label })) ?? [];
  const steps = detail?.steps.filter(value => value.unresolved && !value.providerTaskId)
    .map(value => ({ value: value.step, label: value.step === "music" ? "配乐" : "音效" })) ?? [];
  const submit = async () => {
    if (!loaded || !active || busy) return;
    setBusy(true);
    setError(null);
    try {
      await mediaJobsApi.reviewAudio(job.id, {
        expectedRevision: loaded.detail.revision, expectedFactsHash: loaded.detail.factsHash, action: active, note: note.trim(),
        step, providerTaskId: receipt.trim() || null, confirmedEnded: confirmed, scope: loaded.scope,
      });
      onChanged();
    } catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  return (
    <section className="flex flex-col gap-4 border-t border-border/20 pt-4">
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">核实音频请求</h3>
        {detail && <p className="text-xs text-muted-foreground">{detail.capacityHeld ? "远端尚未确认结束，仍保留并发名额。" : "远端并发名额已释放。"}</p>}
        {detail?.steps.map(value => <p key={value.step} className="break-all text-xs text-muted-foreground">
          {value.step === "music" ? "配乐" : "音效"} · {value.providerTaskId ?? "受理回执未确认"} · {value.unresolved ? "待核实" : "已结束"}
        </p>)}
      </div>
      {!detail && !error && <p className="text-sm text-muted-foreground">正在读取原请求…</p>}
      {!!options.length && <>
        <div className="grid gap-2">
          <Label htmlFor={`${id}-action`}>处理方式</Label>
          <Select items={options} value={active} onValueChange={value => { setAction(value); setConfirmed(false); }} disabled={busy}>
            <SelectTrigger id={`${id}-action`} className="w-full"><SelectValue placeholder="选择处理方式" /></SelectTrigger>
            <SelectContent><SelectGroup>{options.map(item => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}</SelectGroup></SelectContent>
          </Select>
          {active && <p className="text-xs text-muted-foreground">{actions[active].description}</p>}
        </div>
        {active === "RECORD_RECEIPT" && <>
          <div className="grid gap-2">
            <Label htmlFor={`${id}-step`}>原请求步骤</Label>
            <Select items={steps} value={step} onValueChange={setStep} disabled={busy}>
              <SelectTrigger id={`${id}-step`} className="w-full"><SelectValue placeholder="选择步骤" /></SelectTrigger>
              <SelectContent><SelectGroup>{steps.map(item => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}</SelectGroup></SelectContent>
            </Select>
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${id}-receipt`}>供应商原任务编号</Label>
            <Input id={`${id}-receipt`} value={receipt} onChange={event => setReceipt(event.target.value)} maxLength={200} disabled={busy} />
          </div>
        </>}
        <div className="grid gap-2">
          <Label htmlFor={`${id}-note`}>核实依据</Label>
          <Textarea id={`${id}-note`} value={note} onChange={event => setNote(event.target.value)} maxLength={2000} disabled={busy} placeholder="填写供应商记录中的状态、核实时间或确认信息" />
        </div>
        {active === "CONFIRM_FINISHED" && <label className="flex items-start gap-2 text-sm">
          <Checkbox checked={confirmed} onCheckedChange={value => setConfirmed(value === true)} disabled={busy} />
          已核实上方所有待确认的远端请求均已结束
        </label>}
        <Button disabled={busy || !active || !note.trim() || (active === "CONFIRM_FINISHED" && !confirmed)
          || (active === "RECORD_RECEIPT" && (!steps.some(item => item.value === step) || !receipt.trim()))} onClick={() => void submit()}>
          {busy ? "提交中…" : "确认处理"}
        </Button>
      </>}
      {detail && !options.length && <p className="text-xs text-muted-foreground">当前账号没有可用核实操作，或任务状态已经变化。</p>}
      {error && <div className="flex flex-col items-start gap-2">
        <p role="alert" className="text-sm text-destructive">{error}</p>
        <Button variant="outline" size="sm" disabled={busy} onClick={() => { setRefresh(value => value + 1); onChanged(); }}>刷新状态</Button>
      </div>}
    </section>
  );
}
