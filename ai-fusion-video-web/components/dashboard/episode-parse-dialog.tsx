"use client";

import { useEffect, useState } from "react";
import { Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { scriptApi, type ScriptEpisode } from "@/lib/api/script";

interface Props {
  open: boolean;
  episode: ScriptEpisode | null;
  hasExistingScenes: boolean;
  existingSceneCount?: number;
  onClose: () => void;
  onStartParse: (
    source: string,
    version: number,
    replaceExisting: boolean,
  ) => Promise<void>;
}

export function EpisodeParseDialog(props: Props) {
  return props.open && props.episode ? (
    <ParseForm {...props} episode={props.episode} />
  ) : null;
}

function ParseForm({
  episode,
  onClose,
  onStartParse,
}: Props & { episode: ScriptEpisode }) {
  const [source, setSource] = useState("");
  const [loaded, setLoaded] = useState<ScriptEpisode | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    let active = true;
    scriptApi
      .getEpisode(episode.id)
      .then((value) => {
        if (active) {
          setLoaded(value);
          setSource(value.rawContent ?? "");
          setError("");
        }
      })
      .catch((failure) => {
        if (active) {
          setError(
            failure instanceof Error
              ? failure.message
              : "本集原文加载失败，请重试",
          );
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [episode.id, revision]);

  const submit = async () => {
    if (!loaded || !source.trim()) {
      setError("请输入本集原文");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await onStartParse(source, loaded.version, confirmed);
      onClose();
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : "无法开始解析，请重试",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) {
          onClose();
        }
      }}
    >
      <DialogContent
        showCloseButton={!busy}
        className="flex max-h-[calc(100dvh-2rem)] flex-col overflow-hidden sm:max-w-2xl"
      >
        <DialogHeader className="shrink-0">
          <DialogTitle>解析第 {episode.episodeNumber} 集</DialogTitle>
          <DialogDescription>
            {episode.title} · 提取场次、对白并生成本集梗概
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="flex flex-col gap-4">
          {loading ? (
            <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              正在加载本集原文
            </div>
          ) : (
            loaded && (
              <div className="flex min-w-0 flex-col gap-2">
                <label
                  htmlFor="parse-episode-source"
                  className="text-sm font-medium"
                >
                  本集原文
                </label>
                <Textarea
                  id="parse-episode-source"
                  value={source}
                  disabled={busy}
                  onChange={(event) => setSource(event.target.value)}
                  rows={12}
                  className="field-sizing-fixed min-h-64 resize-y"
                  placeholder="填写本集原文"
                />
              </div>
            )
          )}
          {!!loaded?.totalScenes && (
            <label className="flex items-start gap-2 text-sm">
              <Checkbox
                checked={confirmed}
                disabled={busy}
                onCheckedChange={(checked) => setConfirmed(checked === true)}
              />
              <span>
                重新解析并替换本集 {loaded.totalScenes}{" "}
                个场次。已有分镜保留，可按更新后的剧本重新生成。
              </span>
            </label>
          )}
          {!!loaded?.totalScenes &&
            source !== (loaded.rawContent ?? "") &&
            !confirmed && (
              <p role="alert" className="text-sm text-destructive">
                原文已修改，请勾选重新解析后继续。
              </p>
            )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </DialogScrollBody>
        <DialogFooter className="shrink-0 gap-2">
          {!loaded && !loading && (
            <Button
              variant="outline"
              onClick={() => {
                setLoading(true);
                setRevision((value) => value + 1);
              }}
            >
              重新加载
            </Button>
          )}
          <Button variant="outline" disabled={busy} onClick={onClose}>
            取消
          </Button>
          <Button
            variant="ai"
            disabled={
              busy ||
              loading ||
              !loaded ||
              !source.trim() ||
              (!!loaded.totalScenes &&
                source !== (loaded.rawContent ?? "") &&
                !confirmed)
            }
            onClick={submit}
          >
            {busy ? <Loader2 className="animate-spin" /> : <Sparkles />}
            {busy
              ? "正在启动"
              : loaded?.totalScenes && !confirmed
                ? "继续解析"
                : "开始解析"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
