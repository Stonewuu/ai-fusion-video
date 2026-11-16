"use client";

import { useEffect, useState } from "react";
import { Loader2, Pencil, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { scriptApi, type ScriptEpisode } from "@/lib/api/script";

interface Props {
  episode: ScriptEpisode;
  onSaved?: (episode: ScriptEpisode) => void;
  onParse: () => void;
}

export function EpisodeSourcePanel({ episode, onSaved, onParse }: Props) {
  const [loaded, setLoaded] = useState<ScriptEpisode | null>(null);
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const requests = useCommandRequestKey();

  useEffect(() => {
    if (editing) {
      return;
    }
    let active = true;
    scriptApi
      .getEpisode(episode.id)
      .then((result) => {
        if (active) {
          setLoaded(result);
          setDraft(result.rawContent ?? "");
          setError("");
        }
      })
      .catch((failure) => {
        if (active) {
          setError(
            failure instanceof Error ? failure.message : "本集原文加载失败",
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
  }, [episode.id, episode.version, editing, revision]);

  const save = async () => {
    if (!loaded || !draft.trim()) {
      setError("请输入本集原文");
      return;
    }
    setSaving(true);
    setError("");
    try {
      const payload = {
        id: episode.id,
        version: loaded.version,
        rawContent: draft,
      };
      const result = await scriptApi.updateEpisode(
        payload,
        requests.forPayload(payload),
      );
      requests.clear();
      setLoaded(result);
      setEditing(false);
      onSaved?.(result);
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : "原文保存失败，请重试",
      );
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        正在加载本集原文
      </div>
    );
  }
  return (
    <div className="flex min-w-0 flex-col gap-3 p-1">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <label
          htmlFor={`episode-source-${episode.id}`}
          className="text-sm font-medium"
        >
          本集原文 · {draft.length.toLocaleString()} 字
        </label>
        {!editing && loaded && (
          <div className="flex items-center gap-2">
            <Button variant="ghost" onClick={() => setEditing(true)}>
              <Pencil />
              编辑原文
            </Button>
            <Button
              variant="ai"
              onClick={onParse}
              disabled={!loaded.rawContent?.trim()}
            >
              <Sparkles />
              解析本集
            </Button>
          </div>
        )}
      </div>
      {loaded && (
        <Textarea
          id={`episode-source-${episode.id}`}
          readOnly={!editing}
          disabled={saving}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          className="field-sizing-fixed min-h-80 w-full resize-y"
          placeholder="填写本集原文"
        />
      )}
      {editing && (
        <div className="flex justify-end gap-2">
          <Button
            variant="outline"
            disabled={saving}
            onClick={() => {
              setEditing(false);
              setDraft(loaded?.rawContent ?? "");
              setError("");
            }}
          >
            取消
          </Button>
          <Button disabled={saving || !draft.trim()} onClick={save}>
            {saving && <Loader2 className="animate-spin" />}保存原文
          </Button>
        </div>
      )}
      {error && (
        <div className="flex flex-wrap items-center gap-2">
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
          {!loaded && (
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
        </div>
      )}
    </div>
  );
}
