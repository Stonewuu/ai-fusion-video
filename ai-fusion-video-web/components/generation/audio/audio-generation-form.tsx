"use client";

import { useRef, useState } from "react";
import { Loader2, Paperclip, Play, X } from "lucide-react";
import type { AiModel } from "@/lib/api/ai-model";
import { audioGenerationApi, type AudioGenerationTask } from "@/lib/api/audio-generation";
import { uploadMedia, type ManagedMedia } from "@/lib/api/media";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

export function AudioGenerationForm({ model, onCreated, onRefresh }: { model: AiModel; onCreated: (task: AudioGenerationTask) => void; onRefresh: () => void }) {
  const [text, setText] = useState("");
  const [voice, setVoice] = useState("");
  const [files, setFiles] = useState<ManagedMedia[]>([]);
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const upload = useRef<HTMLInputElement>(null);
  const pendingRequest = useRef<{ id: string; fingerprint: string } | null>(null);
  const music = model.code.startsWith("lyria-");
  const allowsFiles = model.code.startsWith("lyria-3");

  const addFiles = async (selected: FileList | null) => {
    if (!selected?.length) {
      return;
    }
    if (files.length + selected.length > 6) {
      setError("每次最多添加 6 张参考图");
      return;
    }
    setUploading(true);
    setError("");
    try {
      for (const file of Array.from(selected)) {
        const uploaded = await uploadMedia(file, { requestId: crypto.randomUUID() });
        setFiles(current => [...current, uploaded]);
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "文件上传失败，请重试");
    } finally {
      setUploading(false);
      if (upload.current) {
        upload.current.value = "";
      }
    }
  };

  const submit = async () => {
    if (!text.trim()) {
      setError(music ? "请输入音乐描述" : "请输入朗读内容");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const input = {
        text,
        media: files.map(file => "media://" + file.id),
        ...(!music && voice.trim() ? { voice: voice.trim() } : {}),
      };
      const fingerprint = JSON.stringify({ modelId: model.id, input });
      if (pendingRequest.current?.fingerprint !== fingerprint) {
        pendingRequest.current = { id: crypto.randomUUID(), fingerprint };
      }
      const task = await audioGenerationApi.create(model.id, pendingRequest.current.id, input);
      pendingRequest.current = null;
      onCreated(task);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "音频生成未能开始，请重试");
    } finally {
      setSubmitting(false);
      onRefresh();
    }
  };

  return (
    <div className="flex min-w-0 flex-col gap-4 p-1">
      <div className="flex flex-col gap-2">
        <Label htmlFor="audio-generation-text">{music ? "音乐描述" : "朗读内容"}</Label>
        <Textarea id="audio-generation-text" value={text} maxLength={16000} rows={6} aria-invalid={!!error && !text.trim()}
          placeholder={music ? "描述音乐风格、节奏和氛围" : "输入需要朗读的文字"}
          onChange={event => {
            setText(event.target.value);
            setError("");
          }} />
      </div>
      {!music && <div className="flex flex-col gap-2">
        <Label htmlFor="audio-generation-voice">音色</Label>
        <Input id="audio-generation-voice" value={voice} maxLength={100} placeholder="使用模型默认音色"
          onChange={event => setVoice(event.target.value)} />
      </div>}
      {allowsFiles && <div className="flex flex-col gap-2">
        <Label>参考图片</Label>
        <input ref={upload} type="file" className="hidden" aria-label="上传参考图片" multiple
          accept="image/*"
          onChange={event => void addFiles(event.target.files)} />
        {files.map(file => <div key={file.id} className="flex min-w-0 items-center gap-2">
          <span className="min-w-0 flex-1 truncate text-sm" title={file.name}>{file.name}</span>
          <Button variant="ghost" size="icon" aria-label={"移除 " + file.name}
            onClick={() => setFiles(current => current.filter(item => item.id !== file.id))}><X /></Button>
        </div>)}
        <Button variant="outline" disabled={uploading || submitting} onClick={() => upload.current?.click()}>
          {uploading ? <Loader2 className="animate-spin" /> : <Paperclip />}添加参考图
        </Button>
      </div>}
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      <Button disabled={uploading || submitting} onClick={() => void submit()}>
        {submitting ? <Loader2 className="animate-spin" /> : <Play />}生成音频
      </Button>
    </div>
  );
}
