"use client";

import { useState } from "react";
import { Loader2, Sparkles } from "lucide-react";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { scriptApi } from "@/lib/api/script";

interface ParseScriptDialogProps {
  open: boolean;
  scriptId: number;
  scriptVersion: number;
  episodeCount: number;
  sceneCount: number;
  mode?: "create" | "reparse";
  onClose: () => void;
  onCreated: (script: { id: number; title: string }) => void;
}

export function ParseScriptDialog(props: ParseScriptDialogProps) {
  return props.open ? <ParseScriptForm {...props} /> : null;
}

function ParseScriptForm({ scriptId, scriptVersion, episodeCount, sceneCount, mode = "create", onClose, onCreated }: ParseScriptDialogProps) {
  const { confirm } = useConfirm();
  const [version] = useState(scriptVersion);
  const [rawContent, setRawContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requests = useCommandRequestKey();

  const submit = async () => {
    if (loading) return;
    const content = rawContent.trim();
    if (!content) { setError("请粘贴剧本原文"); return; }
    if (mode === "reparse") {
      const accepted = await confirm({
        title: "替换剧本原文",
        description: `将替换当前 ${episodeCount} 集、${sceneCount} 个场次的结构，原有引用会显示失效。已生成媒体保留，后续解析使用新原文。`,
        variant: "destructive",
        confirmText: "替换并解析",
      });
      if (!accepted) return;
    }
    setLoading(true);
    setError("");
    try {
      const result = await scriptApi.replaceSource(scriptId, version, content, requests.forPayload({ scriptId, version, content }));
      requests.clear();
      onCreated({ id: result.id, title: result.title });
      onClose();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "保存失败，原文已保留，请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open onOpenChange={(open) => { if (!open && !loading) onClose(); }}>
      <DialogContent className="max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>{mode === "reparse" ? "重新解析剧本" : "解析剧本"}</DialogTitle>
          <DialogDescription>保存原文后，由智能体生成分集、场次与对白。</DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="space-y-3">
          <div className="flex min-w-0 flex-col gap-2">
            <label htmlFor="script-source" className="text-sm font-medium">剧本原文</label>
            <Textarea
              id="script-source"
              value={rawContent}
              onChange={(event) => setRawContent(event.target.value)}
              rows={12}
              disabled={loading}
              className="min-h-64 w-full resize-y field-sizing-fixed"
              placeholder="粘贴完整剧本原文"
            />
          </div>
          {scriptVersion !== version && <p className="text-sm text-destructive">剧本已有新修改。当前原文草稿已保留，提交时将检查版本。</p>}
          {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        </DialogScrollBody>
        <DialogFooter className="shrink-0 gap-2">
          <Button variant="outline" onClick={onClose} disabled={loading}>取消</Button>
          <Button variant="ai" onClick={submit} disabled={loading || !rawContent.trim()}>
            {loading ? <Loader2 className="animate-spin" /> : <Sparkles />}
            {loading ? "保存中" : "保存并解析"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
