"use client";

import { useState } from "react";
import { Loader2, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { scriptApi } from "@/lib/api/script";
import { storyboardApi } from "@/lib/api/storyboard";
import { toastApiError } from "@/lib/api/toast-api-error";

interface Props {
  kind: "script" | "storyboard";
  id: number;
  version: number;
  episodeCount: number;
  sceneCount: number;
  itemCount?: number;
  disabled?: boolean;
  onCleared: () => void;
}

export function WorkspaceClearAction({
  kind,
  id,
  version,
  episodeCount,
  sceneCount,
  itemCount = 0,
  disabled,
  onCleared,
}: Props) {
  const { confirm } = useConfirm();
  const requests = useCommandRequestKey();
  const [busy, setBusy] = useState(false);
  const label = kind === "script" ? "剧本" : "分镜";
  const clear = async () => {
    const accepted = await confirm({
      title: `清空${label}`,
      description:
        kind === "script"
          ? `将清空 ${episodeCount} 集、${sceneCount} 个场次，以及原文和梗概。已有分镜和资产保留，分镜需重新关联剧本集。`
          : `将清空 ${episodeCount} 集、${sceneCount} 个场次和 ${itemCount} 个镜头及其媒体关联。剧本和项目资产保留。`,
      note: "此操作无法撤销。",
      variant: "destructive",
      confirmText: `清空${label}`,
    });
    if (!accepted) {
      return;
    }
    setBusy(true);
    try {
      const requestId = requests.forPayload({ kind, id, version });
      if (kind === "script") {
        await scriptApi.clearContent(id, version, requestId);
      } else {
        await storyboardApi.clearContent(id, version, requestId);
      }
      requests.clear();
      toast.success(`${label}已清空`);
      onCleared();
    } catch (error) {
      toastApiError(error, `清空${label}失败，请重试`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Button
      variant="destructive-ghost"
      disabled={disabled || busy}
      onClick={() => void clear()}
    >
      {busy ? <Loader2 className="animate-spin" /> : <Trash2 />}
      {busy ? "正在清空" : `清空${label}`}
    </Button>
  );
}
