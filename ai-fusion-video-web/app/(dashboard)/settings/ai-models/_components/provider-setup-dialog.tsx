"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { useUnsavedDialog } from "@/components/settings/use-unsaved-dialog";
import {
  type AiModel,
  type ApiConfig,
  type ApiConfigSaveReq,
} from "@/lib/api/ai-model";
import {
  providerCatalogApi,
  type ProviderDefinition,
  type ProviderModelCandidate,
} from "@/lib/api/provider-catalog";
import { SponsorNotice } from "./sponsor-notice";
import { ProviderConnectionFields } from "./provider-connection-fields";
import { ProviderModelPicker } from "./provider-model-picker";
import { useProviderModelDiscovery } from "../_hooks/use-provider-model-discovery";

export interface ProviderSetupTarget {
  provider: ProviderDefinition;
  configuration: ApiConfigSaveReq;
  existingConfig?: ApiConfig;
}

interface ProviderSetupDialogProps {
  target: ProviderSetupTarget;
  onOpenChange: (open: boolean) => void;
  models: AiModel[];
  onManual: (
    initial: ApiConfigSaveReq,
    existing?: ApiConfig,
    candidate?: ProviderModelCandidate,
  ) => void;
  onSaved: () => Promise<void>;
}

export function ProviderSetupDialog({
  target,
  onOpenChange,
  models,
  onManual,
  onSaved,
}: ProviderSetupDialogProps) {
  const { provider, existingConfig } = target;
  const [configuration, setConfiguration] = useState(target.configuration);
  const [stage, setStage] = useState<"connection" | "models">(
    existingConfig ? "models" : "connection",
  );
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [invalidField, setInvalidField] = useState<"name" | "key" | null>(null);
  const [requestId, setRequestId] = useState(() => crypto.randomUUID());
  const fieldsRef = useRef<HTMLFieldSetElement>(null);
  const { result, reading, error, read, clear } = useProviderModelDiscovery();
  const waitingForInitialRead = !!existingConfig && !result && !error;
  const busy = reading || saving || waitingForInitialRead;
  const existingCodes = new Set(
    models
      .filter((model) => model.apiConfigId === configuration.id)
      .map((model) => model.code),
  );
  const close = useUnsavedDialog(
    true,
    selected.size > 0 || (!existingConfig && !!configuration.apiKey),
    busy,
    onOpenChange,
  );

  useEffect(() => {
    if (target.existingConfig) {
      void read(target.configuration);
    }
  }, [target, read]);

  const update = (patch: Partial<ApiConfigSaveReq>) => {
    setConfiguration((previous) => ({ ...previous, ...patch }));
    setSaveError(null);
    setInvalidField(null);
    clear();
    setRequestId(crypto.randomUUID());
  };

  const preview = async () => {
    const invalid = !configuration.name.trim()
      ? "name"
      : !configuration.id && !configuration.apiKey?.trim()
        ? "key"
        : null;
    if (invalid) {
      setInvalidField(invalid);
      setSaveError(invalid === "key" ? "请填写 API Key" : "请填写配置名称");
      requestAnimationFrame(() =>
        fieldsRef.current
          ?.querySelector<HTMLInputElement>('[aria-invalid="true"]')
          ?.focus(),
      );
      return;
    }
    setSaveError(null);
    if (await read(configuration)) {
      setSelected(new Set());
      setStage("models");
    }
  };

  const apply = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      const saved = await providerCatalogApi.setup(
        configuration,
        [...selected],
        requestId,
      );
      await onSaved();
      onOpenChange(false);
      toast.success("提供商配置已保存", {
        description: `新增 ${saved.createdModels} 个模型${saved.existingModels ? `，${saved.existingModels} 个已添加` : ""}。`,
      });
    } catch (failure) {
      setSaveError(failure instanceof Error ? failure.message : "保存配置失败");
    } finally {
      setSaving(false);
    }
  };

  const manual = (candidate?: ProviderModelCandidate) => {
    onOpenChange(false);
    onManual(configuration, existingConfig, candidate);
  };

  return (
    <Dialog open onOpenChange={close}>
      <DialogContent className="flex w-[calc(100%-2rem)] max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            {stage === "models" ? "选择模型" : `${provider.label} · 快速配置`}
          </DialogTitle>
          <DialogDescription>
            {existingConfig
              ? `${configuration.name} · 使用已保存的连接与密钥。`
              : stage === "models"
                ? `${configuration.name} · 勾选需要添加的模型。`
                : "填写密钥，再勾选需要的模型。"}
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="p-2">
          <fieldset
            ref={fieldsRef}
            disabled={busy}
            className="flex min-w-0 flex-col gap-4"
          >
            <SponsorNotice provider={provider} />
            {stage === "connection" && (
              <ProviderConnectionFields
                configuration={configuration}
                disabled={busy}
                invalidField={invalidField}
                onChange={update}
              />
            )}
            {stage === "models" && (reading || waitingForInitialRead) && (
              <p
                role="status"
                className="flex items-center gap-2 py-6 text-sm text-muted-foreground"
              >
                <Loader2 className="size-4 animate-spin" />
                正在读取可用模型…
              </p>
            )}
            {stage === "models" && result && (
              <>
                <p className="text-xs text-muted-foreground">{result.notice}</p>
                <ProviderModelPicker
                  models={result.models}
                  selected={selected}
                  existing={existingCodes}
                  busy={busy}
                  onConfigure={manual}
                  onChange={(ids) => {
                    setSelected(ids);
                    setRequestId(crypto.randomUUID());
                  }}
                />
              </>
            )}
          </fieldset>
        </DialogScrollBody>
        {(saveError || error) && (
          <p role="alert" className="shrink-0 text-sm text-destructive">
            {saveError || error}
          </p>
        )}
        <DialogFooter className="shrink-0 gap-2">
          {stage === "models" && !existingConfig && (
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => {
                setStage("connection");
                setSelected(new Set());
                setSaveError(null);
                clear();
              }}
            >
              <ArrowLeft />
              返回
            </Button>
          )}
          <Button variant="ghost" disabled={busy} onClick={() => manual()}>
            手动配置
          </Button>
          <Button
            disabled={
              busy || (stage === "models" && !!result && selected.size === 0)
            }
            onClick={stage === "models" && result ? apply : preview}
          >
            {busy && <Loader2 className="animate-spin" />}
            {stage === "models" && result
              ? `添加 ${selected.size} 个模型`
              : error
                ? "重试读取"
                : busy
                  ? "读取中…"
                  : "读取模型"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
