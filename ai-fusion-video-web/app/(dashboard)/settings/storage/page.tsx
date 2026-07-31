"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Loader2,
  Plus,
  Trash2,
  Edit2,
  Eye,
  EyeOff,
  Star,
  HardDrive,
  Cloud,
  TestTube2,
  Settings2,
  ChevronDown,
} from "lucide-react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import {
  getStorageProviderOption,
  renderEndpointTemplate,
  storageConfigApi,
  STORAGE_PROVIDER_LABELS,
  STORAGE_PROVIDER_OPTIONS,
  STORAGE_TYPE_OPTIONS,
  STORAGE_TYPE_LABELS,
  type StorageConfig as StorageConfigType,
  type StorageConfigSaveReq,
} from "@/lib/api/storage";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { containerVariants, itemVariants, settingsTypography } from "../_shared";

// ============================================================
// 存储配置 Dialog
// ============================================================

function getProviderOption(provider?: string) {
  return getStorageProviderOption(provider || "generic_s3");
}

function getProviderDefaultOptions(provider?: string) {
  const option = getProviderOption(provider);
  return {
    pathStyleAccessEnabled: option.pathStyleAccessEnabled,
    chunkedEncodingEnabled: option.chunkedEncodingEnabled,
    checksumCalculation: "WHEN_REQUIRED" as const,
  };
}

function buildEndpoint(provider?: string, region?: string) {
  return renderEndpointTemplate(provider || "generic_s3", region);
}

function hasEndpointTemplate(provider?: string) {
  return !!getProviderOption(provider).endpointTemplate;
}

function isTemplateEndpoint(provider?: string, region?: string, endpoint?: string) {
  const generated = buildEndpoint(provider, region);
  return !!generated && generated === endpoint;
}

function endpointPlaceholder(provider?: string) {
  const option = getProviderOption(provider);
  return option.endpointTemplate || "https://s3.example.com";
}

function normalizeStorageType(config: Pick<StorageConfigType, "type" | "provider" | "endpoint" | "bucketName">) {
  if (config.type === "local") return "local";
  if (config.type === "s3" || config.type === "aliyun_oss" || config.type === "tencent_cos") return "s3";
  if (config.provider || config.endpoint || config.bucketName) return "s3";
  return "local";
}

function normalizeStorageProvider(config: Pick<StorageConfigType, "type" | "provider">) {
  if (config.provider) return config.provider;
  if (config.type === "aliyun_oss") return "aliyun_oss";
  if (config.type === "tencent_cos") return "tencent_cos";
  return "generic_s3";
}

interface StorageConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingConfig: StorageConfigType | null;
  onSaved: () => void;
}

function StorageConfigDialog({ open, onOpenChange, editingConfig, onSaved }: StorageConfigDialogProps) {
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [form, setForm] = useState<StorageConfigSaveReq>({ name: "", type: "local" });
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [endpointManual, setEndpointManual] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    if (open) {
      if (editingConfig) {
        const type = normalizeStorageType(editingConfig);
        const provider = type === "s3" ? normalizeStorageProvider(editingConfig) : undefined;
        const generatedEndpoint = buildEndpoint(provider, editingConfig.region || "");
        const isManualEndpoint = type === "s3"
          && !!editingConfig.endpoint
          && (!generatedEndpoint || editingConfig.endpoint !== generatedEndpoint);
        setForm({
          id: editingConfig.id,
          name: editingConfig.name,
          type,
          provider,
          endpoint: editingConfig.endpoint || "",
          bucketName: editingConfig.bucketName || "",
          accessKey: editingConfig.accessKey || "",
          secretKey: "",
          region: editingConfig.region || "",
          basePath: editingConfig.basePath || "",
          customDomain: editingConfig.customDomain || "",
          options: provider
            ? { ...getProviderDefaultOptions(provider), ...(editingConfig.options || {}) }
            : editingConfig.options,
          isDefault: editingConfig.isDefault,
          status: editingConfig.status,
          remark: editingConfig.remark || "",
        });
        setEndpointManual(isManualEndpoint);
      } else {
        setForm({ name: "", type: "local", basePath: "./data/media", status: 1 });
        setEndpointManual(false);
      }
      setShowSecrets({});
      setShowAdvanced(false);
    }
  }, [open, editingConfig]);

  const updateField = <K extends keyof StorageConfigSaveReq>(key: K, value: StorageConfigSaveReq[K]) => {
    setForm(prev => ({ ...prev, [key]: value }));
  };

  const updateType = (type: string) => {
    setEndpointManual(false);
    setForm(prev => {
      if (type === "s3") {
        const provider = prev.provider || "aliyun_oss";
        const providerOption = getProviderOption(provider);
        const region = prev.region || providerOption.defaultRegion || "";
        const generatedEndpoint = buildEndpoint(provider, region);
        return {
          ...prev,
          type,
          provider,
          region,
          endpoint: generatedEndpoint,
          options: { ...getProviderDefaultOptions(provider), ...(prev.options || {}) },
          basePath: "",
        };
      }
      return {
        ...prev,
        type,
        provider: undefined,
        endpoint: "",
        bucketName: "",
        accessKey: "",
        secretKey: "",
        region: "",
        customDomain: "",
        options: undefined,
        basePath: prev.basePath || "./data/media",
      };
    });
  };

  const updateProvider = (provider: string) => {
    const providerOption = getProviderOption(provider);
    setEndpointManual(!providerOption.endpointTemplate);
    setForm(prev => {
      const region = prev.region || providerOption.defaultRegion || "";
      const generatedEndpoint = buildEndpoint(provider, region);
      return {
        ...prev,
        provider,
        region,
        endpoint: generatedEndpoint || "",
        options: getProviderDefaultOptions(provider),
      };
    });
  };

  const updateRegion = (region: string) => {
    setForm(prev => {
      const generatedEndpoint = buildEndpoint(prev.provider, region);
      const shouldAutoUpdate = !endpointManual && hasEndpointTemplate(prev.provider);
      const endpoint = shouldAutoUpdate ? generatedEndpoint : prev.endpoint;
      return { ...prev, region, endpoint };
    });
  };

  const setManualEndpoint = (manual: boolean) => {
    setEndpointManual(manual);
    if (!manual) {
      setForm(prev => ({ ...prev, endpoint: buildEndpoint(prev.provider, prev.region) }));
    }
  };

  const updateOptionField = <K extends keyof NonNullable<StorageConfigSaveReq["options"]>>(
    key: K,
    value: NonNullable<StorageConfigSaveReq["options"]>[K]
  ) => {
    setForm(prev => ({
      ...prev,
      options: {
        ...getProviderDefaultOptions(prev.provider),
        ...(prev.options || {}),
        [key]: value,
      },
    }));
  };

  const handleSave = async () => {
    if (!form.name.trim()) return;
    setSaving(true);
    try {
      if (editingConfig) {
        await storageConfigApi.update(form);
      } else {
        await storageConfigApi.create(form);
      }
      onSaved();
      onOpenChange(false);
    } catch (err) {
      console.error("保存存储配置失败:", err);
      toastApiError(err, "保存存储配置失败");
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      const result = await storageConfigApi.test(form);
      if (result.success) {
        toast.success(result.message || "存储连接正常", {
          description: result.publicUrl,
        });
      } else {
        toast.error(result.message || "存储连接测试失败", {
          description: result.publicUrl,
        });
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "存储连接测试失败");
    } finally {
      setTesting(false);
    }
  };

  const providerOption = getProviderOption(form.provider);
  const isS3 = form.type === "s3";
  const generatedEndpoint = buildEndpoint(form.provider, form.region);
  const endpointIsAuto = isS3 && hasEndpointTemplate(form.provider) && !endpointManual;
  const endpointStatus = isS3 && hasEndpointTemplate(form.provider)
    ? endpointIsAuto
      ? "访问端点会随地域自动更新"
      : isTemplateEndpoint(form.provider, form.region, form.endpoint)
        ? "当前访问端点与厂商模板一致"
        : "正在使用自定义访问端点，地域不会覆盖它"
    : "该厂商需要手动填写访问端点";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>{editingConfig ? "编辑存储配置" : "新建存储配置"}</DialogTitle>
          <DialogDescription>
            配置文件存储后端（本地磁盘 / S3 兼容存储）
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 overflow-y-auto min-h-0 px-1 -mx-1">
          {/* 配置名称 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">配置名称</Label>
            <Input
              placeholder="例如：本地存储 / 阿里云 OSS"
              value={form.name}
              onChange={e => updateField("name", e.target.value)}
              className="text-sm"
            />
          </div>

          {/* 存储类型 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">存储类型</Label>
            <Select
              value={form.type || "local"}
              onValueChange={v => updateType(v as string)}
              items={STORAGE_TYPE_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
            >
              <SelectTrigger className="w-full text-sm">
                <SelectValue placeholder="选择存储类型" />
              </SelectTrigger>
              <SelectContent className="text-sm">
                <SelectGroup>
                  {STORAGE_TYPE_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      <div>
                        <div>{opt.label}</div>
                        <div className="text-[10px] text-muted-foreground">{opt.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          {isS3 && (
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">厂商</Label>
              <Select
                value={form.provider || "generic_s3"}
                onValueChange={v => updateProvider(v as string)}
                items={STORAGE_PROVIDER_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
              >
                <SelectTrigger className="w-full text-sm">
                  <SelectValue placeholder="选择厂商" />
                </SelectTrigger>
                <SelectContent className="text-sm">
                  <SelectGroup>
                    {STORAGE_PROVIDER_OPTIONS.map(opt => (
                      <SelectItem key={opt.value} value={opt.value} className="text-sm">
                        <div>
                          <div className="flex items-center gap-1.5">
                            <Cloud className="h-3.5 w-3.5 text-muted-foreground" />
                            {opt.label}
                          </div>
                          <div className="text-[10px] text-muted-foreground">{opt.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          )}

          {!isS3 ? (
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">
                存储路径<span className="text-destructive ml-0.5">*</span>
              </Label>
              <Input
                placeholder="./data/media"
                value={form.basePath || ""}
                onChange={e => updateField("basePath", e.target.value)}
                className="text-sm"
              />
            </div>
          ) : (
            <>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">地域 (Region)</Label>
                  <Input
                    placeholder={providerOption.defaultRegion || "cn-hangzhou / ap-shanghai"}
                    value={form.region || ""}
                    onChange={e => updateRegion(e.target.value)}
                    className="text-sm"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">
                    存储桶名称 (Bucket)<span className="text-destructive ml-0.5">*</span>
                  </Label>
                  <Input
                    placeholder="my-bucket"
                    value={form.bucketName || ""}
                    onChange={e => updateField("bucketName", e.target.value)}
                    className="text-sm"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-3">
                  <Label className="text-xs text-muted-foreground">
                    访问端点 (Endpoint){providerOption.endpointRequired && <span className="text-destructive ml-0.5">*</span>}
                  </Label>
                  {hasEndpointTemplate(form.provider) && (
                    <button
                      type="button"
                      onClick={() => setManualEndpoint(!endpointManual)}
                      className="text-[11px] text-primary hover:text-primary/80 transition-colors"
                    >
                      {endpointManual ? "使用自动生成" : "自定义端点"}
                    </button>
                  )}
                </div>
                {endpointIsAuto ? (
                  <div className={cn(
                    "min-h-9 rounded-4xl border border-border/60 bg-muted/35 px-3 py-2",
                    "font-mono text-xs text-muted-foreground break-all"
                  )}>
                    {generatedEndpoint || "填写地域后自动生成"}
                  </div>
                ) : (
                  <Input
                    placeholder={endpointPlaceholder(form.provider)}
                    value={form.endpoint || ""}
                    onChange={e => updateField("endpoint", e.target.value)}
                    className="text-sm"
                  />
                )}
                <p className="text-[11px] text-muted-foreground">{endpointStatus}</p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">
                    访问密钥 ID (Access Key)<span className="text-destructive ml-0.5">*</span>
                  </Label>
                  <div className="relative">
                    <Input
                      type={showSecrets.accessKey ? "text" : "password"}
                      placeholder="请输入 Access Key ID"
                      value={form.accessKey || ""}
                      onChange={e => updateField("accessKey", e.target.value)}
                      className="text-sm pr-9"
                    />
                    <button
                      type="button"
                      onClick={() => setShowSecrets(prev => ({ ...prev, accessKey: !prev.accessKey }))}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {showSecrets.accessKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">
                    访问密钥 Secret (Secret Key){!editingConfig && <span className="text-destructive ml-0.5">*</span>}
                  </Label>
                  <div className="relative">
                    <Input
                      type={showSecrets.secretKey ? "text" : "password"}
                      placeholder={editingConfig ? "留空则保留原值" : "请输入 Secret Access Key"}
                      value={form.secretKey || ""}
                      onChange={e => updateField("secretKey", e.target.value)}
                      className="text-sm pr-9"
                    />
                    <button
                      type="button"
                      onClick={() => setShowSecrets(prev => ({ ...prev, secretKey: !prev.secretKey }))}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {showSecrets.secretKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border border-border/40 bg-muted/15">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(prev => !prev)}
                  className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left"
                >
                  <span className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <Settings2 className="h-3.5 w-3.5" />
                    更多设置
                    <span className="hidden sm:inline text-[11px] font-normal text-muted-foreground/70">
                      默认即可，兼容异常或自建 S3 时再修改
                    </span>
                  </span>
                  <ChevronDown className={cn(
                    "h-3.5 w-3.5 text-muted-foreground transition-transform",
                    showAdvanced && "rotate-180"
                  )} />
                </button>
                {showAdvanced && (
                  <div className="space-y-3 border-t border-border/40 p-3">
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1.5">
                        <Label className="text-xs text-muted-foreground">对象路径前缀</Label>
                        <Input
                          placeholder="ai-fusion/（可选）"
                          value={form.basePath || ""}
                          onChange={e => updateField("basePath", e.target.value)}
                          className="text-sm"
                        />
                      </div>
                      <div className="space-y-1.5">
                        <Label className="text-xs text-muted-foreground">公开域名/CDN</Label>
                        <Input
                          placeholder="https://cdn.example.com（可选）"
                          value={form.customDomain || ""}
                          onChange={e => updateField("customDomain", e.target.value)}
                          className="text-sm"
                        />
                      </div>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <label className="flex items-center justify-between gap-3 cursor-pointer select-none rounded-lg bg-muted/20 px-3 py-2">
                        <span className="text-xs text-muted-foreground">路径样式访问 (Path-style Access)</span>
                        <Checkbox
                          checked={!!form.options?.pathStyleAccessEnabled}
                          onCheckedChange={(checked) => updateOptionField("pathStyleAccessEnabled", !!checked)}
                        />
                      </label>
                      <label className="flex items-center justify-between gap-3 cursor-pointer select-none rounded-lg bg-muted/20 px-3 py-2">
                        <span className="text-xs text-muted-foreground">分块编码 (Chunked Encoding)</span>
                        <Checkbox
                          checked={!!form.options?.chunkedEncodingEnabled}
                          onCheckedChange={(checked) => updateOptionField("chunkedEncodingEnabled", !!checked)}
                        />
                      </label>
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs text-muted-foreground">签名地域 (Signing Region)</Label>
                      <Input
                        placeholder={form.region || providerOption.defaultRegion || "us-east-1"}
                        value={form.options?.signingRegion || ""}
                        onChange={e => updateOptionField("signingRegion", e.target.value)}
                        className="text-sm"
                      />
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {/* 备注 */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">备注</Label>
            <Input
              placeholder="可选备注信息"
              value={form.remark || ""}
              onChange={e => updateField("remark", e.target.value)}
              className="text-sm"
            />
          </div>
        </div>

        <DialogFooter className="shrink-0 sm:justify-between">
          <Button
            variant="outline"
            size="sm"
            onClick={handleTest}
            disabled={testing || saving || !form.name.trim()}
          >
            {testing ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : <TestTube2 className="h-3.5 w-3.5 mr-1.5" />}
            测试连接
          </Button>
          <div className="flex items-center justify-end gap-2">
            <DialogClose render={<Button variant="outline" size="sm" />}>
              取消
            </DialogClose>
            <Button size="sm" onClick={handleSave} disabled={saving || testing || !form.name.trim()}>
              {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
              {editingConfig ? "保存" : "创建"}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// 主页面
// ============================================================

export default function StoragePage() {
  const { confirm } = useConfirm();
  const [storageConfigs, setStorageConfigs] = useState<StorageConfigType[]>([]);
  const [storageLoading, setStorageLoading] = useState(true);
  const [storageDialogOpen, setStorageDialogOpen] = useState(false);
  const [editingStorageConfig, setEditingStorageConfig] = useState<StorageConfigType | null>(null);

  const loadStorageConfigs = useCallback(async () => {
    try {
      setStorageLoading(true);
      const data = await storageConfigApi.list();
      setStorageConfigs(data);
    } catch (err) {
      console.error("加载存储配置列表失败:", err);
      toastApiError(err, "加载存储配置列表失败");
    } finally {
      setStorageLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStorageConfigs();
  }, [loadStorageConfigs]);

  const handleDeleteStorageConfig = async (id: number) => {
    const ok = await confirm({ title: "删除存储配置", description: "确定要删除该存储配置吗？此操作不可撤销。", variant: "destructive", confirmText: "确定删除" }); if (!ok) return;
    try {
      await storageConfigApi.delete(id);
      await loadStorageConfigs();
    } catch (err) {
      console.error("删除存储配置失败:", err);
      toastApiError(err, "删除存储配置失败");
    }
  };

  const handleSetDefaultStorage = async (id: number) => {
    try {
      await storageConfigApi.setDefault(id);
      await loadStorageConfigs();
    } catch (err) {
      console.error("设置默认存储失败:", err);
      toastApiError(err, "设置默认存储失败");
    }
  };

  return (
    <motion.div
      className="w-full"
      variants={containerVariants}
      initial={false}
      animate="visible"
    >
      {/* 页面标题 */}
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className={settingsTypography.pageTitle}>存储配置</h1>
        <p className={settingsTypography.pageDescription}>
          管理文件存储后端，支持本地磁盘和 S3 兼容存储
        </p>
      </motion.div>

      {/* ========== 存储配置管理 ========== */}
      <motion.div variants={itemVariants}>
        <div className="flex items-center justify-between mb-3 px-1">
          <h3 className={settingsTypography.sectionTitle}>
            存储后端
          </h3>
          <button
            onClick={() => { setEditingStorageConfig(null); setStorageDialogOpen(true); }}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium",
              "border border-dashed border-border/40 hover:border-primary/50",
              "text-muted-foreground hover:text-primary",
              "transition-all duration-200"
            )}
          >
            <Plus className="h-3.5 w-3.5" />
            添加存储配置
          </button>
        </div>
        <div className="space-y-3">
          {storageLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : storageConfigs.length === 0 ? (
            <div className={cn(
              "rounded-xl border border-dashed border-border/30 py-10 text-center",
              "bg-card/30"
            )}>
              <HardDrive className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">还没有存储配置</p>
              <p className="text-xs text-muted-foreground/60 mt-1">点击上方「添加存储配置」开始</p>
            </div>
          ) : (
            storageConfigs.map((sc) => {
              const isLocal = normalizeStorageType(sc) === "local";
              const providerLabel = !isLocal
                ? STORAGE_PROVIDER_LABELS[sc.provider || "generic_s3"] || sc.provider || "S3 兼容"
                : STORAGE_TYPE_LABELS[sc.type] || sc.type;
              return (
                <div
                  key={sc.id}
                  className={cn(
                    "rounded-xl border overflow-hidden transition-colors",
                    "bg-card/50 backdrop-blur-sm",
                    "border-border/30"
                  )}
                >
                  <div className="flex items-center gap-3 px-4 py-3 group">
                    <div className={cn(
                      "h-9 w-9 rounded-lg flex items-center justify-center shrink-0",
                      isLocal ? "bg-green-500/10" : "bg-sky-500/10"
                    )}>
                      <HardDrive className={cn(
                        "h-4.5 w-4.5",
                        isLocal ? "text-green-400" : "text-sky-400"
                      )} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium">{sc.name}</p>
                        <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px] text-muted-foreground">
                          {providerLabel}
                        </span>
                        {sc.isDefault && (
                          <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-primary/10 text-[10px] text-primary font-medium">
                            <Star className="h-2.5 w-2.5" />
                            默认
                          </span>
                        )}
                        <div className={cn(
                          "w-1.5 h-1.5 rounded-full shrink-0",
                          sc.status === 1 ? "bg-green-400" : "bg-muted-foreground/30"
                        )} />
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                        {isLocal && sc.basePath && (
                          <span className="font-mono text-[10px]">{sc.basePath}</span>
                        )}
                        {!isLocal && sc.endpoint && (
                          <span className="font-mono text-[10px]">{sc.endpoint}</span>
                        )}
                        {!isLocal && sc.bucketName && (
                          <span className="font-mono text-[10px]">/ {sc.bucketName}</span>
                        )}
                      </div>
                    </div>
                    {/* 右侧操作区 */}
                    <div className="flex items-center gap-1.5 shrink-0">
                      {!sc.isDefault && (
                        <button
                          onClick={() => handleSetDefaultStorage(sc.id)}
                          className={cn(
                            "flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all",
                            "border border-amber-500/30 text-amber-500",
                            "hover:bg-amber-500/10 hover:border-amber-500/50"
                          )}
                        >
                          <Star className="h-3 w-3" />
                          设为默认
                        </button>
                      )}
                      <button
                        onClick={() => { setEditingStorageConfig(sc); setStorageDialogOpen(true); }}
                        className="p-1.5 rounded-md text-muted-foreground/40 hover:text-primary hover:bg-primary/10 transition-colors opacity-0 group-hover:opacity-100"
                      >
                        <Edit2 className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => handleDeleteStorageConfig(sc.id)}
                        className="p-1.5 rounded-md text-muted-foreground/40 hover:text-destructive hover:bg-destructive/10 transition-colors opacity-0 group-hover:opacity-100"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </motion.div>

      {/* Dialog */}
      <StorageConfigDialog
        open={storageDialogOpen}
        onOpenChange={setStorageDialogOpen}
        editingConfig={editingStorageConfig}
        onSaved={() => { loadStorageConfigs(); }}
      />
    </motion.div>
  );
}
