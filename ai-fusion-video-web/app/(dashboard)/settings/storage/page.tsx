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
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  storageConfigApi,
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
import { containerVariants, itemVariants } from "../_shared";

// ============================================================
// 存储配置 Dialog
// ============================================================

function getStorageTypeFields(type: string) {
  switch (type) {
    case "local":
      return [
        { key: "basePath", label: "存储路径", placeholder: "./data/media", type: "text" as const, required: true },
      ];
    case "s3":
      return [
        { key: "endpoint", label: "Endpoint", placeholder: "https://oss-cn-hangzhou.aliyuncs.com", type: "text" as const, required: true },
        { key: "bucketName", label: "Bucket 名称", placeholder: "my-bucket", type: "text" as const, required: true },
        { key: "accessKey", label: "Access Key", placeholder: "LTAI...", type: "password" as const, required: true },
        { key: "secretKey", label: "Secret Key", placeholder: "...", type: "password" as const, required: true },
        { key: "region", label: "Region", placeholder: "cn-hangzhou / us-east-1", type: "text" as const },
        { key: "basePath", label: "Key 前缀", placeholder: "ai-fusion/（可选）", type: "text" as const },
        { key: "customDomain", label: "自定义域名", placeholder: "https://cdn.example.com（可选）", type: "text" as const },
      ];
    default:
      return [];
  }
}

interface StorageConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingConfig: StorageConfigType | null;
  onSaved: () => void;
}

function StorageConfigDialog({ open, onOpenChange, editingConfig, onSaved }: StorageConfigDialogProps) {
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<StorageConfigSaveReq>({ name: "", type: "local" });
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (open) {
      if (editingConfig) {
        setForm({
          id: editingConfig.id,
          name: editingConfig.name,
          type: editingConfig.type,
          endpoint: editingConfig.endpoint || "",
          bucketName: editingConfig.bucketName || "",
          accessKey: editingConfig.accessKey || "",
          secretKey: editingConfig.secretKey || "",
          region: editingConfig.region || "",
          basePath: editingConfig.basePath || "",
          customDomain: editingConfig.customDomain || "",
          isDefault: editingConfig.isDefault,
          status: editingConfig.status,
          remark: editingConfig.remark || "",
        });
      } else {
        setForm({ name: "", type: "local", basePath: "./data/media", status: 1 });
      }
      setShowSecrets({});
    }
  }, [open, editingConfig]);

  const updateField = <K extends keyof StorageConfigSaveReq>(key: K, value: StorageConfigSaveReq[K]) => {
    setForm(prev => ({ ...prev, [key]: value }));
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
    } finally {
      setSaving(false);
    }
  };

  const fields = getStorageTypeFields(form.type || "local");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
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
              onValueChange={v => updateField("type", v as string)}
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

          {/* 动态字段 */}
          {fields.map(field => (
            <div key={field.key} className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">
                {field.label}
                {field.required && <span className="text-destructive ml-0.5">*</span>}
              </Label>
              <div className="relative">
                <Input
                  type={field.type === "password" && !showSecrets[field.key] ? "password" : "text"}
                  placeholder={field.placeholder}
                  value={(form as unknown as Record<string, string>)[field.key] || ""}
                  onChange={e => updateField(field.key as keyof StorageConfigSaveReq, e.target.value)}
                  className="text-sm pr-9"
                />
                {field.type === "password" && (
                  <button
                    type="button"
                    onClick={() => setShowSecrets(prev => ({ ...prev, [field.key]: !prev[field.key] }))}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showSecrets[field.key] ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  </button>
                )}
              </div>
            </div>
          ))}

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

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button size="sm" onClick={handleSave} disabled={saving || !form.name.trim()}>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            {editingConfig ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// 主页面
// ============================================================

export default function StoragePage() {
  const [storageConfigs, setStorageConfigs] = useState<StorageConfigType[]>([]);
  const [storageLoading, setStorageLoading] = useState(true);
  const [storageDialogOpen, setStorageDialogOpen] = useState(false);
  const [editingStorageConfig, setEditingStorageConfig] = useState<StorageConfigType | null>(null);

  useEffect(() => {
    loadStorageConfigs();
  }, []);

  const loadStorageConfigs = useCallback(async () => {
    try {
      setStorageLoading(true);
      const data = await storageConfigApi.list();
      setStorageConfigs(data);
    } catch (err) {
      console.error("加载存储配置列表失败:", err);
    } finally {
      setStorageLoading(false);
    }
  }, []);

  const handleDeleteStorageConfig = async (id: number) => {
    if (!confirm("确定要删除该存储配置吗？")) return;
    try {
      await storageConfigApi.delete(id);
      await loadStorageConfigs();
    } catch (err) {
      console.error("删除存储配置失败:", err);
    }
  };

  const handleSetDefaultStorage = async (id: number) => {
    try {
      await storageConfigApi.setDefault(id);
      await loadStorageConfigs();
    } catch (err) {
      console.error("设置默认存储失败:", err);
    }
  };

  return (
    <motion.div
      className="max-w-[1200px]"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* 页面标题 */}
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">存储配置</h1>
        <p className="text-muted-foreground mt-1">
          管理文件存储后端，支持本地磁盘和 S3 兼容存储
        </p>
      </motion.div>

      {/* ========== 存储配置管理 ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <div className="flex items-center justify-between mb-3 px-1">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
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
              const isLocal = sc.type === "local";
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
                          {STORAGE_TYPE_LABELS[sc.type] || sc.type}
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
