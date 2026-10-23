"use client";

import { useEffect, useState, useCallback } from "react";
import { Loader2, Plus, Trash2, Edit2, Star, HardDrive } from "lucide-react";
import { motion } from "framer-motion";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import { storageConfigApi, STORAGE_PROVIDER_LABELS, STORAGE_TYPE_LABELS, type StorageConfig as StorageConfigType } from "@/lib/api/storage";
import { Button } from "@/components/ui/button";
import { containerVariants, itemVariants, settingsTypography } from "../_shared";
import { normalizeStorageType } from "./_components/storage-form-options";
import { StorageConfigDialog } from "./_components/storage-config-dialog";

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
          <Button variant="outline"
            onClick={() => { setEditingStorageConfig(null); setStorageDialogOpen(true); }}
          >
            <Plus className="h-3.5 w-3.5" />
            添加存储配置
          </Button>
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
                    <div className="flex items-center gap-2 shrink-0">
                      {!sc.isDefault && (
                        <Button variant="outline"
                          onClick={() => handleSetDefaultStorage(sc.id)}
                        >
                          <Star className="h-3 w-3" />
                          设为默认
                        </Button>
                      )}
                      <Button variant="ghost" size="icon-sm" aria-label={`编辑存储 ${sc.name}`}
                        onClick={() => { setEditingStorageConfig(sc); setStorageDialogOpen(true); }}
                      >
                        <Edit2 className="h-3.5 w-3.5" />
                      </Button>
                      <Button variant="destructive-ghost" size="icon-sm" aria-label={`删除存储 ${sc.name}`}
                        onClick={() => handleDeleteStorageConfig(sc.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
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
