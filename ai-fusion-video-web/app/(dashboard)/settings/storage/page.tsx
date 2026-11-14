"use client";

import { useEffect, useState, useCallback } from "react";
import { Loader2, Plus } from "lucide-react";
import { motion } from "framer-motion";
import { toastApiError } from "@/lib/api/toast-api-error";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { storageConfigApi, type StorageConfig } from "@/lib/api/storage";
import { Button } from "@/components/ui/button";
import { containerVariants, itemVariants, settingsTypography } from "../_shared";
import { StorageConfigDialog } from "./_components/storage-config-dialog";
import { StorageConfigList } from "./_components/storage-config-list";

export default function StoragePage() {
  const { confirm } = useConfirm();
  const [storageConfigs, setStorageConfigs] = useState<StorageConfig[]>([]);
  const [builtInStorage, setBuiltInStorage] = useState<StorageConfig | null>(null);
  const [storageLoading, setStorageLoading] = useState(true);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [storageDialogOpen, setStorageDialogOpen] = useState(false);
  const [editingStorageConfig, setEditingStorageConfig] = useState<StorageConfig | null>(null);

  const loadStorageConfigs = useCallback(async () => {
    setStorageLoading(true);
    setStorageError(null);
    try {
      const data = await storageConfigApi.overview();
      setStorageConfigs(data.configurations);
      setBuiltInStorage(data.builtIn);
    } catch (error) {
      setStorageError(getApiErrorMessage(error));
    } finally {
      setStorageLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadStorageConfigs();
  }, [loadStorageConfigs]);

  const deleteStorageConfig = async (id: number) => {
    const accepted = await confirm({
      title: "删除存储配置",
      description: "确定要删除该存储配置吗？此操作不可撤销。",
      variant: "destructive",
      confirmText: "确定删除",
    });
    if (!accepted) {
      return;
    }
    try {
      await storageConfigApi.delete(id);
      await loadStorageConfigs();
    } catch (error) {
      toastApiError(error, "删除存储配置失败");
    }
  };

  const setDefaultStorage = async (id: number) => {
    try {
      await storageConfigApi.setDefault(id);
      await loadStorageConfigs();
    } catch (error) {
      toastApiError(error, "设置默认存储失败");
    }
  };

  return (
    <motion.div className="w-full" variants={containerVariants} initial={false} animate="visible">
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className={settingsTypography.pageTitle}>存储配置</h1>
        <p className={settingsTypography.pageDescription}>
          管理文件存储后端，支持本地磁盘和 S3 兼容存储
        </p>
      </motion.div>
      <motion.div variants={itemVariants} className="flex min-w-0 flex-col gap-3">
        <div className="flex items-center justify-between gap-2 px-1">
          <h3 className={settingsTypography.sectionTitle}>存储后端</h3>
          <Button
            variant="outline"
            disabled={storageLoading || !!storageError}
            onClick={() => {
              setEditingStorageConfig(null);
              setStorageDialogOpen(true);
            }}
          >
            <Plus />
            添加存储配置
          </Button>
        </div>
        {storageLoading ? (
          <div role="status" aria-label="加载存储配置" className="flex justify-center py-8">
            <Loader2 className="size-5 animate-spin text-muted-foreground" />
          </div>
        ) : storageError ? (
          <div role="alert" className="flex items-center justify-between gap-3 rounded-xl border border-border/30 bg-card p-4">
            <p className="text-sm text-destructive">{storageError}</p>
            <Button variant="outline" onClick={() => void loadStorageConfigs()}>重试</Button>
          </div>
        ) : (
          <StorageConfigList
            configurations={storageConfigs}
            builtIn={builtInStorage}
            onEdit={(config) => {
              setEditingStorageConfig(config);
              setStorageDialogOpen(true);
            }}
            onDelete={(id) => void deleteStorageConfig(id)}
            onSetDefault={(id) => void setDefaultStorage(id)}
          />
        )}
      </motion.div>
      <StorageConfigDialog
        open={storageDialogOpen}
        onOpenChange={setStorageDialogOpen}
        editingConfig={editingStorageConfig}
        isFirstConfig={storageConfigs.length === 0}
        onSaved={() => void loadStorageConfigs()}
      />
    </motion.div>
  );
}
