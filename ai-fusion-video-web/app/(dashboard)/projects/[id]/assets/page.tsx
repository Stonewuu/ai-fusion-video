"use client";
import { useResourceInvalidation } from "@/hooks/use-resource-invalidation";

import { useConfirm } from "@/components/ui/confirm-dialog";
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Images, Loader2, Plus, X } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { toast } from "sonner";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import {
  assetApi,
  type Asset,
  type AssetItem,
  type AssetWithItems,
} from "@/lib/api/asset";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { Button } from "@/components/ui/button";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { AssetSidebar } from "./_components/asset-sidebar";
import { AssetDetailPanel } from "./_components/asset-detail-panel";
import {
  MasterAssetForm,
  MasterAssetHeader,
} from "./_components/asset-master-panels";
import { AssetItemTabs } from "./_components/asset-item-tabs";
import { AssetItemWorkspace } from "./_components/asset-item-workspace";
import { AssetItemPreview } from "./_components/asset-item-preview";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { parseProperties } from "./_components/asset-types";
import { AssetPanelTransition } from "./_components/asset-panel-transition";
import {
  assetPageVariants,
  assetSectionVariants,
} from "./_components/asset-page-motion";

function enqueueAssetGeneration(
  projectId: number,
  assetIds: number[],
  itemIds?: number[],
  label = "批量生成资产图",
) {
  const { addPipeline, setNotificationOpen } = usePipelineStore.getState();
  addPipeline({
    label,
    projectId,
    request: {
      agentType: "asset_image_gen",
      toolExecutionMode: "FULL_ACCESS",
      projectId,
      context: {
        selectedAssetIds: assetIds,
        ...(itemIds?.length ? { selectedAssetItemIds: itemIds } : {}),
      },
    },
  });
  setNotificationOpen(true);
}

export default function ProjectAssetsPage() {
  const reduceMotion = useReducedMotion();
  const { confirm } = useConfirm();
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const projectId = Number(params.id);
  const highlightId = searchParams.get("highlight");

  const [assets, setAssets] = useState<AssetWithItems[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<number | null>(null);
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [initialLoading, setInitialLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [itemUploading, setItemUploading] = useState(false);
  const [creatingAsset, setCreatingAsset] = useState(false);
  const [editingAsset, setEditingAsset] = useState(false);
  const [creatingItem, setCreatingItem] = useState(false);
  const [editingItem, setEditingItem] = useState<"details" | "image" | null>(
    null,
  );
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [detailOpen, setDetailOpen] = useState(false);
  const [error, setError] = useState("");

  const selectedAssetIdRef = useRef<number | null>(null);
  const showCreatedAssetRef = useRef(false);

  useEffect(() => {
    selectedAssetIdRef.current = selectedAssetId;
  }, [selectedAssetId]);

  const loadAssets = useCallback(
    async (keepSelection = true, background = false) => {
      if (!background) {
        setLoading(true);
      }
      try {
        const data = await assetApi.listWithItems(projectId);
        setAssets(data);
        if (
          !keepSelection ||
          !data.some((asset) => asset.id === selectedAssetIdRef.current)
        ) {
          const first = data[0];
          setSelectedAssetId(first?.id ?? null);
          setSelectedItemId(first?.items?.[0]?.id ?? null);
          setCreatingItem(false);
        }
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "资产加载失败");
      } finally {
        setLoading(false);
        setInitialLoading(false);
      }
    },
    [projectId],
  );

  useEffect(() => {
    void loadAssets(false);
  }, [loadAssets]);
  useResourceInvalidation("assets", () => loadAssets(true, true));

  useEffect(() => {
    if (loading || !highlightId || assets.length === 0) {
      return;
    }

    const target = assets.find((asset) => String(asset.id) === highlightId);
    if (!target) {
      return;
    }

    setSelectedAssetId(target.id);
    setSelectedItemId(target.items?.[0]?.id ?? null);
    setCreatingItem(false);
    setDetailOpen(true);
    router.replace(`/projects/${projectId}/assets`, { scroll: false });
  }, [assets, highlightId, loading, projectId, router]);

  const selectedAsset =
    assets.find((asset) => asset.id === selectedAssetId) || null;
  const selectedItem =
    selectedAsset?.items.find((item) => item.id === selectedItemId) || null;

  const selectAsset = (asset: AssetWithItems) => {
    setCreatingAsset(false);
    setCreatingItem(false);
    setSelectedAssetId(asset.id);
    if (asset.id !== selectedAssetId) {
      setSelectedItemId(asset.items[0]?.id ?? null);
    }
    setDetailOpen(true);
  };

  const requests = useCommandRequestKey();

  const saveAsset = async (
    base: AssetWithItems,
    payload: { name: string; type: string; description: string },
  ) => {
    if (!base || !payload.name.trim()) {
      return;
    }
    setSaving(true);
    try {
      const data = {
        id: base.id,
        version: base.version,
        type: payload.type,
        name: payload.name.trim(),
        description: payload.description.trim(),
        properties: { ...base.properties, setting: payload.description.trim() },
      };
      await assetApi.update(
        data,
        requests.forPayload({ operation: "update", ...data }),
      );
      requests.clear();
      await loadAssets(true);
    } finally {
      setSaving(false);
    }
  };

  const createAsset = async (
    name: string,
    type: string,
    description: string,
  ) => {
    if (!name.trim()) {
      return;
    }
    setSaving(true);
    try {
      const data = {
        projectId,
        type,
        name: name.trim(),
        description: description.trim(),
        properties: { setting: description.trim() },
      };
      const created = await assetApi.create(
        data,
        requests.forPayload({ operation: "create", ...data }),
      );
      requests.clear();
      await loadAssets(false);
      setSelectedAssetId(created.id);
      setSelectedItemId(created.items[0]?.id ?? null);
      setCreatingItem(false);
      showCreatedAssetRef.current = true;
    } finally {
      setSaving(false);
    }
  };

  const saveItem = async (
    item: AssetItem,
    name: string,
    appearance: string,
    imageUrl: string,
  ) => {
    if (!name.trim()) {
      throw new Error("名称不能为空");
    }
    setSaving(true);
    try {
      const data = {
        id: item.id,
        version: item.version,
        assetVersion: item.assetVersion,
        name: name.trim(),
        imageUrl: imageUrl.trim(),
        properties: {
          ...parseProperties(item.properties),
          appearanceDescription: appearance.trim(),
        },
      };
      const updated = await assetApi.updateItem(
        data,
        requests.forPayload({ operation: "updateItem", ...data }),
      );
      requests.clear();
      await loadAssets(true);
      return updated;
    } finally {
      setSaving(false);
    }
  };

  const deleteAsset = async (asset: Asset) => {
    const ok = await confirm({
      title: "删除主资产",
      description: `确定删除主资产“${asset.name}”及其所有子资产吗？已有引用会显示失效，媒体文件会保留。`,
      variant: "destructive",
      confirmText: "确定删除",
    });
    if (!ok) {
      return;
    }
    await assetApi.delete(
      asset.id,
      asset.version,
      requests.forPayload({
        operation: "delete",
        id: asset.id,
        version: asset.version,
      }),
    );
    requests.clear();
    setDetailOpen(false);
    await loadAssets(false);
  };

  const deleteItem = async (item: AssetItem) => {
    const ok = await confirm({
      title: "删除子资产",
      description: `确定删除子资产“${item.name || "未命名"}”吗？已有引用会显示失效，媒体文件会保留。`,
      variant: "destructive",
      confirmText: "确定删除",
    });
    if (!ok) {
      return;
    }
    const nextItem = selectedAsset?.items.find(
      (candidate) => candidate.id !== item.id,
    );
    await assetApi.deleteItem(
      item.id,
      item.version,
      item.assetVersion,
      requests.forPayload({
        operation: "deleteItem",
        id: item.id,
        version: item.version,
        parentVersion: item.assetVersion,
      }),
    );
    requests.clear();
    await loadAssets(true);
    setSelectedItemId(nextItem?.id ?? null);
    setCreatingItem(false);
  };

  const createItem = async (
    expectedParentVersion: number,
    name: string,
    appearance: string,
    imageUrl: string,
    generateAfterCreate: boolean,
  ) => {
    if (!selectedAsset || !name.trim()) {
      return;
    }
    setSaving(true);
    try {
      const data = {
        assetId: selectedAsset.id,
        parentVersion: expectedParentVersion,
        name: name.trim(),
        imageUrl: imageUrl.trim() || undefined,
        properties: { appearanceDescription: appearance.trim() },
        sortOrder: selectedAsset.items.length,
      };
      const created = await assetApi.createItem(
        data,
        requests.forPayload({ operation: "createItem", ...data }),
      );
      requests.clear();
      await loadAssets(true);
      setSelectedItemId(created.id);
      if (generateAfterCreate) {
        enqueueAssetGeneration(
          projectId,
          [selectedAsset.id],
          [created.id],
          `生成 ${created.name || name.trim()}`,
        );
      }
    } finally {
      setSaving(false);
    }
  };

  const generateSelectedItem = () => {
    if (!selectedAsset || !selectedItem) {
      return;
    }
    enqueueAssetGeneration(
      projectId,
      [selectedAsset.id],
      [selectedItem.id],
      `生成 ${selectedItem.name || "子资产"}`,
    );
  };

  if (initialLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <motion.div
      initial={reduceMotion ? false : "hidden"}
      animate="visible"
      variants={assetPageVariants}
      className="flex min-h-0 w-full grow basis-0 flex-col gap-4 lg:gap-6"
    >
      <motion.header
        variants={assetSectionVariants}
        className="flex shrink-0 flex-wrap items-end justify-between gap-4 px-1"
      >
        <div>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            资产管理
          </h1>
          <p className="mt-1 text-xs text-muted-foreground sm:text-sm">
            角色、场景与道具
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            disabled={saving || itemUploading}
            onClick={() => {
              setCreatingAsset(true);
              setCreatingItem(false);
            }}
          >
            <Plus />
            新建主资产
          </Button>
        </div>
      </motion.header>

      {error && (
        <div className="flex items-center justify-between rounded-xl border border-destructive/20 bg-destructive/8 px-3 py-2 text-xs text-destructive">
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError("")}
            title="关闭"
            aria-label="关闭"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      <motion.div
        variants={assetSectionVariants}
        className="grid min-h-0 grow basis-0 gap-4 lg:grid-cols-[300px_minmax(0,1fr)] lg:gap-6"
      >
        <AssetSidebar
          disabled={saving || itemUploading}
          assets={assets}
          selectedAssetId={selectedAssetId}
          search={search}
          typeFilter={typeFilter}
          onSearchChange={setSearch}
          onTypeFilterChange={setTypeFilter}
          onSelect={selectAsset}
          onBatchGenerate={() =>
            enqueueAssetGeneration(
              projectId,
              assets.map((asset) => asset.id),
              undefined,
              "批量生成全部资产图",
            )
          }
          onRefresh={() => void loadAssets(true)}
        />
        <AssetDetailPanel
          open={detailOpen}
          editing={editingAsset || creatingItem || editingItem !== null}
          onOpenChange={setDetailOpen}
        >
          <AssetPanelTransition
            panelKey={selectedAsset?.id ?? "empty"}
            className="flex min-h-0 flex-1 flex-col"
          >
            {selectedAsset ? (
              <OverlayScrollArea
                className="min-h-0 flex-1"
                viewportClassName="scroll-p-1"
              >
                <MasterAssetHeader
                  disabled={itemUploading}
                  key={selectedAsset.id}
                  asset={selectedAsset}
                  saving={saving}
                  onEditingChange={setEditingAsset}
                  onSave={saveAsset}
                  onDelete={(base) =>
                    void deleteAsset(base).catch((cause) =>
                      toast.error(getApiErrorMessage(cause)),
                    )
                  }
                />
                <AssetItemTabs
                  disabled={saving || itemUploading}
                  items={selectedAsset.items}
                  assetType={selectedAsset.type}
                  selectedItemId={selectedItemId}
                  creating={creatingItem}
                  onSelect={(id) => {
                    setCreatingItem(false);
                    setSelectedItemId(id);
                  }}
                  onCreate={() => {
                    setCreatingItem(true);
                  }}
                  onGenerate={() =>
                    enqueueAssetGeneration(
                      projectId,
                      [selectedAsset.id],
                      selectedAsset.items.map((item) => item.id),
                      `批量生成 ${selectedAsset.name} 的子资产图`,
                    )
                  }
                />
                <AssetPanelTransition panelKey={selectedItem?.id ?? "empty"}>
                  {selectedItem ? (
                    <AssetItemPreview
                      item={selectedItem}
                      disabled={saving || itemUploading}
                      onEdit={() => setEditingItem("details")}
                      onEditImage={() => setEditingItem("image")}
                      onDelete={() =>
                        void deleteItem(selectedItem).catch((cause) =>
                          toast.error(getApiErrorMessage(cause)),
                        )
                      }
                      onGenerate={generateSelectedItem}
                    />
                  ) : (
                    <div className="flex h-full min-h-64 flex-col items-center justify-center text-center">
                      <Images className="mb-3 h-9 w-9 text-muted-foreground/20" />
                      <p className="text-sm font-medium">选择或新增子资产</p>
                    </div>
                  )}
                </AssetPanelTransition>
              </OverlayScrollArea>
            ) : (
              <div className="flex h-full flex-col items-center justify-center">
                <Images className="mb-3 h-10 w-10 text-muted-foreground/20" />
                <p className="text-sm font-medium">选择主资产</p>
              </div>
            )}
          </AssetPanelTransition>
          {selectedAsset && (creatingItem || (editingItem && selectedItem)) && (
            <AssetItemWorkspace
              key={creatingItem ? `create-${selectedAsset.id}` : selectedItem?.id}
              mode={creatingItem ? "create" : "edit"}
              initialSection={editingItem === "image" ? "image" : "details"}
              item={creatingItem ? null : selectedItem}
              assetType={selectedAsset.type}
              parentVersion={selectedAsset.version}
              saving={saving}
              onUploadingChange={setItemUploading}
              onCreate={createItem}
              onSave={saveItem}
              onClose={() => {
                setCreatingItem(false);
                setEditingItem(null);
              }}
              onGenerate={generateSelectedItem}
            />
          )}
        </AssetDetailPanel>
      </motion.div>
      {creatingAsset && (
        <MasterAssetForm
          saving={saving}
          onCancel={() => {
            setCreatingAsset(false);
            if (showCreatedAssetRef.current) {
              showCreatedAssetRef.current = false;
              setDetailOpen(true);
            }
          }}
          onSubmit={createAsset}
        />
      )}
    </motion.div>
  );
}
