"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Images, Loader2, Plus, X } from "lucide-react";
import { assetApi, type Asset, type AssetItem, type AssetWithItems } from "@/lib/api/asset";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { Button } from "@/components/ui/button";
import { AssetSidebar } from "./_components/asset-sidebar";
import { MasterAssetForm, MasterAssetHeader } from "./_components/asset-master-panels";
import { AssetItemTabs } from "./_components/asset-item-tabs";
import { AssetItemWorkspace } from "./_components/asset-item-workspace";
import { parseProperties } from "./_components/asset-types";

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
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const projectId = Number(params.id);
  const highlightId = searchParams.get("highlight");

  const [assets, setAssets] = useState<AssetWithItems[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<number | null>(null);
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [creatingAsset, setCreatingAsset] = useState(false);
  const [creatingItem, setCreatingItem] = useState(false);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [error, setError] = useState("");

  const selectedAssetIdRef = useRef<number | null>(null);

  useEffect(() => {
    selectedAssetIdRef.current = selectedAssetId;
  }, [selectedAssetId]);

  const loadAssets = useCallback(async (keepSelection = true) => {
    setLoading(true);
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
        setCreatingItem(Boolean(first && first.items.length === 0));
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "资产加载失败");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { void loadAssets(false); }, [loadAssets]);

  useEffect(() => {
    if (loading || !highlightId || assets.length === 0) return;

    const target = assets.find((asset) => String(asset.id) === highlightId);
    if (!target) return;

    setSelectedAssetId(target.id);
    setSelectedItemId(target.items?.[0]?.id ?? null);
    setCreatingItem(target.items.length === 0);
    router.replace(`/projects/${projectId}/assets`, { scroll: false });
  }, [assets, highlightId, loading, projectId, router]);

  const selectedAsset =
    assets.find((asset) => asset.id === selectedAssetId) || null;
  const selectedItem =
    selectedAsset?.items.find((item) => item.id === selectedItemId) || null;

  const selectAsset = (asset: AssetWithItems) => {
    setCreatingAsset(false);
    setCreatingItem(asset.items.length === 0);
    setSelectedAssetId(asset.id);
    setSelectedItemId(asset.items[0]?.id ?? null);
  };

  const saveAsset = async (payload: { name: string; type: string; description: string }) => {
    if (!selectedAsset || !payload.name.trim()) return;
    setSaving(true);
    try {
      await assetApi.update({
        id: selectedAsset.id,
        type: payload.type,
        name: payload.name.trim(),
        description: payload.description.trim(),
        properties: JSON.stringify({ setting: payload.description.trim() }),
      });
      await loadAssets(true);
    } finally {
      setSaving(false);
    }
  };

  const createAsset = async (name: string, type: string, description: string) => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      const created = await assetApi.create({
        projectId,
        type,
        name: name.trim(),
        description: description.trim(),
        properties: JSON.stringify({ setting: description.trim() }),
      });
      await loadAssets(false);
      setSelectedAssetId(created.id);
      setSelectedItemId(null);
      setCreatingAsset(false);
      setCreatingItem(true);
    } finally {
      setSaving(false);
    }
  };

  const saveItem = async (item: AssetItem, name: string, appearance: string, imageUrl: string) => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      await assetApi.updateItem({
        id: item.id,
        name: name.trim(),
        imageUrl: imageUrl.trim(),
        properties: JSON.stringify({
          ...parseProperties(item.properties),
          appearanceDescription: appearance.trim(),
        }),
      });
      await loadAssets(true);
    } finally {
      setSaving(false);
    }
  };

  const deleteAsset = async (asset: Asset) => {
    if (!confirm(`确定删除主资产“${asset.name}”及其子资产吗？`)) return;
    await assetApi.delete(asset.id);
    await loadAssets(false);
  };

  const deleteItem = async (item: AssetItem) => {
    if (!confirm(`确定删除子资产“${item.name || "未命名"}”吗？`)) return;
    const nextItem = selectedAsset?.items.find((candidate) => candidate.id !== item.id);
    await assetApi.deleteItem(item.id);
    await loadAssets(true);
    setSelectedItemId(nextItem?.id ?? null);
    setCreatingItem(!nextItem);
  };

  const createItem = async (
    name: string,
    appearance: string,
    imageUrl: string,
    generateAfterCreate: boolean,
  ) => {
    if (!selectedAsset || !name.trim()) return;
    setSaving(true);
    try {
      const created = await assetApi.createItem({
        assetId: selectedAsset.id,
        name: name.trim(),
        imageUrl: imageUrl.trim() || undefined,
        properties: JSON.stringify({
          appearanceDescription: appearance.trim(),
        }),
        sortOrder: selectedAsset.items.length,
      });
      await loadAssets(true);
      setSelectedItemId(created.id);
      setCreatingItem(false);
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

  if (loading && assets.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="flex min-h-[680px] w-full flex-1 flex-col gap-5">
      <header className="flex shrink-0 flex-wrap items-end justify-between gap-4 px-1">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">资产管理</h1>
          <p className="mt-1 text-sm text-muted-foreground">角色、场景与道具</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="image"
            size="sm"
            onClick={() =>
              enqueueAssetGeneration(
                projectId,
                assets.map((asset) => asset.id),
                undefined,
                "批量生成全部资产图",
              )
            }
            disabled={!assets.length}
          >
            <Images className="h-3.5 w-3.5" />
            批量生图
          </Button>
          <Button
            size="sm"
            onClick={() => {
              setCreatingAsset(true);
              setCreatingItem(false);
            }}
          >
            <Plus className="h-3.5 w-3.5" />
            新建主资产
          </Button>
        </div>
      </header>

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

      <div className="grid min-h-0 flex-1 gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
        <AssetSidebar
          assets={assets}
          selectedAssetId={selectedAssetId}
          search={search}
          typeFilter={typeFilter}
          onSearchChange={setSearch}
          onTypeFilterChange={setTypeFilter}
          onSelect={selectAsset}
          onRefresh={() => void loadAssets(true)}
        />
        <main className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-[30px] border border-border/50 bg-card/75 shadow-[0_24px_70px_-42px_rgba(15,23,42,.45)]">
          {creatingAsset ? (
            <MasterAssetForm
              saving={saving}
              onCancel={() => setCreatingAsset(false)}
              onSubmit={createAsset}
            />
          ) : selectedAsset ? (
            <>
              <MasterAssetHeader
                key={`${selectedAsset.id}-${selectedAsset.updateTime}`}
                asset={selectedAsset}
                saving={saving}
                onSave={saveAsset}
                onDelete={() => void deleteAsset(selectedAsset)}
              />
              <AssetItemTabs
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
                  setSelectedItemId(null);
                }}
                onGenerate={() =>
                  enqueueAssetGeneration(
                    projectId,
                    [selectedAsset.id],
                    undefined,
                    `生成 ${selectedAsset.name} 全部资产图`,
                  )
                }
              />
              <div className="min-h-0 flex-1 overflow-y-auto">
                {creatingItem || selectedItem ? (
                  <AssetItemWorkspace
                    key={
                      creatingItem
                        ? `create-${selectedAsset.id}`
                        : `${selectedItem?.id}-${selectedItem?.updateTime}`
                    }
                    mode={creatingItem ? "create" : "edit"}
                    item={creatingItem ? null : selectedItem}
                    assetType={selectedAsset.type}
                    saving={saving}
                    onCreate={createItem}
                    onSave={saveItem}
                    onDelete={() => {
                      if (selectedItem) void deleteItem(selectedItem);
                    }}
                    onCancelCreate={() => {
                      setCreatingItem(false);
                      setSelectedItemId(selectedAsset.items[0]?.id ?? null);
                    }}
                    onGenerate={() => {
                      if (!selectedItem) return;
                      enqueueAssetGeneration(
                        projectId,
                        [selectedAsset.id],
                        [selectedItem.id],
                        `生成 ${selectedItem.name || "子资产"}`,
                      );
                    }}
                  />
                ) : (
                  <div className="flex h-full min-h-64 flex-col items-center justify-center text-center">
                    <Images className="mb-3 h-9 w-9 text-muted-foreground/20" />
                    <p className="text-sm font-medium">选择或新增子资产</p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex h-full flex-col items-center justify-center">
              <Images className="mb-3 h-10 w-10 text-muted-foreground/20" />
              <p className="text-sm font-medium">选择主资产</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
