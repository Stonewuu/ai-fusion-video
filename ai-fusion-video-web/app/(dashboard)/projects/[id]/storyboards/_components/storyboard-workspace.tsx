"use client";

import { useCommandRequestKey } from "@/hooks/use-command-request-key";

import { Film, Loader2, Sparkles, Table2, LayoutGrid, Menu, Info } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { VideoPreviewDialog } from "@/components/dashboard/video-preview-dialog";
import { motion, useReducedMotion, type Variants } from "framer-motion";
import { storyboardApi } from "@/lib/api/storyboard";
import { StoryboardSidebar } from "./storyboard-sidebar";
import { StoryboardRefPanel } from "./storyboard-ref-panel";
import { StoryboardFrameReferenceDialog } from "./storyboard-frame-reference-dialog";
import { EditItemAssetsDialog } from "./edit-assets-dialog";
import type { useStoryboardWorkspace } from "../_hooks/use-storyboard-workspace";
import { StoryboardComposeActions } from "./storyboard-compose-actions";
import { StoryboardSceneList } from "./storyboard-scene-list";

export function StoryboardWorkspace({ workspace }: { workspace: ReturnType<typeof useStoryboardWorkspace> }) {
  const requests = useCommandRequestKey();

  const {
    projectId,
    project,
    loading,
    isRefreshingStoryboard,
    showAiConfirmDialog,
    setShowAiConfirmDialog,
    storyboard,
    scriptEpisodes,
    assetsList,
    assetLookup,
    editAssetsOpen,
    setEditAssetsOpen,
    editingItem,
    setEditingItem,
    viewMode,
    isSidebarCollapsed,
    handleSetViewMode,
    handleSetSidebarCollapsed,
    frameDialogItemId,
    frameDialogInitialType,
    sidebarSelection,
    setSidebarSelection,
    leftSheetOpen,
    setLeftSheetOpen,
    rightSheetOpen,
    setRightSheetOpen,
    sceneGroups,
    setSceneGroups,
    currentEpisode,
    currentEpisodeId,
    composedPreviewUrl,
    setComposedPreviewUrl,
    runningComposeEpisodeIds,
    submittingComposeEpisodeIds,
    activeSceneId,
    allItems,
    selectedItem,
    frameDialogItem,
    activeSceneGroup,
    handleManualRefreshStoryboard,
    handleBindScriptEpisode,
    handleSidebarInitialLoad,
    handleDeleteEpisode,
    handleDeleteScene,
    handleReorderScenes,
    handleUpdateItemFrame,
    handleCloseFrameDialog,
    handleComposeEpisodeVideo,
    handleAiStoryboard,
    handleGenerateEpisodeStoryboard,
    handleGenerateItemFrame,
    handleBatchGenerateSceneFrames,
  } = workspace;
  const shouldReduceMotion = useReducedMotion();
  const sectionVariants: Variants = {
    hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 8 },
    visible: { opacity: 1, y: 0, transition: { duration: shouldReduceMotion ? 0 : 0.35, ease: [0.25, 0.46, 0.45, 0.94] } },
  };
  // ========== 渲染 ==========

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // 固定容器缺失属于项目初始化异常，不提供运行时创建入口
  if (!storyboard) {
    return (
      <div className="rounded-xl border border-destructive/30 bg-card p-6">
        <p className="font-medium">项目分镜工作区未初始化</p>
        <p className="mt-1 text-sm text-muted-foreground">请检查项目创建流程后重试。</p>
      </div>
    );
  }

  // 有分镜：三栏布局
  return (
    <motion.div
      variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: shouldReduceMotion ? 0 : 0.04 } } }}
      initial={shouldReduceMotion ? false : "hidden"}
      animate="visible"
      className="flex min-h-0 w-full flex-1 rounded-xl border border-border/20 overflow-hidden bg-card/10"
    >
      {/* 左栏：分镜目录 */}
      <motion.div variants={sectionVariants} className="shrink-0 hidden xl:block">
      <StoryboardSidebar
        projectId={projectId}
        onRefresh={handleManualRefreshStoryboard}
        isRefreshing={isRefreshingStoryboard}
        storyboardId={storyboard.id}
        storyboardVersion={storyboard.version}
        selection={sidebarSelection}
        activeSceneId={activeSceneId}
        collapsed={isSidebarCollapsed}
        onSelect={setSidebarSelection}
        onCollapsedChange={handleSetSidebarCollapsed}
        onInitialLoad={handleSidebarInitialLoad}
        onDeleteEpisode={handleDeleteEpisode}
        onDeleteScene={handleDeleteScene}
        onReorderScenes={handleReorderScenes}
        scriptEpisodes={scriptEpisodes}
        onBindScriptEpisode={handleBindScriptEpisode}
        onGenerateEpisodeStoryboard={handleGenerateEpisodeStoryboard}
      />
  
      <Dialog open={showAiConfirmDialog} onOpenChange={setShowAiConfirmDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4 text-violet-500" />
              {sceneGroups.length > 0 ? "确认 AI 补全分镜？" : "确认 AI 生成分镜？"}
            </DialogTitle>
            <DialogDescription className="text-xs text-muted-foreground pt-1">
              {sceneGroups.length > 0
                ? "系统将基于当前剧本内容，智能分析并补全尚未生成分镜的场次与镜头。"
                : "系统将基于当前剧本内容，自动分析剧本并生成全套分镜结构与镜头信息。"}
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-xl border border-border/30 bg-muted/20 p-3 text-xs text-muted-foreground space-y-1">
            <p className="font-medium text-foreground">💡 提示</p>
            <p>
              生成任务启动后将在后台自动运行，您可在右上角任务中心实时查看进度。
            </p>
          </div>

          <DialogFooter className="gap-2">
            <Button variant="outline" size="sm" onClick={() => setShowAiConfirmDialog(false)}>
              取消
            </Button>
            <Button
              variant="ai"
              size="sm"
              onClick={() => {
                setShowAiConfirmDialog(false);
                void handleAiStoryboard();
              }}
            >
              <Sparkles className="h-3.5 w-3.5" />
              {sceneGroups.length > 0 ? "开始 AI 补全" : "开始 AI 生成"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </motion.div>

      {/* 中栏：按场次分组的分镜内容 */}
      <motion.div variants={sectionVariants} className="flex-1 flex flex-col min-w-0">
        {/* 工具栏 */}
        <div className="px-4 md:px-5 py-3 border-b border-border/20 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2 max-w-[60%]">
            <Sheet open={leftSheetOpen} onOpenChange={setLeftSheetOpen}>
              <SheetTrigger
                render={
                  <Button variant="ghost" size="icon-sm" className="xl:hidden shrink-0" aria-label="打开分镜目录">
                    <Menu className="h-5 w-5" />
                  </Button>
                }
              />
              <SheetContent side="left" className="w-[300px] p-0 border-r-0 flex flex-col pt-12">
                <StoryboardSidebar
                  projectId={projectId}
         onRefresh={handleManualRefreshStoryboard}
        isRefreshing={isRefreshingStoryboard}
        storyboardId={storyboard.id}
        storyboardVersion={storyboard.version}
                  selection={sidebarSelection}
                  activeSceneId={activeSceneId}
                  onSelect={setSidebarSelection}
                  onInitialLoad={handleSidebarInitialLoad}
                  onDeleteEpisode={handleDeleteEpisode}
                  onDeleteScene={handleDeleteScene}
                  onReorderScenes={handleReorderScenes}
                  scriptEpisodes={scriptEpisodes}
                  onBindScriptEpisode={handleBindScriptEpisode}
                  onGenerateEpisodeStoryboard={handleGenerateEpisodeStoryboard}
                />
              </SheetContent>
            </Sheet>
            <h2 className="text-base font-semibold flex items-center gap-2 overflow-hidden">
              <Film className="h-4 w-4 text-primary shrink-0" />
              <span className="truncate">{project?.name || "未命名项目"}</span>
              <span className="hidden sm:inline text-xs text-muted-foreground font-normal ml-1 shrink-0">
                · {sceneGroups.length} 场次 · {allItems.length} 镜头
              </span>
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ai" size="sm" onClick={() => setShowAiConfirmDialog(true)}>
              <Sparkles className="h-4 w-4" />
              {sceneGroups.length > 0 ? "AI 补全" : "AI 生成"}
            </Button>
            {/* 合成本集视频 */}
            <StoryboardComposeActions
              episode={currentEpisode}
              submitting={currentEpisodeId != null && submittingComposeEpisodeIds.includes(currentEpisodeId)}
              running={currentEpisodeId != null && runningComposeEpisodeIds.includes(currentEpisodeId)}
              onCompose={handleComposeEpisodeVideo}
              onPreview={setComposedPreviewUrl}
            />

            <div className="flex shrink-0 gap-2" role="group" aria-label="分镜视图">
              <Button variant={viewMode === "table" ? "secondary" : "ghost"} size="sm"
                onClick={() => handleSetViewMode("table")} aria-pressed={viewMode === "table"} title="表格视图">
                <Table2 /><span className="hidden sm:inline">表格</span>
              </Button>
              <Button variant={viewMode === "card" ? "secondary" : "ghost"} size="sm"
                onClick={() => handleSetViewMode("card")} aria-pressed={viewMode === "card"} title="卡片视图">
                <LayoutGrid /><span className="hidden sm:inline">卡片</span>
              </Button>
            </div>
            {/* 右侧边栏触发器 */}
            <Sheet open={rightSheetOpen} onOpenChange={setRightSheetOpen}>
              <SheetTrigger
                render={
                  <Button variant="ghost" size="icon-sm" className="2xl:hidden shrink-0" aria-label="打开分镜详情">
                    <Info className="h-5 w-5" />
                  </Button>
                }
              />
              <SheetContent side="right" className="w-[300px] p-0 border-l-0 flex flex-col pt-12 overflow-y-auto">
                <StoryboardRefPanel
                  storyboard={storyboard}
                  items={allItems}
                  selectedItem={selectedItem}
                  activeSceneGroup={activeSceneGroup}
                  projectId={projectId}
                  project={project}
                  assetLookup={assetLookup}
                  onUpdateFrame={handleUpdateItemFrame}
                  onGenerateFrame={handleGenerateItemFrame}
                  onBatchGenerateFrames={handleBatchGenerateSceneFrames}
                  onEditAssets={(item) => {
                    setEditingItem(item);
                    setEditAssetsOpen(true);
                  }}
                />
              </SheetContent>
            </Sheet>
          </div>
        </div>

        <StoryboardSceneList workspace={workspace} />
      </motion.div>

      {/* 右栏：引用信息 */}
      <motion.div variants={sectionVariants} className="shrink-0 hidden 2xl:block">
      <StoryboardRefPanel
        storyboard={storyboard}
        items={allItems}
        selectedItem={selectedItem}
        activeSceneGroup={activeSceneGroup}
        projectId={projectId}
        project={project}
        assetLookup={assetLookup}
        onUpdateFrame={handleUpdateItemFrame}
        onGenerateFrame={handleGenerateItemFrame}
        onBatchGenerateFrames={handleBatchGenerateSceneFrames}
        onEditAssets={(item) => {
          setEditingItem(item);
          setEditAssetsOpen(true);
        }}
      />
      </motion.div>

      <VideoPreviewDialog
        open={!!composedPreviewUrl}
        title="本集合成视频"
        videoUrl={composedPreviewUrl}
        onClose={() => setComposedPreviewUrl(null)}
      />

      <StoryboardFrameReferenceDialog
        key={`${frameDialogItemId ?? "closed"}-${frameDialogInitialType}`}
        open={frameDialogItemId !== null}
        item={frameDialogItem}
        project={project}
        initialFrameType={frameDialogInitialType}
        onClose={handleCloseFrameDialog}
        onUpdateFrame={handleUpdateItemFrame}
        onGenerateFrame={handleGenerateItemFrame}
      />

      <EditItemAssetsDialog
        open={editAssetsOpen}
        item={editingItem}
        assetsList={assetsList}
        onClose={() => {
          setEditAssetsOpen(false);
          setEditingItem(null);
        }}
        onConfirm={async ({ characterIds, sceneAssetItemId, propIds }) => {
          if (!editingItem) return;
          try {
            const payload = {
              version: editingItem.version,
              characterIds,
              sceneAssetItemId,
              propIds,
            };
            const updated = await storyboardApi.updateItemAssets(editingItem.id, payload, requests.forPayload({ itemId: editingItem.id, ...payload }));
            requests.clear();
            // 局部更新场次数据状态
            setSceneGroups((prev) =>
              prev.map((g) => ({
                ...g,
                items: g.items.map((i) => (i.id === updated.id ? updated : i)),
              }))
            );
            setEditAssetsOpen(false);
            setEditingItem(null);
          } catch (err) {
            console.error("更新关联资产失败:", err);
            throw err;
          }
        }}
      />
    </motion.div>
  );
}
