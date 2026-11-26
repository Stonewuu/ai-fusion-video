"use client";

import type { RefObject } from "react";
import {
  ChevronsDownUp,
  ChevronsUpDown,
  Download,
  LayoutGrid,
  List,
  Plus,
  Search,
  Upload,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { SegmentedControl } from "@/components/shared/segmented-control";
import { MODEL_TYPE_OPTIONS } from "@/lib/api/ai-model";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { settingsTypography } from "../../_shared";
import type { AiModelStatusFilter } from "./ai-models-filter";

export type AiModelViewMode = "list" | "card";

const STATUS_FILTER_OPTIONS: Array<{ value: AiModelStatusFilter; label: string }> = [
  { value: "all", label: "全部状态" },
  { value: "enabled", label: "仅已启用" },
  { value: "disabled", label: "仅已停用" },
];

interface AiModelsToolbarProps {
  query: string;
  onQueryChange: (value: string) => void;
  modelTypeFilter: number;
  onModelTypeFilterChange: (value: number) => void;
  statusFilter: AiModelStatusFilter;
  onStatusFilterChange: (value: AiModelStatusFilter) => void;
  viewMode: AiModelViewMode;
  onViewModeChange: (mode: AiModelViewMode) => void;
  hasConfigs: boolean;
  allCollapsed: boolean;
  onToggleCollapseAll: () => void;
  importInputRef: RefObject<HTMLInputElement | null>;
  onImportFile: (file: File) => void;
  onOpenExport: () => void;
  onOpenCreateConfig: () => void;
  totalConfigCount: number;
  visibleConfigCount: number;
  hasActiveFilters: boolean;
  onResetFilters: () => void;
}

export function AiModelsToolbar({
  query,
  onQueryChange,
  modelTypeFilter,
  onModelTypeFilterChange,
  statusFilter,
  onStatusFilterChange,
  viewMode,
  onViewModeChange,
  hasConfigs,
  allCollapsed,
  onToggleCollapseAll,
  importInputRef,
  onImportFile,
  onOpenExport,
  onOpenCreateConfig,
  totalConfigCount,
  visibleConfigCount,
  hasActiveFilters,
  onResetFilters,
}: AiModelsToolbarProps) {
  return (
    <div className="mb-4 flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h3 className={settingsTypography.sectionTitle}>API 配置与模型</h3>
          {totalConfigCount > 0 && (
            <span className="text-xs text-muted-foreground tabular-nums">
              {hasActiveFilters
                ? `${visibleConfigCount} / ${totalConfigCount} 项服务`
                : `${totalConfigCount} 项服务`}
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <input
            ref={importInputRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                onImportFile(file);
              }
            }}
          />
          <Button
            type="button"
            variant="outline"
            onClick={() => importInputRef.current?.click()}
          >
            <Upload />
            导入
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={!hasConfigs}
            onClick={onOpenExport}
          >
            <Download />
            导出
          </Button>

          <SegmentedControl
            label="展示方式"
            value={viewMode}
            onChange={onViewModeChange}
            compact
            options={[
              { value: "list", label: "列表视图", icon: List },
              { value: "card", label: "卡片视图", icon: LayoutGrid },
            ]}
          />

          <Button type="button" onClick={onOpenCreateConfig}>
            <Plus />
            添加 API 配置
          </Button>
        </div>
      </div>

      {hasConfigs && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[220px] flex-1 sm:max-w-sm">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              aria-label="搜索服务或模型"
              placeholder="搜索服务名、平台、模型名或 ID"
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              className={cn("pl-9", query && "pr-9")}
            />
            {query && (
              <button
                type="button"
                aria-label="清空搜索"
                title="清空搜索"
                onClick={() => onQueryChange("")}
                className="absolute right-2.5 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
              >
                <X className="size-3.5" />
              </button>
            )}
          </div>

          <SegmentedControl
            label="按模型类型筛选"
            value={modelTypeFilter}
            onChange={onModelTypeFilterChange}
            options={[
              { value: 0, label: "全部" },
              ...MODEL_TYPE_OPTIONS.map(({ value, label }) => ({ value, label: label.replace("生成", "") })),
            ]}
          />

          <Select
            value={statusFilter}
            onValueChange={(value) => onStatusFilterChange(value as AiModelStatusFilter)}
            items={STATUS_FILTER_OPTIONS}
          >
            <SelectTrigger aria-label="按状态筛选" className="w-28 text-xs">
              <SelectValue placeholder="全部状态" />
            </SelectTrigger>
            <SelectContent className="text-xs">
              <SelectGroup>
                {STATUS_FILTER_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value} className="text-xs">
                    {option.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>

          <div className="ml-auto flex items-center gap-2 text-muted-foreground">
            {hasActiveFilters && (
              <Button type="button" variant="ghost" onClick={onResetFilters}>
                重置筛选
              </Button>
            )}
            <Button
              type="button"
              variant="ghost"
              onClick={onToggleCollapseAll}
              disabled={visibleConfigCount === 0}
            >
              {allCollapsed ? <ChevronsUpDown /> : <ChevronsDownUp />}
              {allCollapsed ? "全部展开" : "全部折叠"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
