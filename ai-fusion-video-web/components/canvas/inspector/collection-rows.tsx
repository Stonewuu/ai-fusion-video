"use client";
import { ArrowUp, ArrowDown, ArrowUpRight, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { SafeImage } from "@/components/media/managed-image";
import { resourceMedia } from "@/lib/canvas/resource-media";
import { canvasTypeNames } from "@/lib/canvas/labels";
import { resourceKey, resourceTitle } from "@/lib/canvas/resource-presentation";
import type { CanvasCollectionEntry } from "@/lib/api/canvas";
import type { CanvasResource } from "@/lib/canvas/types";
import type { CanvasLibraryEntry } from "@/lib/canvas/resource-library";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { CanvasResourceRead } from "../forms/resource-editor";

export type CollectionMode = "table" | "cards" | "read";
export function CanvasCollectionRows({
  entries,
  mode,
  selected,
  onSelect,
  onOpen,
  onPlace,
  onPreview,
  onMove,
  disabled,
  library,
  emptyMessage,
}: {
  entries: CanvasCollectionEntry[];
  mode: CollectionMode;
  selected: string[];
  onSelect: (key: string, checked: boolean) => void;
  onOpen: (resource: CanvasResource) => void;
  onPlace: (resources: CanvasResource[]) => void;
  onPreview: (media: CanvasMediaPreview) => void;
  onMove?: (index: number, direction: -1 | 1) => void;
  disabled: boolean;
  library: CanvasLibraryEntry[];
  emptyMessage: string;
}) {
  const actions = (entry: CanvasCollectionEntry, index: number) => (
    <div className="flex gap-2">
      <Button size="xs" variant="ghost" onClick={() => onOpen(entry)}>
        打开
      </Button>
      <Button
        size="icon-xs"
        variant="ghost"
        title={`将${resourceTitle(entry)}加入画布`}
        disabled={disabled}
        onClick={() => onPlace([entry])}
      >
        <ArrowUpRight />
      </Button>
      {onMove && (
        <>
          <Button
            size="icon-xs"
            variant="ghost"
            title="向前排序"
            disabled={disabled || index === 0}
            onClick={() => onMove(index, -1)}
          >
            <ArrowUp />
          </Button>
          <Button
            size="icon-xs"
            variant="ghost"
            title="向后排序"
            disabled={disabled || index === entries.length - 1}
            onClick={() => onMove(index, 1)}
          >
            <ArrowDown />
          </Button>
        </>
      )}
    </div>
  );
  const checkbox = (entry: CanvasCollectionEntry) => (
    <Checkbox
      aria-label={`选择${resourceTitle(entry)}`}
      checked={selected.includes(resourceKey(entry))}
      onCheckedChange={(checked) =>
        onSelect(resourceKey(entry), checked === true)
      }
    />
  );
  const media = (entry: CanvasCollectionEntry) =>
    resourceMedia(entry).map((item) => ({
      ...item,
      title: `${resourceTitle(entry)} · ${item.title}`,
    }));
  const showMedia = entries.some((entry) => media(entry).length > 0);
  if (!entries.length)
    return (
      <p className="p-4 text-sm text-muted-foreground">
        {emptyMessage}
      </p>
    );
  if (mode === "table")
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm" aria-label="项目内容列表">
          <thead>
            <tr className="border-b border-border/20">
              <th className="p-3">选择</th>
              <th className="p-3">内容</th>
              {showMedia && <th className="p-3">素材</th>}
              <th className="p-3">操作</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, index) => (
              <tr
                key={resourceKey(entry)}
                className="border-b border-border/20 hover:bg-muted/30"
              >
                <td className="p-3">{checkbox(entry)}</td>
                <td className="min-w-48 p-3">
                  <p className="font-medium">{resourceTitle(entry)}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {canvasTypeNames[entry.header.resourceType]}
                    {entry.data.duration != null
                      ? ` · ${entry.data.duration} 秒`
                      : ""}
                  </p>
                </td>
                {showMedia && (
                  <td className="p-3">
                    <div className="flex flex-wrap gap-2">
                      {media(entry).map((item) => (
                        <Button
                          key={item.title}
                          variant="ghost"
                          size="icon-xs"
                          title={item.title}
                          onClick={() => onPreview(item)}
                        >
                          <Eye />
                        </Button>
                      ))}
                    </div>
                  </td>
                )}
                <td className="p-3">{actions(entry, index)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  return (
    <div
      className={
        mode === "cards"
          ? "grid grid-cols-1 gap-4 p-4 sm:grid-cols-2 xl:grid-cols-3"
          : "space-y-6 p-4"
      }
    >
      {entries.map((entry, index) => (
        <article
          key={resourceKey(entry)}
          className={
            mode === "cards"
              ? "space-y-3 rounded-xl border border-border/30 bg-card p-4"
              : "space-y-4 border-b border-border/20 pb-6"
          }
        >
          <div className="flex items-start gap-2">
            {checkbox(entry)}
            <div className="min-w-0 flex-1">
              <h3 className="break-words text-sm font-semibold">
                {resourceTitle(entry)}
              </h3>
              <p className="text-xs text-muted-foreground">
                {canvasTypeNames[entry.header.resourceType]}
              </p>
            </div>
          </div>
          {mode === "read" ? (
            <CanvasResourceRead
              resource={entry}
              library={library}
              onPreview={onPreview}
            />
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2">
                {media(entry)
                  .filter((item) => item.kind === "image")
                  .map((item) => (
                    <button
                      type="button"
                      key={item.title}
                      aria-label={item.title}
                      onClick={() => onPreview(item)}
                      className="rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <SafeImage
                        src={item.source}
                        alt={item.title}
                        className="aspect-video w-full rounded-lg object-contain"
                      />
                    </button>
                  ))}
              </div>
              <p className="line-clamp-4 whitespace-pre-wrap text-sm text-muted-foreground">
                {String(
                  entry.data.content ??
                    entry.data.synopsis ??
                    entry.data.sceneDescription ??
                    entry.data.aiPrompt ??
                    "",
                )}
              </p>
              {media(entry)
                .filter((item) => item.kind === "video")
                .map((item) => (
                  <Button
                    key={item.title}
                    size="sm"
                    variant="outline"
                    onClick={() => onPreview(item)}
                  >
                    <Eye />
                    预览视频
                  </Button>
                ))}
            </>
          )}
          {actions(entry, index)}
        </article>
      ))}
    </div>
  );
}
