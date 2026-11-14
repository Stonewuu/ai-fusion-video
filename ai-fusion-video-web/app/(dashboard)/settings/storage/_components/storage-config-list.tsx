import { Edit2, HardDrive, Star, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  STORAGE_PROVIDER_LABELS,
  STORAGE_TYPE_LABELS,
  type StorageConfig,
} from "@/lib/api/storage";
import { normalizeStorageType } from "./storage-form-options";

interface StorageConfigListProps {
  configurations: StorageConfig[];
  builtIn: StorageConfig | null;
  onEdit: (config: StorageConfig) => void;
  onDelete: (id: number) => void;
  onSetDefault: (id: number) => void;
}

export function StorageConfigList({
  configurations,
  builtIn,
  onEdit,
  onDelete,
  onSetDefault,
}: StorageConfigListProps) {
  if (builtIn) {
    return (
      <section aria-label="内置本地存储" className="flex items-start gap-3 rounded-xl border border-border/30 bg-card/50 p-4 backdrop-blur-sm">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted/60">
          <HardDrive className="size-5 text-muted-foreground" />
        </span>
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="text-sm font-medium">{builtIn.name}</h4>
            <Badge variant="secondary">内置默认</Badge>
          </div>
          <p className="break-all font-mono text-xs text-muted-foreground">{builtIn.basePath}</p>
          <p className="text-xs text-muted-foreground">未配置存储时自动使用。新增存储配置后，将使用你的配置。</p>
        </div>
      </section>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {configurations.map((config) => {
        const isLocal = normalizeStorageType(config) === "local";
        const providerLabel = isLocal
          ? STORAGE_TYPE_LABELS[config.type]
          : STORAGE_PROVIDER_LABELS[config.provider || "generic_s3"];
        return (
          <section key={config.id} className="flex min-w-0 flex-wrap items-center gap-3 rounded-xl border border-border/30 bg-card/50 p-4 backdrop-blur-sm">
            <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted/60">
              <HardDrive className="size-5 text-muted-foreground" />
            </span>
            <div className="flex min-w-0 flex-1 flex-col gap-1">
              <div className="flex flex-wrap items-center gap-2">
                <h4 className="text-sm font-medium">{config.name}</h4>
                <Badge variant="secondary">{providerLabel}</Badge>
                {config.isDefault && <Badge variant="secondary"><Star className="size-3" />默认</Badge>}
                {config.status === 0 && <span className="text-xs text-muted-foreground">已停用</span>}
              </div>
              <p className="break-all font-mono text-xs text-muted-foreground">
                {isLocal ? config.basePath : config.endpoint}
                {!isLocal && config.bucketName && ` / ${config.bucketName}`}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {!config.isDefault && (
                <Button variant="outline" onClick={() => onSetDefault(config.id)}>
                  <Star />设为默认
                </Button>
              )}
              <Button variant="ghost" size="icon" aria-label={`编辑存储 ${config.name}`} onClick={() => onEdit(config)}>
                <Edit2 />
              </Button>
              <Button variant="destructive-ghost" size="icon" aria-label={`删除存储 ${config.name}`} onClick={() => onDelete(config.id)}>
                <Trash2 />
              </Button>
            </div>
          </section>
        );
      })}
    </div>
  );
}
