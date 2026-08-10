import {
  getPipelineStatus,
  type PipelineRunStatusResponse,
} from "@/lib/api/ai-pipeline";

export interface PipelineStatusSyncTarget {
  taskId: string;
  runId: string;
}

interface PipelineStatusSyncOptions {
  intervalMs: number;
  listTargets: () => PipelineStatusSyncTarget[];
  applyStatus: (
    target: PipelineStatusSyncTarget,
    status: PipelineRunStatusResponse,
  ) => void;
  reportError: (target: PipelineStatusSyncTarget, error: unknown) => void;
}

/** Keeps durable run metadata synchronized while no journal hydration owns it. */
export class PipelineStatusSynchronizer {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private polling = false;

  constructor(private readonly options: PipelineStatusSyncOptions) {}

  schedule(delayMs = this.options.intervalMs): void {
    if (this.polling || this.options.listTargets().length === 0) {
      return;
    }
    if (this.timer) {
      if (delayMs > 0) return;
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.poll();
    }, delayMs);
  }

  stop(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private async poll(): Promise<void> {
    if (this.polling) return;
    this.polling = true;
    const targets = this.options.listTargets();
    try {
      await Promise.all(targets.map(async (target) => {
        try {
          const status = await getPipelineStatus({ runId: target.runId });
          if (status.runId !== target.runId) {
            throw new Error("Pipeline status returned a different runId");
          }
          this.options.applyStatus(target, status);
        } catch (error) {
          this.options.reportError(target, error);
        }
      }));
    } finally {
      this.polling = false;
      this.schedule();
    }
  }
}
