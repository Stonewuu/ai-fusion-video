"use client";

import type { ReactNode } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import type { ExecutionRecordRef } from "@/lib/api/execution-records";
import { useAuthStore } from "@/lib/store/auth-store";
import { ExecutionRecordList } from "./execution-records";
import { useExecutionRecords } from "./use-execution-records";

export type NotificationView = "tasks" | "exceptions";

interface Props {
  view: NotificationView;
  onViewChange: (view: NotificationView) => void;
  selectedRecord: ExecutionRecordRef | null;
  onSelectRecord: (record: ExecutionRecordRef) => void;
  compact?: boolean;
  children: ReactNode;
}

export function NotificationViews(props: Props) {
  const userId = useAuthStore(state => state.user?.id);
  return <Views key={userId} {...props} userId={userId} />;
}

function Views({ view, onViewChange, selectedRecord, onSelectRecord, children, compact = false, userId }: Props & { userId: number | undefined }) {
  const records = useExecutionRecords(userId);
  return (
    <Tabs value={view} onValueChange={value => onViewChange(value as NotificationView)} className="min-h-0 flex-1 gap-0">
      <div className="shrink-0 p-3">
        <TabsList aria-label="通知分类" className="w-full">
          <TabsTrigger value="tasks" className="transition-colors motion-reduce:transition-none">任务</TabsTrigger>
          <TabsTrigger value="exceptions" className="transition-colors motion-reduce:transition-none">
            异常记录
            {records.total !== null && records.total > 0 && <span className="text-xs tabular-nums" title={`${records.total} 条异常记录`}>
              {records.total > 99 ? "99+" : records.total}
            </span>}
          </TabsTrigger>
        </TabsList>
      </div>
      <TabsContent value="tasks" keepMounted className="min-h-0">
        <OverlayScrollArea className="h-full" aria-label="任务列表">
          {children}
        </OverlayScrollArea>
      </TabsContent>
      <TabsContent value="exceptions" keepMounted className="min-h-0">
        <ExecutionRecordList records={records} selected={selectedRecord} onSelect={onSelectRecord} showFilters={!compact} />
      </TabsContent>
    </Tabs>
  );
}
