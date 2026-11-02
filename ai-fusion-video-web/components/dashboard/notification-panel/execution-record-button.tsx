"use client";

import { Button } from "@/components/ui/button";
import type { ExecutionRecordRef } from "@/lib/api/execution-records";
import { openExecutionRecord } from "@/lib/execution-record-navigation";

export function ExecutionRecordButton({ target, onOpen }: { target: ExecutionRecordRef; onOpen?: () => void }) {
  return <Button variant="outline" size="sm" onClick={() => {
    onOpen?.();
    openExecutionRecord(target);
  }}>在运行与通知中查看</Button>;
}
