# AgentScope V2 GA 前端兼容、切换与验收 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重写现有通知面板和 Pipeline UI 的前提下，让前端可靠消费带标准 SSE 游标的 Durable Run 事件，并完成 V1 Runtime 清理、回滚文档与全链路验收。

**Architecture:** 前端新增纯函数 SSE parser、事件规范化器和按 `(runId, sequence)` 去重的游标层，现有 Zustand reducer 只消费规范化事件。传输 EOF 不再代表业务完成，页面刷新通过 run status 与 MySQL 回放恢复；最终切换只在后端 durable runtime、模型和工具阶段全部通过后进行，并保留数据库向后兼容的二进制回滚窗口。

**Tech Stack:** Next.js 16、React 19、TypeScript 5、Zustand 5、Vitest、Web Fetch/ReadableStream、Spring WebFlux SSE、Maven、Flyway、MySQL 8、Redis。

## Global Constraints

- 所有 AgentScope 依赖必须为 `2.0.0` GA；不得残留 V1、RC、starter 或 session-mysql。
- 前端包管理器固定为 `pnpm@10.32.1`，所有命令使用 `corepack pnpm`。
- 标准 SSE `id` 必须为 `{runId}:{sequence}`；MySQL event journal 是顺序和回放真相，Redis 只负责 wake-up。
- 旧 `outputType`、通知面板、历史 timeline、缺失 `TOOL_CALL` placeholder 和数据 invalidation 行为必须保持兼容。
- 只有主 Agent 的 `DONE/ERROR/CANCELLED` 能收敛 Pipeline；子 Agent 终态不得关闭主 run。
- HTTP/SSE 断线只卸载观察者，不取消业务 run；EOF 不能推断 `done`。
- 未知 schema 或 raw event 必须显式报告，不能伪装成 `CONTENT` 或静默丢弃。
- 任一 protocol error 都把当前 parser/stream 置为 terminal；同一 byte chunk 中位于失败帧之后的帧不得进入 `onEvent`、store 或 cursor。
- `rawEventType` 与 `controlType` 是两个命名空间：确认请求使用 raw `PLATFORM_USER_CONFIRM_REQUIRED` 与 `controlType=USER_CONFIRM_REQUIRED`，不得互换。
- `RunExecutionSupervisor.start/resume` 只返回 server-owned execution 的 admission；HTTP/SSE observer 取消不得传播为 execution 取消。
- 所有执行命令都显式 `Set-Location` 到仓库根、后端模块或前端模块，并在多命令步骤中检查每一次 `$LASTEXITCODE`。
- 不修改已执行 Flyway；保留现有资产、子资产、分镜、任务、conversation/message ID。
- AgentScope/Reactor 链路禁止 `.block()`、`.toIterable()`、`Thread.sleep()`、ThreadLocal 和未隔离的阻塞 I/O。
- 最终必须真实执行后端全量、集成 profile、Provider/Ark smoke，以及前端 test/lint/build；凭据缺失必须报告“未验证”，不能报告通过。

---

## File responsibility map

- `ai-fusion-video-web/lib/api/sse-frame-parser.ts`：与业务无关的增量 SSE wire parser。
- `ai-fusion-video-web/lib/api/pipeline-event-normalizer.ts`：验证 schema、SSE id 与 JSON identity，生成强类型规范事件。
- `ai-fusion-video-web/lib/store/pipeline-event-cursor.ts`：按 run 维护最后已接收 sequence，拒绝重复/倒序。
- `ai-fusion-video-web/lib/api/ai-pipeline.ts`：启动、重连、状态和取消 API；不承担 reducer 逻辑。
- `ai-fusion-video-web/lib/api/ai-assistant.ts`：扩展兼容事件类型，不改变现有对话 REST API。
- `ai-fusion-video-web/lib/store/pipeline-store.ts`：业务 reducer、重连编排、terminal/invalidation 回调恰好一次。
- `ai-fusion-video-web/components/dashboard/agent-pipeline/use-agent-pipeline.ts`：组件级观察者生命周期；abort 不等于业务 cancel。
- `ai-fusion-video-web/components/dashboard/agent-pipeline/agent-confirmation-card.tsx`：渲染服务端已持久化的待确认工具集合，提交批准/拒绝并显示过期或已处理反馈。
- `docs/operations/agentscope-v2-cutover.md`：停写、切换、验证、回滚和重新启用 V2 的操作手册。

### Task 1: 建立前端事件协议测试基线

**Files:**
- Modify: `ai-fusion-video-web/package.json`
- Modify: `ai-fusion-video-web/pnpm-lock.yaml`
- Create: `ai-fusion-video-web/vitest.config.ts`
- Create: `ai-fusion-video-web/tests/unit/test-globals.d.ts`
- Create: `ai-fusion-video-web/tests/unit/test-setup.ts`

**Interfaces:**
- Consumes: 仓库固定的 `pnpm@10.32.1`。
- Produces: `corepack pnpm test` 与 `corepack pnpm test -- <file>`；`@/*` alias 与 Next 项目一致。

- [ ] **Step 1: 在 package manifest 中声明测试命令和固定测试依赖**

```json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.3.0",
    "@testing-library/user-event": "^14.6.1",
    "@vitest/coverage-v8": "^3.2.4",
    "jsdom": "^26.1.0",
    "vitest": "^3.2.4"
  }
}
```

- [ ] **Step 2: 生成并冻结 lockfile**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm install; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: `pnpm-lock.yaml` 由 pnpm 10.32.1 更新，命令退出码为 0，未切换 package manager。

- [ ] **Step 3: 写入 Vitest 配置与类型契约**

```ts
// vitest.config.ts
import path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: { alias: { "@": path.resolve(__dirname, ".") } },
  test: {
    environment: "node",
    include: ["tests/unit/**/*.test.{ts,tsx}"],
    setupFiles: ["tests/unit/test-setup.ts"],
    restoreMocks: true,
    clearMocks: true,
  },
});
```

```ts
// tests/unit/test-globals.d.ts
/// <reference types="vitest/globals" />
```

```ts
// tests/unit/test-setup.ts
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 4: 验证空测试基线可启动**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- --passWithNoTests; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS；Vitest 正常启动，未出现 alias、ESM 或 TypeScript 配置错误。

- [ ] **Step 5: 提交测试基础设施**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video-web/package.json ai-fusion-video-web/pnpm-lock.yaml ai-fusion-video-web/vitest.config.ts ai-fusion-video-web/tests/unit/test-globals.d.ts ai-fusion-video-web/tests/unit/test-setup.ts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "test(web): add pipeline event test harness"
```

### Task 2: 实现增量 SSE wire parser

**Files:**
- Create: `ai-fusion-video-web/lib/api/sse-frame-parser.ts`
- Create: `ai-fusion-video-web/tests/unit/sse-frame-parser.test.ts`

**Interfaces:**
- Consumes: 任意分块的 UTF-8 解码字符串。
- Produces: `SseFrame`、`createSseFrameParser(onFrame)`；支持 CRLF、跨 chunk、multi-line data、comment、`event:` 与 `id:`。

- [ ] **Step 1: 写入失败测试，固定 wire 行为**

```ts
import { describe, expect, it, vi } from "vitest";
import { createSseFrameParser } from "@/lib/api/sse-frame-parser";

describe("createSseFrameParser", () => {
  it("parses id event and multiline data across CRLF chunks", () => {
    const frames: unknown[] = [];
    const parser = createSseFrameParser((frame) => frames.push(frame));
    parser.push(":keepalive\r\nid: run-1:7\r\nevent: pipeline-event\r\ndata: {\"a\":\r\n");
    parser.push("data: 1}\r\n\r\n");
    parser.end();
    expect(frames).toEqual([{ id: "run-1:7", event: "pipeline-event", data: "{\"a\":\n1}" }]);
  });

  it("keeps one CRLF delimiter when CR and LF arrive in separate chunks", () => {
    const frames: unknown[] = [];
    const parser = createSseFrameParser((frame) => frames.push(frame));
    parser.push("id: run-2:8\r");
    parser.push("\nevent: pipeline-event\r");
    parser.push("\ndata: ok\r");
    parser.push("\n\r");
    parser.push("\n");
    parser.end();
    expect(frames).toEqual([{ id: "run-2:8", event: "pipeline-event", data: "ok" }]);
  });

  it("flushes a final frame without a blank line", () => {
    const onFrame = vi.fn();
    const parser = createSseFrameParser(onFrame);
    parser.push("data: {\"ok\":true}");
    parser.end();
    expect(onFrame).toHaveBeenCalledOnce();
  });

  it("becomes terminal when a frame callback rejects the protocol", () => {
    const onFrame = vi.fn(() => { throw new Error("SSE_OUTPUT_TYPE_UNSUPPORTED"); });
    const parser = createSseFrameParser(onFrame);
    expect(() => parser.push("data: first\n\ndata: must-not-run\n\n"))
      .toThrow("SSE_OUTPUT_TYPE_UNSUPPORTED");
    expect(onFrame).toHaveBeenCalledOnce();
    parser.push("data: still-terminal\n\n");
    expect(onFrame).toHaveBeenCalledOnce();
  });
});
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/sse-frame-parser.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: FAIL，错误包含 `Cannot find module '@/lib/api/sse-frame-parser'`。

- [ ] **Step 3: 实现无业务语义的 parser**

```ts
export interface SseFrame {
  event?: string;
  id?: string;
  data: string;
}

export interface SseFrameParser {
  push(chunk: string): void;
  end(): void;
}

export function createSseFrameParser(onFrame: (frame: SseFrame) => void): SseFrameParser {
  let buffer = "";
  let pendingCr = false;
  let terminal = false;

  const appendNormalized = (chunk: string, final: boolean) => {
    for (const char of chunk) {
      if (pendingCr) {
        buffer += "\n";
        pendingCr = false;
        if (char === "\n") continue;
      }
      if (char === "\r") pendingCr = true;
      else buffer += char;
    }
    if (final && pendingCr) {
      buffer += "\n";
      pendingCr = false;
    }
  };

  const consume = (final: boolean) => {
    const blocks = buffer.split("\n\n");
    buffer = final ? "" : (blocks.pop() ?? "");
    for (const block of blocks) {
      if (!block) continue;
      let id: string | undefined;
      let event: string | undefined;
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (!line || line.startsWith(":")) continue;
        const colon = line.indexOf(":");
        const field = colon < 0 ? line : line.slice(0, colon);
        const raw = colon < 0 ? "" : line.slice(colon + 1);
        const value = raw.startsWith(" ") ? raw.slice(1) : raw;
        if (field === "id") id = value;
        if (field === "event") event = value;
        if (field === "data") data.push(value);
      }
      if (data.length > 0) {
        try {
          onFrame({ id, event, data: data.join("\n") });
        } catch (error) {
          terminal = true;
          buffer = "";
          throw error;
        }
      }
    }
  };

  return {
    push(chunk) {
      if (terminal) return;
      appendNormalized(chunk, false);
      consume(false);
    },
    end() {
      if (terminal) return;
      appendNormalized("", true);
      consume(true);
    },
  };
}
```

- [ ] **Step 4: 运行 parser 测试**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/sse-frame-parser.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS，4 tests passed；CR/LF 跨 chunk 时不会误切帧，callback 拒绝后 parser 保持 terminal。

- [ ] **Step 5: 提交 parser**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video-web/lib/api/sse-frame-parser.ts ai-fusion-video-web/tests/unit/sse-frame-parser.test.ts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "feat(web): parse cursor-aware SSE frames"
```

### Task 3: 规范化事件身份并维护 per-run cursor

**Files:**
- Modify: `ai-fusion-video-web/lib/api/ai-assistant.ts`
- Create: `ai-fusion-video-web/lib/api/pipeline-event-normalizer.ts`
- Create: `ai-fusion-video-web/lib/store/pipeline-event-cursor.ts`
- Create: `ai-fusion-video-web/tests/unit/pipeline-event-normalizer.test.ts`
- Create: `ai-fusion-video-web/tests/unit/pipeline-event-cursor.test.ts`

**Interfaces:**
- Consumes: `SseFrame`、兼容 `AiChatStreamEvent`。
- Produces: `NormalizedPipelineEvent`、`normalizePipelineEvent(frame, expectedRunId?)`、`acceptPipelineEvent(cursor,event)`。

- [ ] **Step 1: 写入 identity 冲突和去重失败测试**

```ts
import { expect, it } from "vitest";
import { normalizePipelineEvent } from "@/lib/api/pipeline-event-normalizer";

it("rejects identity conflicts and unknown protocol discriminators before cursor advance", () => {
  expect(() => normalizePipelineEvent({
    id: "run-a:9",
    data: JSON.stringify({ schemaVersion: 1, runId: "run-b", sequence: 9, outputType: "CONTENT" }),
  })).toThrow("SSE_EVENT_IDENTITY_MISMATCH");

  expect(() => normalizePipelineEvent({
    id: "run-a:10",
    data: JSON.stringify({ schemaVersion: 1, runId: "run-a", sequence: 10, outputType: "FUTURE_OUTPUT" }),
  })).toThrow("SSE_OUTPUT_TYPE_UNSUPPORTED");

  expect(() => normalizePipelineEvent({
    id: "run-a:11",
    data: JSON.stringify({
      schemaVersion: 1, runId: "run-a", sequence: 11,
      outputType: "CONTENT", rawEventType: "FUTURE_AGENT_EVENT",
    }),
  })).toThrow("SSE_RAW_EVENT_UNSUPPORTED");

  expect(() => normalizePipelineEvent({
    id: "run-a:12",
    data: JSON.stringify({
      schemaVersion: 1, runId: "run-a", sequence: 12,
      outputType: "CONTENT", controlType: "FUTURE_CONTROL",
    }),
  })).toThrow("SSE_CONTROL_TYPE_UNSUPPORTED");

  expect(() => normalizePipelineEvent({
    id: "run-a:13",
    data: JSON.stringify({
      schemaVersion: 1, runId: "run-a", sequence: 13, outputType: "TOOL_CALL",
      rawEventType: "PLATFORM_USER_CONFIRM_REQUIRED",
      controlType: "USER_CONFIRM_REQUIRED", replyId: "reply-1",
      expiresAt: "not-a-date", pendingToolCalls: [],
    }),
  })).toThrow("SSE_CONTROL_PAYLOAD_INVALID");

  for (const id of ["run-a:0", "run-a:9007199254740993"]) {
    expect(() => normalizePipelineEvent({
      id,
      data: JSON.stringify({ schemaVersion: 1, runId: "run-a", sequence: 0, outputType: "CONTENT" }),
    })).toThrow("SSE_CURSOR_INVALID");
  }
});

it("accepts the exact persisted backend confirmation projection", () => {
  const event = normalizePipelineEvent({
    id: "run-a:14",
    event: "pipeline-event",
    data: JSON.stringify({
      schemaVersion: 1,
      runId: "run-a",
      sequence: 14,
      outputType: "TOOL_CALL",
      rawEventType: "PLATFORM_USER_CONFIRM_REQUIRED",
      controlType: "USER_CONFIRM_REQUIRED",
      replyId: "reply-1",
      expiresAt: "2026-07-21T12:00:00Z",
      pendingToolCalls: [
        { toolCallId: "tool-1", toolName: "generate_image", argumentsPreview: "{\"prompt\":\"sunrise\"}" },
      ],
    }),
  }, "run-a");
  expect(event).toMatchObject({
    rawEventType: "PLATFORM_USER_CONFIRM_REQUIRED",
    controlType: "USER_CONFIRM_REQUIRED",
    replyId: "reply-1",
  });
});

it("accepts every frozen platform synthetic raw type", () => {
  for (const [sequence, rawEventType] of [
    [15, "PLATFORM_USER_CONFIRM_REQUIRED"],
    [16, "PLATFORM_REQUIRE_EXTERNAL_EXECUTION"],
  ] as const) {
    expect(() => normalizePipelineEvent({
      id: `run-a:${sequence}`,
      data: JSON.stringify({ schemaVersion: 1, runId: "run-a", sequence, outputType: "CONTENT", rawEventType }),
    }, "run-a")).not.toThrow();
  }
});
```

```ts
import { expect, it } from "vitest";
import { acceptPipelineEvent } from "@/lib/store/pipeline-event-cursor";

it("accepts sequence gaps but rejects duplicates and older events", () => {
  const first = acceptPipelineEvent(new Map(), { runId: "r", sequence: 3 });
  expect(first.accepted).toBe(true);
  expect(acceptPipelineEvent(first.cursor, { runId: "r", sequence: 3 }).accepted).toBe(false);
  expect(acceptPipelineEvent(first.cursor, { runId: "r", sequence: 2 }).accepted).toBe(false);
  expect(acceptPipelineEvent(first.cursor, { runId: "r", sequence: 7 }).accepted).toBe(true);
});
```

- [ ] **Step 2: 运行两个测试并确认模块缺失**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/pipeline-event-normalizer.test.ts tests/unit/pipeline-event-cursor.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: FAIL，两个目标模块均无法解析。

- [ ] **Step 3: 扩展兼容事件并实现严格规范化**

```ts
export interface NormalizedPipelineEvent extends AiChatStreamEvent {
  schemaVersion: number;
  runId: string;
  sequence: number;
}

const KNOWN_OUTPUT_TYPES = new Set([
  "REASONING", "CONTENT", "TOOL_CALL", "TOOL_FINISHED",
  "SUB_AGENT_FINISHED", "DONE", "ERROR", "CANCELLED",
]);

export const AGENT_EVENT_TYPES = [
  "AGENT_START", "AGENT_END", "AGENT_RESULT", "MODEL_CALL_START", "MODEL_CALL_END",
  "TEXT_BLOCK_START", "TEXT_BLOCK_DELTA", "TEXT_BLOCK_END",
  "THINKING_BLOCK_START", "THINKING_BLOCK_DELTA", "THINKING_BLOCK_END",
  "DATA_BLOCK_START", "DATA_BLOCK_DELTA", "DATA_BLOCK_END",
  "TOOL_CALL_START", "TOOL_CALL_DELTA", "TOOL_CALL_END",
  "TOOL_RESULT_START", "TOOL_RESULT_TEXT_DELTA", "TOOL_RESULT_DATA_DELTA", "TOOL_RESULT_END",
  "EXCEED_MAX_ITERS", "REQUIRE_USER_CONFIRM", "REQUIRE_EXTERNAL_EXECUTION",
  "USER_CONFIRM_RESULT", "EXTERNAL_EXECUTION_RESULT", "REQUEST_STOP",
  "SUBAGENT_EXPOSED", "HINT_BLOCK", "ALL_TOOLS_DENIED", "CUSTOM",
] as const;
export type AgentEventType = (typeof AGENT_EVENT_TYPES)[number];
export const PLATFORM_RAW_EVENT_TYPES = [
  "PLATFORM_USER_CONFIRM_REQUIRED",
  "PLATFORM_REQUIRE_EXTERNAL_EXECUTION",
] as const;
const KNOWN_RAW_EVENT_TYPES = new Set<string>([
  ...AGENT_EVENT_TYPES,
  ...PLATFORM_RAW_EVENT_TYPES,
]);

const KNOWN_CONTROL_TYPES = new Set(["USER_CONFIRM_REQUIRED"]);

function isValidConfirmationPayload(payload: Record<string, unknown>): boolean {
  if (typeof payload.replyId !== "string" || payload.replyId.length === 0) return false;
  if (typeof payload.expiresAt !== "string" || !Number.isFinite(Date.parse(payload.expiresAt))) return false;
  if (!Array.isArray(payload.pendingToolCalls) || payload.pendingToolCalls.length === 0) return false;
  const ids = new Set<string>();
  return payload.pendingToolCalls.every((candidate) => {
    if (typeof candidate !== "object" || candidate === null) return false;
    const tool = candidate as Record<string, unknown>;
    if (typeof tool.toolCallId !== "string" || tool.toolCallId.length === 0 || ids.has(tool.toolCallId)) return false;
    if (typeof tool.toolName !== "string" || typeof tool.argumentsPreview !== "string") return false;
    ids.add(tool.toolCallId);
    return true;
  });
}

export function normalizePipelineEvent(frame: SseFrame, expectedRunId?: string): NormalizedPipelineEvent {
  const payload = JSON.parse(frame.data) as Record<string, unknown>;
  const match = /^(.+):(\d+)$/.exec(frame.id ?? "");
  if (!match) throw new Error("SSE_CURSOR_INVALID");
  const wireRunId = match[1];
  const wireSequence = Number(match[2]);
  if (!Number.isSafeInteger(wireSequence) || wireSequence < 1) {
    throw new Error("SSE_CURSOR_INVALID");
  }
  if (expectedRunId && wireRunId !== expectedRunId) throw new Error("SSE_RUN_ID_MISMATCH");
  if (payload.runId !== wireRunId || payload.sequence !== wireSequence) {
    throw new Error("SSE_EVENT_IDENTITY_MISMATCH");
  }
  if (payload.schemaVersion !== 1) throw new Error("SSE_SCHEMA_UNSUPPORTED");
  if (!KNOWN_OUTPUT_TYPES.has(String(payload.outputType))) {
    throw new Error("SSE_OUTPUT_TYPE_UNSUPPORTED");
  }
  if (payload.rawEventType !== undefined && !KNOWN_RAW_EVENT_TYPES.has(String(payload.rawEventType))) {
    throw new Error("SSE_RAW_EVENT_UNSUPPORTED");
  }
  if (payload.controlType !== undefined && !KNOWN_CONTROL_TYPES.has(String(payload.controlType))) {
    throw new Error("SSE_CONTROL_TYPE_UNSUPPORTED");
  }
  if (payload.controlType === "USER_CONFIRM_REQUIRED" && !isValidConfirmationPayload(payload)) {
    throw new Error("SSE_CONTROL_PAYLOAD_INVALID");
  }
  return payload as unknown as NormalizedPipelineEvent;
}
```

```ts
export function acceptPipelineEvent<T extends { runId: string; sequence: number }>(
  current: ReadonlyMap<string, number>, event: T
): { accepted: boolean; cursor: Map<string, number> } {
  const previous = current.get(event.runId) ?? 0;
  if (event.sequence <= previous) return { accepted: false, cursor: new Map(current) };
  const cursor = new Map(current);
  cursor.set(event.runId, event.sequence);
  return { accepted: true, cursor };
}
```

- [ ] **Step 4: 运行规范化和 cursor 测试**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/pipeline-event-normalizer.test.ts tests/unit/pipeline-event-cursor.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS；identity、未知 output/raw event 在进入 cursor/store 前被拒绝，合法 gap 被接受，重复/倒序被忽略。

- [ ] **Step 5: 提交协议规范化层**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video-web/lib/api/ai-assistant.ts ai-fusion-video-web/lib/api/pipeline-event-normalizer.ts ai-fusion-video-web/lib/store/pipeline-event-cursor.ts ai-fusion-video-web/tests/unit/pipeline-event-normalizer.test.ts ai-fusion-video-web/tests/unit/pipeline-event-cursor.test.ts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "feat(web): normalize and deduplicate pipeline events"
```

### Task 4: 把启动和重连 API 接到标准 SSE 游标

**Files:**
- Modify: `ai-fusion-video-web/lib/api/ai-pipeline.ts`
- Create: `ai-fusion-video-web/tests/unit/ai-pipeline-stream.test.ts`

**Interfaces:**
- Consumes: 后端 `POST /api/ai/pipeline/run` 与 `GET /api/ai/pipeline/reconnect?runId=...&afterSequence=...`。
- Produces: `pipelineStream(request,callbacks)`、`reconnectPipelineStream(options,callbacks)`、`PipelineStreamHandle`、`PipelineStreamCallbacks`；初始 POST 从首个合法事件 latch runId，重连使用预知 runId 与显式 `Last-Event-ID`。

- [ ] **Step 1: 写入请求 header、跨 chunk 和 EOF 语义测试**

```ts
it("sends the composite Last-Event-ID when reconnecting", async () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(streamResponse([]));
  reconnectPipelineStream({ runId: "run-7", afterSequence: 12 }, callbacks);
  await vi.waitFor(() => expect(fetchSpy).toHaveBeenCalled());
  const init = fetchSpy.mock.calls[0][1] as RequestInit;
  expect(new Headers(init.headers).get("Last-Event-ID")).toBe("run-7:12");
});

it("reports transport close without synthesizing DONE", async () => {
  const onTransportClosed = vi.fn();
  pipelineStream(request, { onEvent: vi.fn(), onTransportClosed });
  await vi.waitFor(() => expect(onTransportClosed).toHaveBeenCalledOnce());
});

it("makes protocol failure terminal before a later frame in the same chunk", async () => {
  const onEvent = vi.fn();
  const onProtocolError = vi.fn();
  vi.spyOn(globalThis, "fetch").mockResolvedValue(streamResponseInOneChunk([
    frame({ id: "run-7:13", runId: "run-7", sequence: 13, outputType: "FUTURE_OUTPUT" }),
    frame({ id: "run-7:14", runId: "run-7", sequence: 14, outputType: "CONTENT", content: "must-not-run" }),
  ]));
  pipelineStream(request, { onEvent, onProtocolError });
  await vi.waitFor(() => expect(onProtocolError).toHaveBeenCalledOnce());
  expect(onEvent).not.toHaveBeenCalled();
});

it("latches the first run id on POST and rejects a later run switch", async () => {
  const onEvent = vi.fn();
  const onProtocolError = vi.fn();
  vi.spyOn(globalThis, "fetch").mockResolvedValue(streamResponseInOneChunk([
    frame({ id: "run-7:1", runId: "run-7", sequence: 1, outputType: "CONTENT" }),
    frame({ id: "run-8:2", runId: "run-8", sequence: 2, outputType: "CONTENT" }),
  ]));
  pipelineStream(request, { onEvent, onProtocolError });
  await vi.waitFor(() => expect(onProtocolError).toHaveBeenCalledOnce());
  expect(onEvent).toHaveBeenCalledTimes(1);
  expect(onEvent.mock.calls[0][0].runId).toBe("run-7");
});
```

- [ ] **Step 2: 运行 API 测试并确认旧签名失败**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/ai-pipeline-stream.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: FAIL；旧 `reconnectPipelineStream(conversationId, callbacks)` 不接受 run cursor，且 EOF 只触发 `onComplete`。

- [ ] **Step 3: 实现 run-aware 传输签名**

```ts
export interface PipelineReconnectOptions {
  runId: string;
  afterSequence: number;
}

export interface RunningPipelineRun {
  runId: string;
  conversationId: string;
  projectId: number;
  title: string;
  category?: string;
  status: "RUNNING" | "WAITING_CONFIRMATION" | "WAITING_EXTERNAL" | "CANCEL_REQUESTED";
  startedAt: string;
}

export type PipelineRunStatusCode =
  | "RUNNING" | "WAITING_CONFIRMATION" | "WAITING_EXTERNAL" | "CANCEL_REQUESTED"
  | "COMPLETED" | "FAILED" | "CANCELLED";

export interface PipelineRunStatus {
  runId: string;
  status: PipelineRunStatusCode;
  lastSequence: number;
  waitingReplyId?: string;
  terminalEvent?: NormalizedPipelineEvent;
}

export interface PipelineStreamCallbacks {
  onEvent: (event: NormalizedPipelineEvent) => void;
  onProtocolError?: (error: Error) => void;
  onTransportError?: (error: Error) => void;
  onTransportClosed?: () => void;
}

export interface PipelineStreamHandle {
  abortObserver(): void;
}

function openPipelineEventStream(
  url: string,
  init: RequestInit,
  initialExpectedRunId: string | undefined,
  callbacks: PipelineStreamCallbacks,
): PipelineStreamHandle {
  const observerController = new AbortController();
  let expectedRunId = initialExpectedRunId;
  let protocolFailed = false;
  const parser = createSseFrameParser((frame) => {
    try {
      // normalize 必须先于 onEvent；cursor 只在 store 的 onEvent 内前移。
      const event = normalizePipelineEvent(frame, expectedRunId);
      expectedRunId ??= event.runId;
      callbacks.onEvent(event);
    } catch (error) {
      protocolFailed = true;
      observerController.abort();
      callbacks.onProtocolError?.(error as Error);
      throw error; // 让 parser 同步停止当前 chunk 中的剩余 frame。
    }
  });

  void (async () => {
    const response = await authenticatedFetch(url, { ...init, signal: observerController.signal });
    if (!response.ok || !response.body) throw new Error(`SSE_HTTP_${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.end();
    if (!protocolFailed && !observerController.signal.aborted) callbacks.onTransportClosed?.();
  })().catch((error) => {
    if (!protocolFailed && !observerController.signal.aborted) callbacks.onTransportError?.(error as Error);
  });

  return { abortObserver: () => observerController.abort() };
}

export function pipelineStream(
  request: AiChatReq,
  callbacks: PipelineStreamCallbacks,
): PipelineStreamHandle {
  return openPipelineEventStream(
    `${API_BASE_URL}/api/ai/pipeline/run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    },
    undefined,
    callbacks,
  );
}

export function reconnectPipelineStream(
  options: PipelineReconnectOptions,
  callbacks: PipelineStreamCallbacks
): PipelineStreamHandle {
  return openPipelineEventStream(
    `${API_BASE_URL}/api/ai/pipeline/reconnect?runId=${encodeURIComponent(options.runId)}&afterSequence=${options.afterSequence}`,
    { headers: { "Last-Event-ID": `${options.runId}:${options.afterSequence}` } },
    options.runId,
    callbacks
  );
}

export function getPipelineStatus(runId: string): Promise<PipelineRunStatus> {
  return http.get("/api/ai/pipeline/status", { params: { runId } });
}

export function listRunningPipelines(): Promise<RunningPipelineRun[]> {
  return http.get("/api/ai/pipeline/running");
}

export function cancelPipeline(runId: string): Promise<void> {
  return http.post("/api/ai/pipeline/cancel", undefined, { params: { runId } });
}
```

- [ ] **Step 4: 运行 API 与 parser/normalizer 回归**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/ai-pipeline-stream.test.ts tests/unit/sse-frame-parser.test.ts tests/unit/pipeline-event-normalizer.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS；读取结束只触发 `onTransportClosed`；初始 POST latch 首个 runId；run switch 和未知协议只触发一次 `onProtocolError`，失败帧及其后同 chunk 帧不进入 store/cursor；UTF-8 decoder 保留跨 byte chunk 字符。

- [ ] **Step 5: 提交标准 SSE 客户端**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video-web/lib/api/ai-pipeline.ts ai-fusion-video-web/tests/unit/ai-pipeline-stream.test.ts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "feat(web): reconnect pipeline streams by run cursor"
```

### Task 5: 让 Zustand reducer 只由主终态收敛

**Files:**
- Modify: `ai-fusion-video-web/lib/store/pipeline-store.ts`
- Create: `ai-fusion-video-web/tests/unit/pipeline-store.test.ts`

**Interfaces:**
- Consumes: `NormalizedPipelineEvent`、`getPipelineStatus(runId)`、`reconnectPipelineStream`。
- Produces: `PipelineState.runId/lastSequence`；主终态 callback 和 invalidation 恰好一次；transport close 触发 status/reconnect。

- [ ] **Step 1: 写入重复、子终态、EOF 与 placeholder 失败测试**

```ts
it("reduces one event once and settles only on the main terminal", () => {
  store.getState().acceptEvent(taskId, event({ runId: "r", sequence: 2, outputType: "CONTENT", content: "x" }));
  store.getState().acceptEvent(taskId, event({ runId: "r", sequence: 2, outputType: "CONTENT", content: "x" }));
  store.getState().acceptEvent(taskId, event({ runId: "r", sequence: 3, outputType: "ERROR", agentName: "child" }));
  expect(store.getState().tasks[0].state.timeline).toHaveLength(1);
  expect(store.getState().tasks[0].status).toBe("running");
});

it("keeps a task running when transport closes before terminal", async () => {
  await store.getState().handleTransportClosed(taskId);
  expect(store.getState().tasks[0].status).toBe("running");
});

it("creates a placeholder when TOOL_FINISHED arrives without TOOL_CALL", () => {
  store.getState().acceptEvent(taskId, event({ runId: "r", sequence: 5, outputType: "TOOL_FINISHED", toolCallId: "tc" }));
  expect(store.getState().tasks[0].state.timeline[0]).toMatchObject({ type: "tool", id: "tc" });
});

it("rebuilds a running task after refresh and replays that run from sequence zero", async () => {
  listRunningPipelinesMock.mockResolvedValue([{ runId: "run-r", conversationId: "conv-r", projectId: 7, title: "恢复", status: "RUNNING" }]);
  await store.getState().tryReconnect();
  expect(store.getState().tasks.find((task) => task.state.runId === "run-r")?.state.authoritativeRunStatus)
    .toBe("RUNNING");
  expect(reconnectPipelineStreamMock).toHaveBeenCalledWith(
    { runId: "run-r", afterSequence: 0 }, expect.any(Object));
});

it("keeps the observer and running state after a cancel request is accepted", async () => {
  await store.getState().cancelPipeline(taskId);
  expect(abortObserverMock).not.toHaveBeenCalled();
  expect(store.getState().tasks[0]).toMatchObject({ status: "running", state: { cancelRequested: true } });
});

it("backs off repeated transport closes and caps reconnect delay at 30 seconds", () => {
  expect(reconnectDelayMs(0, () => 0)).toBe(250);
  expect(reconnectDelayMs(20, () => 0)).toBe(30_000);
});
```

- [ ] **Step 2: 运行 store 测试并确认旧 EOF 行为失败**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/pipeline-store.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: FAIL；旧 `onComplete` 把 running 直接标为 done，状态中也没有 `runId/lastSequence`。

- [ ] **Step 3: 加入游标、主终态和一次性副作用状态**

```ts
export interface PipelineState {
  status: "running" | "done" | "error" | "cancelled";
  runId?: string;
  lastSequence: number;
  terminalSequence?: number;
  terminalEffectsApplied: boolean;
  cancelRequested: boolean;
  reconnectAttempts: number;
  authoritativeRunStatus?: PipelineRunStatusCode;
  reasoningText: string;
  timeline: TimelineItem[];
  conversationId?: string;
  error?: string;
}

interface PipelineStoreState {
  tryReconnect: () => Promise<void>;
}

function isMainTerminal(event: NormalizedPipelineEvent): boolean {
  return (event.outputType === "DONE" || event.outputType === "ERROR" || event.outputType === "CANCELLED")
    && event.parentToolCallId == null
    && event.agentName == null;
}

async function reconnectRunningRuns(): Promise<void> {
  for (const run of await listRunningPipelines()) {
    addRecoveredTask(run, { authoritativeRunStatus: run.status });
    attachPipelineObserver(run.runId, 0);
  }
}

async function requestPersistentCancel(task: PipelineTask): Promise<void> {
  if (!task.state.runId || task.status !== "running") return;
  markCancelRequested(task.id, true);
  try {
    await cancelPipeline(task.state.runId);
  } catch (error) {
    markCancelRequested(task.id, false);
    throw error;
  }
}
```

- [ ] **Step 4: 把 transport close 改为查询状态并从 lastSequence 重连**

```ts
async function recoverAfterTransportClose(task: PipelineTask): Promise<void> {
  if (!task.state.runId || task.status !== "running") return;
  const status = await getPipelineStatus(task.state.runId);
  if (status.terminalEvent) {
    acceptEvent(task.id, status.terminalEvent);
    return;
  }
  const delay = reconnectDelayMs(task.state.reconnectAttempts, Math.random);
  scheduleReconnect(task.id, delay, () => reconnectPipelineStream(
      { runId: task.state.runId!, afterSequence: task.state.lastSequence },
      callbacksFor(task.id)));
}

function reconnectDelayMs(attempt: number, random: () => number): number {
  const base = Math.min(30_000, 250 * 2 ** Math.min(attempt, 7));
  return Math.min(30_000, Math.round(base * (1 + random() * 0.2)));
}
```

- [ ] **Step 5: 运行 store 全部测试**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/pipeline-store.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS；重复和倒序不重复 reduce，sequence gap 合法，子终态不 settle，EOF 后保持 running，主终态副作用一次。

- [ ] **Step 6: 提交 durable store**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video-web/lib/store/pipeline-store.ts ai-fusion-video-web/tests/unit/pipeline-store.test.ts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "feat(web): recover durable pipeline runs"
```

### Task 6: 修正组件观察者取消与业务取消边界

**Files:**
- Modify: `ai-fusion-video-web/components/dashboard/agent-pipeline/use-agent-pipeline.ts`
- Create: `ai-fusion-video-web/tests/unit/use-agent-pipeline-contract.test.ts`

**Interfaces:**
- Consumes: `PipelineStreamHandle.abortObserver()`、`cancelPipeline(runId)`。
- Produces: effect cleanup 只断开观察；用户点击取消才调用持久化 cancel API。

- [ ] **Step 1: 写入卸载不取消业务 run 的静态契约测试**

```ts
it("separates observer cleanup from explicit business cancellation", async () => {
  const source = await readFile("components/dashboard/agent-pipeline/use-agent-pipeline.ts", "utf8");
  expect(source).toContain("abortObserver");
  expect(source).toContain("await cancelPipeline(state.runId)");
  expect(source).not.toMatch(/return\s*\(\)\s*=>\s*cancelPipeline/);
});

it("does not synthesize CANCELLED when the cancel request is only acknowledged", async () => {
  await result.current.cancelRun();
  expect(result.current.state.cancelRequested).toBe(true);
  expect(result.current.state.status).toBe("running");
});
```

- [ ] **Step 2: 运行契约测试并确认旧 AbortController 语义失败**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/use-agent-pipeline-contract.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: FAIL；旧 hook 只持有裸 `AbortController`，取消 API 使用 conversationId。

- [ ] **Step 3: 实现观察者与业务取消分离**

```ts
const streamRef = useRef<PipelineStreamHandle | null>(null);

useEffect(() => () => {
  streamRef.current?.abortObserver();
}, []);

const cancelRun = useCallback(async () => {
  if (!state.runId) return;
  setState((previous) => ({ ...previous, cancelRequested: true }));
  try {
    await cancelPipeline(state.runId);
  } catch (error) {
    setState((previous) => ({ ...previous, cancelRequested: false }));
    throw error;
  }
}, [state.runId]);
```

- [ ] **Step 4: 运行 hook 契约和 store 回归**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/use-agent-pipeline-contract.test.ts tests/unit/pipeline-store.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS；组件卸载不调用 cancel endpoint，显式取消按 runId 调用，请求确认后仍等待持久化 `CANCELLED` 终态。

- [ ] **Step 5: 提交观察者生命周期修正**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video-web/components/dashboard/agent-pipeline/use-agent-pipeline.ts ai-fusion-video-web/tests/unit/use-agent-pipeline-contract.test.ts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "fix(web): separate pipeline observation from cancellation"
```

### Task 7: 锁定通知面板和历史渲染兼容性

**Files:**
- Modify: `ai-fusion-video-web/lib/api/pipeline-event-normalizer.ts`
- Test: `ai-fusion-video-web/components/dashboard/notification-panel/history.ts`
- Test: `ai-fusion-video-web/components/dashboard/notification-panel/hooks.tsx`
- Test: `ai-fusion-video-web/components/dashboard/notification-panel/timeline.tsx`
- Test: `ai-fusion-video-web/components/dashboard/notification-panel/results/index.tsx`
- Create: `ai-fusion-video-web/tests/unit/notification-pipeline-compat.test.ts`

**Interfaces:**
- Consumes: 现有 `PipelineTask` 和消息历史 DTO。
- Produces: `messagesToTimeline`、结果 renderer 和 invalidation 映射无结构性回归。

- [ ] **Step 1: 写入主/子 Agent timeline 与结果兼容测试**

```ts
it("preserves child tools and terminal results in notification history", () => {
  const timeline = messagesToTimeline(fixtureMessages);
  expect(timeline).toContainEqual(expect.objectContaining({ type: "tool", agentName: "asset_image_gen" }));
  expect(timeline).toContainEqual(expect.objectContaining({ type: "content" }));
});

it("applies each successful tool invalidation once", () => {
  const state = reduceFixture([toolFinished(4), toolFinished(4), done(5)]);
  expect(state.invalidation.assets).toBe(1);
});
```

- [ ] **Step 2: 运行兼容测试并记录任何真实差异**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/notification-pipeline-compat.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: 如果规范化层遗漏旧字段则 FAIL；失败必须通过适配事件 DTO 修复，不能重写通知面板结构。

- [ ] **Step 3: 在事件适配边界保留旧字段**

```ts
function toLegacyProjection(event: NormalizedPipelineEvent): AiChatStreamEvent {
  return {
    ...event,
    parentToolCallId: event.parentToolCallId ?? undefined,
    agentName: event.agentName ?? undefined,
    messageId: event.messageId,
    conversationId: event.conversationId,
  };
}
```

- [ ] **Step 4: 运行前端全量测试和 lint**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS，全部 unit tests passed。

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm lint; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS，无新增 ESLint error。

- [ ] **Step 5: 提交通知兼容测试**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video-web/tests/unit/notification-pipeline-compat.test.ts ai-fusion-video-web/lib/api/pipeline-event-normalizer.ts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "test(web): preserve pipeline notification behavior"
```

### Task 8: 完成 WAITING_CONFIRMATION 的可操作前端闭环

**Files:**
- Modify: `ai-fusion-video-web/lib/api/client.ts`
- Modify: `ai-fusion-video-web/lib/api/ai-assistant.ts`
- Modify: `ai-fusion-video-web/lib/api/ai-pipeline.ts`
- Modify: `ai-fusion-video-web/lib/store/pipeline-store.ts`
- Modify: `ai-fusion-video-web/components/dashboard/agent-pipeline/types.ts`
- Modify: `ai-fusion-video-web/components/dashboard/agent-pipeline/use-agent-pipeline.ts`
- Create: `ai-fusion-video-web/components/dashboard/agent-pipeline/agent-confirmation-card.tsx`
- Modify: `ai-fusion-video-web/components/dashboard/agent-pipeline/index.tsx`
- Create: `ai-fusion-video-web/tests/unit/pipeline-confirmation-api.test.ts`
- Create: `ai-fusion-video-web/tests/unit/pipeline-confirmation-store.test.ts`
- Create: `ai-fusion-video-web/tests/unit/agent-confirmation-card.test.tsx`
- Create: `ai-fusion-video-web/tests/unit/agent-pipeline-confirmation-integration.test.tsx`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/common/AgentRunConflictException.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AgentRunConflictRespVO.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/ConfirmPipelineRunRespVO.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/common/GlobalExceptionHandler.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/AiPipelineController.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/controller/ai/AiPipelineWaitingControllerTests.java`

**Interfaces:**
- Consumes: 已授权后端 `POST /api/ai/pipeline/confirm`、只在 StateStore 保存和 run CAS 成功后出现的 raw `PLATFORM_USER_CONFIRM_REQUIRED` + `controlType=USER_CONFIRM_REQUIRED` 事件，以及 `replyId/pendingToolCalls/expiresAt`。
- Produces: `ConfirmPipelineRunRespVO(runId,status,acceptedReplyId)`、HTTP 409 `AgentRunConflictRespVO(code,msg,errorCode,data)`、`confirmPipelineRun(request)`、`PendingPipelineConfirmation` 和 `AgentConfirmationCard`；批准/拒绝只提交 toolCallId 与 boolean，不回传或信任工具 input。
- Cross-plan gate: model/tool plan 的 `confirm` controller 必须在执行本任务前同步为这里冻结的成功/冲突响应；过期抛 `AgentRunConflictException("CONFIRMATION_EXPIRED",...)`，重复或旧 reply 抛 `AgentRunConflictException("CONFIRMATION_ALREADY_RESOLVED",...)`。两份计划不允许各自保留不同 DTO。

- [ ] **Step 1: 写入 API、store 与真实交互失败测试**

```ts
it("posts the exact persisted tool decision set", async () => {
  mockHttpPost.mockResolvedValue({ runId: "run-1", status: "RUNNING", acceptedReplyId: "reply-7" });
  const response = await confirmPipelineRun({
    runId: "run-1",
    replyId: "reply-7",
    decisions: [
      { toolCallId: "tool-a", confirmed: true },
      { toolCallId: "tool-b", confirmed: false },
    ],
  });
  expect(mockHttpPost).toHaveBeenCalledWith("/api/ai/pipeline/confirm", {
    runId: "run-1",
    replyId: "reply-7",
    decisions: [
      { toolCallId: "tool-a", confirmed: true },
      { toolCallId: "tool-b", confirmed: false },
    ],
  });
  expect(response).toEqual({ runId: "run-1", status: "RUNNING", acceptedReplyId: "reply-7" });
});
```

```ts
it("only exposes a confirmation after the persisted platform control event", () => {
  store.getState().acceptEvent(taskId, event({
    runId: "run-1", sequence: 9, outputType: "TOOL_CALL",
    rawEventType: "PLATFORM_USER_CONFIRM_REQUIRED",
    controlType: "USER_CONFIRM_REQUIRED", replyId: "reply-7",
    expiresAt: "2026-07-21T12:00:00Z",
    pendingToolCalls: [{ toolCallId: "tool-a", toolName: "generate_image", argumentsPreview: "{\"prompt\":\"sunrise\"}" }],
  }));
  expect(store.getState().tasks.find((task) => task.id === taskId)?.pendingConfirmation?.replyId)
    .toBe("reply-7");
});

it("clears the accepted reply and suppresses its stale replay after refresh", () => {
  store.getState().reconcileConfirmationAccepted(taskId, {
    runId: "run-1", status: "RUNNING", acceptedReplyId: "reply-7",
  });
  expect(store.getState().tasks.find((task) => task.id === taskId)?.pendingConfirmation).toBeUndefined();

  store.getState().setAuthoritativeRunStatus(taskId, "RUNNING");
  store.getState().acceptEvent(taskId, event({
    runId: "run-1", sequence: 9, outputType: "TOOL_CALL",
    rawEventType: "PLATFORM_USER_CONFIRM_REQUIRED",
    controlType: "USER_CONFIRM_REQUIRED", replyId: "reply-7",
    expiresAt: "2026-07-21T12:00:00Z",
    pendingToolCalls: [{ toolCallId: "tool-a", toolName: "generate_image", argumentsPreview: "{}" }],
  }));
  expect(store.getState().tasks.find((task) => task.id === taskId)?.pendingConfirmation).toBeUndefined();
});
```

```tsx
/** @vitest-environment jsdom */
it("disables duplicate submit and reports expiry without synthesizing RUNNING", async () => {
  const user = userEvent.setup();
  confirmMock.mockRejectedValue(new ApiClientError("expired", "CONFIRMATION_EXPIRED", 409));
  render(<AgentConfirmationCard confirmation={pending} onAccepted={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "批准全部" }));
  expect(confirmMock).toHaveBeenCalledOnce();
  expect(await screen.findByText("确认已过期，请刷新任务状态")).toBeVisible();
  expect(screen.getByRole("button", { name: "批准全部" })).toBeDisabled();
});

it("rejects the complete persisted tool set and reconciles the accepted response", async () => {
  const user = userEvent.setup();
  const onAccepted = vi.fn();
  confirmMock.mockResolvedValue({ runId: pending.runId, status: "RUNNING", acceptedReplyId: pending.replyId });
  render(<AgentConfirmationCard confirmation={pending} onAccepted={onAccepted} />);
  await user.click(screen.getByRole("button", { name: "拒绝全部" }));
  expect(confirmMock).toHaveBeenCalledWith({
    runId: pending.runId,
    replyId: pending.replyId,
    decisions: pending.tools.map((tool) => ({ toolCallId: tool.toolCallId, confirmed: false })),
  });
  expect(onAccepted).toHaveBeenCalledWith({
    runId: pending.runId, status: "RUNNING", acceptedReplyId: pending.replyId,
  });
  expect(await screen.findByText("决定已提交，等待 Agent 恢复")).toBeVisible();
});

it("submits a mixed per-tool decision without trusting the displayed arguments", async () => {
  const user = userEvent.setup();
  confirmMock.mockResolvedValue({ runId: pending.runId, status: "RUNNING", acceptedReplyId: pending.replyId });
  render(<AgentConfirmationCard confirmation={pending} onAccepted={vi.fn()} />);
  await user.click(screen.getByLabelText(`批准 ${pending.tools[0].toolName}`));
  await user.click(screen.getByLabelText(`拒绝 ${pending.tools[1].toolName}`));
  await user.click(screen.getByRole("button", { name: "提交逐项决定" }));
  expect(confirmMock).toHaveBeenCalledWith({
    runId: pending.runId,
    replyId: pending.replyId,
    decisions: [
      { toolCallId: pending.tools[0].toolCallId, confirmed: true },
      { toolCallId: pending.tools[1].toolCallId, confirmed: false },
    ],
  });
});

it("renders the card through index and resets local choices when replyId changes", async () => {
  hookMock.mockReturnValue(hookState({ pendingConfirmation: pending }));
  const view = render(<AgentPipeline request={request} />);
  await userEvent.click(screen.getByLabelText(`批准 ${pending.tools[0].toolName}`));

  const next = { ...pending, replyId: "reply-8", tools: [{ ...pending.tools[0], toolCallId: "tool-c" }] };
  hookMock.mockReturnValue(hookState({ pendingConfirmation: next }));
  view.rerender(<AgentPipeline request={request} />);

  expect(screen.queryByLabelText(`批准 ${pending.tools[1].toolName}`)).not.toBeInTheDocument();
  expect(screen.getByLabelText(`批准 ${next.tools[0].toolName}`)).not.toBeChecked();
});

it("removes the index card after the accepted response is reconciled", async () => {
  confirmMock.mockResolvedValue({
    runId: pending.runId, status: "RUNNING", acceptedReplyId: pending.replyId,
  });
  hookMock.mockImplementation(() => {
    const [pendingConfirmation, setPendingConfirmation] = useState(pending);
    return hookState({
      pendingConfirmation,
      reconcileConfirmationAccepted: () => setPendingConfirmation(undefined),
    });
  });
  render(<AgentPipeline request={request} />);
  await userEvent.click(screen.getByRole("button", { name: "拒绝全部" }));
  await waitFor(() => expect(screen.queryByLabelText("工具执行确认")).not.toBeInTheDocument());
});

it("explains a duplicate cross-page response", async () => {
  const user = userEvent.setup();
  confirmMock.mockRejectedValue(new ApiClientError("conflict", 40901, 409));
  render(<AgentConfirmationCard confirmation={pending} onAccepted={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "批准全部" }));
  expect(await screen.findByText("该确认已由其他页面或节点处理")).toBeVisible();
});
```

```java
@Test
void confirmExpiryReturnsHttp409AndMachineReadableCode() throws Exception {
    when(confirmations.resume(any(), eq(currentUserId)))
        .thenReturn(Mono.error(new AgentRunConflictException(
            "CONFIRMATION_EXPIRED", "确认已过期")));
    MvcResult pending = mockMvc.perform(post("/api/ai/pipeline/confirm")
            .contentType(APPLICATION_JSON).content(validConfirmJson()))
        .andExpect(request().asyncStarted()).andReturn();
    mockMvc.perform(asyncDispatch(pending))
        .andExpect(status().isConflict())
        .andExpect(jsonPath("$.code").value(409))
        .andExpect(jsonPath("$.errorCode").value("CONFIRMATION_EXPIRED"));
}

@Test
void duplicateReplyReturnsDedicatedConflictCode() throws Exception {
    when(confirmations.resume(any(), eq(currentUserId)))
        .thenReturn(Mono.error(new AgentRunConflictException(
            "CONFIRMATION_ALREADY_RESOLVED", "确认已处理")));
    MvcResult pending = mockMvc.perform(post("/api/ai/pipeline/confirm")
            .contentType(APPLICATION_JSON).content(validConfirmJson()))
        .andExpect(request().asyncStarted()).andReturn();
    mockMvc.perform(asyncDispatch(pending))
        .andExpect(status().isConflict())
        .andExpect(jsonPath("$.errorCode").value("CONFIRMATION_ALREADY_RESOLVED"));
}
```

- [ ] **Step 2: 运行定向测试并确认确认能力尚不存在**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/pipeline-confirmation-api.test.ts tests/unit/pipeline-confirmation-store.test.ts tests/unit/agent-confirmation-card.test.tsx tests/unit/agent-pipeline-confirmation-integration.test.tsx; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd "-Dtest=AiPipelineWaitingControllerTests" test; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: FAIL；`confirmPipelineRun`、pending confirmation reducer 和确认卡片至少一项不存在。

- [ ] **Step 3: 冻结后端成功/冲突响应并保留前端 HTTP 状态与错误码**

```ts
export class ApiClientError extends Error {
  constructor(
    message: string,
    readonly code?: string | number,
    readonly httpStatus?: number,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

// response/error interceptor 都使用该类型；401 刷新失败仍执行原认证清理。
function toApiClientError(payload: unknown, httpStatus?: number): ApiClientError {
  const body = payload as { code?: string | number; errorCode?: string; msg?: string } | undefined;
  return new ApiClientError(body?.msg ?? "请求失败", body?.errorCode ?? body?.code, httpStatus);
}

// 成功分支中的 CommonResult 非零与非 401 HTTP 失败分支分别替换原来的 new Error(msg)。
if (result.code !== 0) {
  return Promise.reject(new ApiClientError(result.msg || "请求失败", result.code, response.status));
}
if (!originalRequest || error.response?.status !== 401) {
  return Promise.reject(toApiClientError(error.response?.data, error.response?.status));
}
```

```java
@Getter
public final class AgentRunConflictException extends RuntimeException {
    private final String errorCode;

    public AgentRunConflictException(String errorCode, String message) {
        super(message);
        this.errorCode = Objects.requireNonNull(errorCode, "errorCode");
    }
}

public record AgentRunConflictRespVO(
        int code, String msg, String errorCode, Void data) {
    public static AgentRunConflictRespVO conflict(AgentRunConflictException error) {
        return new AgentRunConflictRespVO(409, error.getMessage(), error.getErrorCode(), null);
    }
}

public record ConfirmPipelineRunRespVO(
        String runId, String status, String acceptedReplyId) {}

@ExceptionHandler(AgentRunConflictException.class)
@ResponseStatus(HttpStatus.CONFLICT)
public AgentRunConflictRespVO handleAgentRunConflict(AgentRunConflictException error) {
    log.warn("Agent run conflict: errorCode={}", error.getErrorCode());
    return AgentRunConflictRespVO.conflict(error);
}

@PostMapping("/confirm")
public Mono<CommonResult<ConfirmPipelineRunRespVO>> confirm(
        @Valid @RequestBody AgentConfirmReqVO request) {
    long currentUserId = requireCurrentUserId();
    return confirmations.resume(ConfirmRequest.from(request), currentUserId)
        .thenReturn(CommonResult.success(new ConfirmPipelineRunRespVO(
            request.runId(), "RUNNING", request.replyId())));
}
```

不得把字符串 errorCode 塞入现有整数 `BusinessException.code`。WAITING service 的过期、重复/旧 reply 和 CAS loser 分支必须抛上述专用异常；`GlobalExceptionHandler` 的专用 handler 必须位于通用 `BusinessException`/`Exception` handler 之外，并真实设置 HTTP 409。

- [ ] **Step 4: 冻结确认 DTO 与 API，客户端不得发送工具输入**

```ts
export interface PendingToolConfirmation {
  toolCallId: string;
  toolName: string;
  argumentsPreview: string;
}

export interface PendingPipelineConfirmation {
  runId: string;
  replyId: string;
  expiresAt: string;
  tools: PendingToolConfirmation[];
}

export interface AiChatStreamEvent {
  // 保留已有字段
  controlType?: "USER_CONFIRM_REQUIRED";
  replyId?: string;
  rawEventType?: string; // normalizePipelineEvent 在进入 store 前按 31 项 GA + 已冻结 PLATFORM_* allowlist 收窄
  pendingToolCalls?: PendingToolConfirmation[];
  expiresAt?: string;
}

export interface ConfirmPipelineRunRequest {
  runId: string;
  replyId: string;
  decisions: Array<{ toolCallId: string; confirmed: boolean }>;
}

export interface ConfirmPipelineRunResponse {
  runId: string;
  status: "RUNNING";
  acceptedReplyId: string;
}

export function confirmPipelineRun(request: ConfirmPipelineRunRequest): Promise<ConfirmPipelineRunResponse> {
  return http.post<never, ConfirmPipelineRunResponse>("/api/ai/pipeline/confirm", request);
}
```

- [ ] **Step 5: 只由持久化控制事件创建卡片，并用确认响应/authoritative status 清理 stale replay**

```ts
if (event.controlType === "USER_CONFIRM_REQUIRED") {
  const currentStatus = task.state.authoritativeRunStatus;
  if (currentStatus === undefined || currentStatus === "WAITING_CONFIRMATION") {
    task.state.authoritativeRunStatus = "WAITING_CONFIRMATION";
    task.pendingConfirmation = {
      runId: event.runId,
      replyId: requireString(event.replyId, "replyId"),
      expiresAt: requireString(event.expiresAt, "expiresAt"),
      tools: requirePendingToolCalls(event.pendingToolCalls),
    };
  }
}
if (event.rawEventType === "USER_CONFIRM_RESULT" || isTerminalEvent(event)) {
  task.pendingConfirmation = undefined;
}

function reconcileConfirmationAccepted(task: PipelineTask, response: ConfirmPipelineRunResponse): void {
  if (task.state.runId !== response.runId) throw new Error("CONFIRMATION_RUN_ID_MISMATCH");
  if (task.pendingConfirmation?.replyId === response.acceptedReplyId) {
    task.pendingConfirmation = undefined;
  }
  task.state.authoritativeRunStatus = response.status;
}

function setAuthoritativeRunStatus(task: PipelineTask, status: PipelineRunStatusCode): void {
  task.state.authoritativeRunStatus = status;
  if (status !== "WAITING_CONFIRMATION") task.pendingConfirmation = undefined;
}

interface PipelineStoreState {
  reconcileConfirmationAccepted(taskId: string, response: ConfirmPipelineRunResponse): void;
  setAuthoritativeRunStatus(taskId: string, status: PipelineRunStatusCode): void;
}

const reconcileConfirmationAccepted = useCallback((response: ConfirmPipelineRunResponse) => {
  setState((previous) => {
    if (previous.runId !== response.runId) throw new Error("CONFIRMATION_RUN_ID_MISMATCH");
    return {
      ...previous,
      authoritativeRunStatus: response.status,
      pendingConfirmation: previous.pendingConfirmation?.replyId === response.acceptedReplyId
        ? undefined
        : previous.pendingConfirmation,
    };
  });
}, []);
```

Reducer 必须校验 `pendingToolCalls` 非空、toolCallId 唯一、expiry 可解析；无效控制事件作为 protocol error 拒绝且不推进 cursor。初始 live stream 尚无 authoritative snapshot，可由持久化控制事件进入 WAITING；refresh/reconnect 必须先保存 `running/status` endpoint 的当前状态，当前状态不是 `WAITING_CONFIRMATION` 时忽略历史 actionable projection。成功响应只按服务端返回的 `acceptedReplyId/status` 清卡并更新 authoritative status，不本地合成工具结果或终态。

- [ ] **Step 6: 实现可访问、可防重且能解释失败的确认卡片**

```tsx
interface Props {
  confirmation: PendingPipelineConfirmation;
  onAccepted: (response: ConfirmPipelineRunResponse) => void;
}

function useExpiryClock(expiresAt: string): boolean {
  const deadline = Date.parse(expiresAt);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return !Number.isFinite(deadline) || now >= deadline;
}

export function AgentConfirmationCard({ confirmation, onAccepted }: Props) {
  const [submitting, setSubmitting] = useState(false);
  const [settled, setSettled] = useState(false);
  const [feedback, setFeedback] = useState<string>();
  const [decisions, setDecisions] = useState<Record<string, boolean | undefined>>(() =>
    Object.fromEntries(confirmation.tools.map((tool) => [tool.toolCallId, undefined])),
  );
  const apparentlyExpired = useExpiryClock(confirmation.expiresAt);

  const submit = async (forced?: boolean) => {
    if (submitting || settled) return;
    const selected = confirmation.tools.map((tool) => ({
      toolCallId: tool.toolCallId,
      confirmed: forced ?? decisions[tool.toolCallId],
    }));
    if (selected.some((decision) => decision.confirmed === undefined)) {
      setFeedback("请先为每个工具选择批准或拒绝");
      return;
    }
    setSubmitting(true);
    try {
      const response = await confirmPipelineRun({
        runId: confirmation.runId,
        replyId: confirmation.replyId,
        decisions: selected as Array<{ toolCallId: string; confirmed: boolean }>,
      });
      setFeedback("决定已提交，等待 Agent 恢复");
      setSettled(true);
      onAccepted(response);
    } catch (error) {
      const apiError = error as ApiClientError;
      setFeedback(apiError.code === "CONFIRMATION_EXPIRED"
        ? "确认已过期，请刷新任务状态"
        : apiError.httpStatus === 409
          ? "该确认已由其他页面或节点处理"
          : "提交失败，请重试");
      if (apiError.code === "CONFIRMATION_EXPIRED" || apiError.httpStatus === 409) setSettled(true);
    } finally {
      setSubmitting(false);
    }
  };

  const disabled = submitting || settled;
  const allSelected = confirmation.tools.every((tool) => decisions[tool.toolCallId] !== undefined);
  return (
    <section aria-label="工具执行确认">
      <h3>Agent 请求执行以下工具</h3>
      <ul>
        {confirmation.tools.map((tool) => (
          <li key={tool.toolCallId}>
            <strong>{tool.toolName}</strong>
            <pre>{tool.argumentsPreview}</pre>
            <label>
              <input
                type="radio"
                name={`decision-${tool.toolCallId}`}
                checked={decisions[tool.toolCallId] === true}
                onChange={() => setDecisions((current) => ({ ...current, [tool.toolCallId]: true }))}
              />
              批准 {tool.toolName}
            </label>
            <label>
              <input
                type="radio"
                name={`decision-${tool.toolCallId}`}
                checked={decisions[tool.toolCallId] === false}
                onChange={() => setDecisions((current) => ({ ...current, [tool.toolCallId]: false }))}
              />
              拒绝 {tool.toolName}
            </label>
          </li>
        ))}
      </ul>
      <time dateTime={confirmation.expiresAt}>有效期至 {confirmation.expiresAt}</time>
      {apparentlyExpired && <p>本机估算已到期，提交后由服务端数据库时间最终校验</p>}
      {feedback && <p role="status">{feedback}</p>}
      <button type="button" disabled={disabled || !allSelected} onClick={() => void submit()}>提交逐项决定</button>
      <button type="button" disabled={disabled} onClick={() => void submit(true)}>批准全部</button>
      <button type="button" disabled={disabled} onClick={() => void submit(false)}>拒绝全部</button>
    </section>
  );
}
```

```tsx
{state.pendingConfirmation && (
  <AgentConfirmationCard
    key={state.pendingConfirmation.replyId}
    confirmation={state.pendingConfirmation}
    onAccepted={reconcileConfirmationAccepted}
  />
)}
```

卡片逐项展示工具名与只读参数摘要，并允许每个工具分别批准或拒绝；“批准全部/拒绝全部”只是便利入口。父组件必须以 `replyId` 作为 React `key`，避免下一次确认复用上一次本地选择。卡片显示绝对到期时间和本机估算剩余时间；本机时钟不能作为授权真相，因此仅因本机估算到期不禁止首次提交，最终由服务端数据库时间判定。提交中、服务端确认已过期/已处理或成功提交后按钮禁用。所有入口都提交完整 toolCallId 集合，避免部分集合绕过服务端一致性校验。

- [ ] **Step 7: 运行 WAITING 前端回归、lint 与 build**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test -- tests/unit/pipeline-confirmation-api.test.ts tests/unit/pipeline-confirmation-store.test.ts tests/unit/agent-confirmation-card.test.tsx tests/unit/agent-pipeline-confirmation-integration.test.tsx tests/unit/pipeline-event-normalizer.test.ts tests/unit/pipeline-store.test.ts; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd "-Dtest=AiPipelineWaitingControllerTests" test; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS；覆盖逐项混合决定、批准全部、拒绝全部、双击防重、本机到期提示、服务端 `CONFIRMATION_EXPIRED`、重复/跨页 409、成功后等待回放。

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm lint; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; corepack pnpm build; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: 两条均 PASS，无 hydration、可访问性或类型错误。

- [ ] **Step 8: 提交确认 UI 闭环**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video-web/lib/api/client.ts ai-fusion-video-web/lib/api/ai-assistant.ts ai-fusion-video-web/lib/api/ai-pipeline.ts ai-fusion-video-web/lib/store/pipeline-store.ts ai-fusion-video-web/components/dashboard/agent-pipeline/types.ts ai-fusion-video-web/components/dashboard/agent-pipeline/use-agent-pipeline.ts ai-fusion-video-web/components/dashboard/agent-pipeline/agent-confirmation-card.tsx ai-fusion-video-web/components/dashboard/agent-pipeline/index.tsx ai-fusion-video-web/tests/unit/pipeline-confirmation-api.test.ts ai-fusion-video-web/tests/unit/pipeline-confirmation-store.test.ts ai-fusion-video-web/tests/unit/agent-confirmation-card.test.tsx ai-fusion-video-web/tests/unit/agent-pipeline-confirmation-integration.test.tsx ai-fusion-video/src/main/java/com/stonewu/fusion/common/AgentRunConflictException.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AgentRunConflictRespVO.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/ConfirmPipelineRunRespVO.java ai-fusion-video/src/main/java/com/stonewu/fusion/common/GlobalExceptionHandler.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/AiPipelineController.java ai-fusion-video/src/test/java/com/stonewu/fusion/controller/ai/AiPipelineWaitingControllerTests.java
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "feat(web): handle durable tool confirmations"
```

### Task 9: 切换生产入口并删除已完成 GA 编译过渡的 legacy runtime bridge

**Files:**
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeShutdownConfig.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/DefaultRunExecutionSupervisor.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/OwnedExecutionRegistry.java`
- Delete: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/StreamingEventHook.java`
- Delete: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentCancellationToken.java`
- Delete: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentCancelledException.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantServiceTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/RunExecutionSupervisorTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/build/NoAgentScopeV1RuntimeTests.java`

**Interfaces:**
- Consumes: model/tool 阶段已经定义且只构建一次 snapshot/spec/runtime 的 `startRoot(StartRootAgentRequest)`、`RunExecutionSupervisor`、`OwnedExecutionRegistry`、`AgentRunReplayService`、`CancellationCoordinator`、`HarnessLeaseCache`。
- Produces: `RunExecutionSupervisor.start/resume` 在 server-owned handle 注册且订阅启动成功后返回 admission completion，不返回 execution completion；生产 facade 只调用 `startRoot` 后 attach replay。优雅停机先拒绝新 run、drain、持久取消，再关闭 owned resources。`StreamingEventHook` 已在原子依赖任务中真实通过 GA 全量编译，本任务在 durable `streamEvents` 路径已就绪后删除这层短期 bridge，不曾与 V1 artifact 双运行。

- [ ] **Step 1: 写入源码扫描和入口委托失败测试**

```java
@Test
void productionSourcesContainNoV1RuntimeImports() throws IOException {
    String sources = Files.walk(Path.of("src/main/java"))
        .filter(path -> path.toString().endsWith(".java"))
        .map(NoAgentScopeV1RuntimeTests::read)
        .collect(Collectors.joining("\n"));
    assertThat(sources).doesNotContain("io.agentscope.core.session.");
    assertThat(sources).doesNotContain("StreamingEventHook");
    assertThat(sources).doesNotContain("AnthropicAgentScopeProxySupport");
    assertThat(sources).doesNotContain("ProxyAwareAnthropicChatModel");
    assertThat(sources).doesNotContain("GeminiToolResponseAwareChatFormatter");
    assertThat(sources).doesNotContain("VertexAgentScopeProxySupport");
}
```

```java
@Test
void startReturnsAfterServerOwnedAdmissionAndObserverDisposalDoesNotInterruptRun() {
    StepVerifier.create(supervisor.start(command)).verifyComplete();
    assertThat(executions.isOwned(command.run().runId(), command.run().ownerEpoch())).isTrue();
    assertThat(fakeExecution.isRunning()).isTrue();

    StepVerifier.create(replayService.replayThenLive(command.run().runId(), 0L))
        .thenCancel()
        .verify();

    assertThat(fakeExecution.isRunning()).isTrue();
    assertThat(fakeExecution.interruptCount()).isZero();
}

@Test
void resumeAlsoReturnsAdmissionInsteadOfExecutionCompletion() {
    StepVerifier.create(supervisor.resume(resumeCommand)).verifyComplete();
    assertThat(executions.isOwned(resumeCommand.run().runId(), resumeCommand.run().newOwnerEpoch())).isTrue();
    assertThat(fakeExecution.completion().isTerminated()).isFalse();
}

@Test
void facadeDelegatesToFrozenStartRootBeforeAttachingReplay() {
    when(startRoot(rootRequest)).thenReturn(Mono.just(started));
    when(replayService.replayThenLive(started.runId(), 0L)).thenReturn(Flux.just(firstEvent));
    StepVerifier.create(service.run(httpRequest, currentUserId).take(1))
        .expectNextMatches(sse -> sse.id().equals(started.runId() + ":1"))
        .verifyComplete();
    verify(service).startRoot(rootRequest);
    verify(replayService).replayThenLive(started.runId(), 0L);
}
```

GA 仍包含重叠 FQN `Msg`、`Model`、`ReActAgent`、`AgentTool`、`ToolCallParam` 和 `core.hook.*`，测试不得把这些名称本身当作 V1 证据；只扫描已核验的旧 artifact、session 和已删除兼容类。强类型 `UserMessage/AssistantMessage` 的构造契约由 `AgentScopeMessageMapperTests` 正向验证。

- [ ] **Step 2: 运行测试并确认 V1 路径仍存在**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd "-Dtest=NoAgentScopeV1RuntimeTests,AgentScopeAssistantServiceTests,RunExecutionSupervisorTests" test; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: 若原子依赖切换有遗漏则 FAIL；正常情况下扫描已通过，但 AssistantService facade 委托测试仍因旧 conversation subscription 所有权而 FAIL。

- [ ] **Step 3: 让 supervisor 返回 server-owned admission，并把 AssistantService 收窄为 startRoot facade**

```java
// OwnedExecutionRegistry：先注册未启动 handle，再启动 owned subscription；返回值只代表 admission。
public Mono<Void> admitAndStart(
        AgentExecution execution,
        Function<Flux<AgentEventEnvelope>, Publisher<Void>> consume) {
    return Mono.defer(() -> {
        AgentExecutionHandle handle = AgentExecutionHandle.unstarted(execution);
        if (!register(handle)) return Mono.error(new IllegalStateException("RUN_ALREADY_OWNED"));
        try {
            handle.start(consume); // 内部订阅 execution completion，并在 doFinally 中 remove/close。
            return Mono.empty();
        } catch (Throwable failure) {
            remove(handle.identity());
            handle.close();
            return Mono.error(failure);
        }
    });
}

// DefaultRunExecutionSupervisor：start/resume 都复用此 admission 边界。
private Mono<Void> startResolved(String runId, String owner, long epoch,
        List<Msg> messages, AgentKernelSpec spec,
        AgentScopeRuntimeContextRequest runtimeRequest) {
    return factory.start(runId, owner, epoch, messages, spec, runtimeRequest)
        .flatMap(execution -> executions.admitAndStart(execution,
            events -> chunks.coalesce(events)
                .concatMap(event -> journal.appendOwned(runId, owner, epoch, event))));
}

public Flux<ServerSentEvent<AiChatStreamRespVO>> run(AiChatReqVO request, long userId) {
    return startRoot(toStartRootAgentRequest(request, userId))
        .flatMapMany(started -> replayService.replayThenLive(started.runId(), 0L))
        .map(this::toSse);
}

public Mono<Void> cancel(String runId, long userId) {
    return cancellationCoordinator.cancel(runId, userId).then();
}
```

`AgentExecutionHandle.start` 是唯一允许订阅 execution completion 的 server lifecycle 边界：订阅前 handle 已进入 registry；订阅创建失败则回滚 registry；completion/error/cancel 都执行一次 remove/close，残余 error 必须记录且 execution pipeline 已通过 terminal coordinator 持久化。controller/facade/replay 返回的 publisher 不持有该 subscription，取消 observer 只能关闭 replay observer。

- [ ] **Step 4: 删除 legacy Hook/token/exception 并接入优雅停机；不得关闭 Spring 共享 AgentStateStore/Redis 连接**

```java
@Override
public void stop(Runnable callback) {
    executionSupervisor.shutdown(runtimeProperties.shutdownDrainTimeout())
        .then(harnessLeaseCache.drainAndClose(runtimeProperties.shutdownDrainTimeout()))
        .doFinally(signal -> callback.run())
        .subscribe(ignored -> { }, error -> log.error("AgentScope runtime shutdown failed", error));
}
```

- [ ] **Step 5: 运行 V1 扫描、AgentScope 定向和后端全量测试**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd "-Dtest=NoAgentScopeV1RuntimeTests,AgentScopeAssistantServiceTests,RunExecutionSupervisorTests" test; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS。

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd test; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS，全部后端单元测试通过；不得保留已知 fixture failure。

- [ ] **Step 6: 提交生产切换**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeShutdownConfig.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/DefaultRunExecutionSupervisor.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/OwnedExecutionRegistry.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/StreamingEventHook.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentCancellationToken.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentCancelledException.java ai-fusion-video/src/test/java/com/stonewu/fusion/build/NoAgentScopeV1RuntimeTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantServiceTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/RunExecutionSupervisorTests.java
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "refactor(ai): cut over AgentScope V2 runtime"
```

### Task 10: 编写可执行切换和回滚手册

**Files:**
- Create: `docs/operations/agentscope-v2-cutover.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Migration `V1.0.6.1.5`、run/outbox 状态、V1/V2 二进制。
- Produces: 明确的 preflight、停写、切换、回滚、重新启用 V2 SQL 和判定条件。

- [ ] **Step 1: 写入手册的强制 preflight 命令**

```markdown
## Preflight

1. 进入维护停写窗口，关闭 `/api/ai/pipeline/run` 新写入。
2. 执行 `SELECT VERSION()`，确认 MySQL >= 8.0.16。
3. 备份 `afv_agent_conversation`、`afv_agent_message` 和 Flyway history。
4. 运行 `./mvnw.cmd -Pagentscope-integration verify`，必须通过而非跳过。
5. 确认 `SELECT COUNT(*) FROM afv_agent_run WHERE status IN ('RUNNING','WAITING_CONFIRMATION','WAITING_EXTERNAL','CANCEL_REQUESTED')` 为 0 后才替换二进制。
```

- [ ] **Step 2: 写入回滚顺序和重新启用 V2 对账 SQL**

```sql
UPDATE afv_agent_conversation c
LEFT JOIN (
  SELECT conversation_id, COALESCE(MAX(message_order), 0) + 1 AS next_order
  FROM afv_agent_message
  GROUP BY conversation_id
) m ON m.conversation_id = c.conversation_id
SET c.next_message_order = GREATEST(c.next_message_order, COALESCE(m.next_order, 1));
```

```markdown
回滚只替换应用二进制，不回滚已执行 Flyway。先停止入口，再取消或终态化活动 V2 run，停止 owner/reconciler，等待 outbox 清空或记录保留原因，最后启动 V1。重新启用 V2 前必须再次停写并运行 next_message_order 对账；禁止 V1/V2 writer 无协调混跑。
```

- [ ] **Step 3: 验证文档中的所有路径、profile 和 SQL 名称与实现一致**

Run: `Set-Location 'D:\develop\my\ai-fusion-video'; $required = 'V1\.0\.6\.1\.5','agentscope-integration','agentscope-provider-smoke','ark-smoke','next_message_order','CANCEL_REQUESTED'; foreach ($pattern in $required) { rg -n $pattern docs/operations/agentscope-v2-cutover.md ai-fusion-video/pom.xml ai-fusion-video/src/main/resources/db/migration; if ($LASTEXITCODE -ne 0) { throw "missing required cutover token: $pattern" } }`

Expected: 每个 profile、Migration 和状态名均有实现命中；不存在拼写漂移。

- [ ] **Step 4: 提交操作手册**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add docs/operations/agentscope-v2-cutover.md README.md
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "docs: add AgentScope V2 cutover runbook"
```

### Task 11: 执行最终验收并记录真实证据

**Files:**
- Create: `docs/verification/2026-07-21-agentscope-v2-ga-verification.md`

**Interfaces:**
- Consumes: 四个阶段计划完成后的同一 commit。
- Produces: 命令、时间、环境、退出码、测试数量和未验证项；这是 P-1 完成判定的唯一汇总证据。

- [ ] **Step 1: 验证依赖树只有 2.0.0 GA**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd dependency:tree "-Dincludes=io.agentscope"; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: 所有 `io.agentscope` 行为 `2.0.0`；没有 `1.0.12`、`RC`、`agentscope-spring-boot-starter`、`agentscope-extensions-session-mysql`。

- [ ] **Step 2: 运行后端单元测试和打包**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd test; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: BUILD SUCCESS，0 failures，0 errors，0 skipped-by-assumption 的 AgentScope 核心测试。

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd package; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: BUILD SUCCESS，应用 JAR 成功生成。

- [ ] **Step 3: 运行 MySQL/Redis 多实例集成测试**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd -Pagentscope-integration verify; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: BUILD SUCCESS；MySQL 8 与 Redis Testcontainers 实际启动，Migration、fencing、outbox、replay-live、跨实例恢复均执行。

- [ ] **Step 4: 运行真实 Provider 和 Ark smoke**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd -Pagentscope-provider-smoke verify; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: 五个 Provider 最小流式 smoke 全部 PASS；如任一凭据缺失，profile 必须快速失败，并在验证文档记录“未验证”。

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video'; .\mvnw.cmd -Park-smoke verify; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: Ark 图片/视频 smoke PASS；缺少 Ark 凭据时 profile 必须快速失败并在验证文档中明确记为“未验证”，不得写成 PASS。

- [ ] **Step 5: 运行前端冻结安装、测试、lint 和生产构建**

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm install --frozen-lockfile; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS，lockfile 无变更。

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm test; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS，parser、normalizer、cursor、store、hook、通知兼容测试全部执行。

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm lint; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS，0 errors。

Run: `Set-Location 'D:\develop\my\ai-fusion-video\ai-fusion-video-web'; corepack pnpm build; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`

Expected: PASS，Next.js production build 完成。

- [ ] **Step 6: 写入验证证据，不美化缺失凭据或失败**

```markdown
| 命令 | 环境/凭据 | 退出码 | 结果 | 测试数/关键证据 |
|---|---|---:|---|---|
| `./mvnw.cmd test` | Temurin Java 21.0.11（或执行时记录的兼容 Java 21 patch） | 0 | PASS | 记录 Surefire `Tests run` 原文 |
| `./mvnw.cmd -Pagentscope-integration verify` | Docker/MySQL8/Redis | 0 | PASS | 记录 Failsafe `Tests run` 原文和容器版本 |
| `./mvnw.cmd -Pagentscope-provider-smoke verify` | 五 Provider 凭据 | 实际退出码 | PASS/未验证 | 逐 Provider 记录 |
| `./mvnw.cmd -Park-smoke verify` | Ark 凭据 | 实际退出码 | PASS/未验证 | 图片/视频分别记录 |
| `corepack pnpm test` | pnpm 10.32.1 | 0 | PASS | 记录 Vitest `Test Files/Tests` 原文 |
| `corepack pnpm lint && corepack pnpm build` | Node 运行时 | 0 | PASS | 记录 ESLint 退出码和 Next route/build 摘要 |
```

- [ ] **Step 7: 提交验收证据**

```powershell
Set-Location 'D:\develop\my\ai-fusion-video'
git add docs/verification/2026-07-21-agentscope-v2-ga-verification.md
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git commit -m "test: record AgentScope V2 GA verification"
```
