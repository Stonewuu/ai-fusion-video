export function CanvasAgentModelStatus({ requestIds }: { requestIds: string[] }) {
  return (
    <div
      role="status"
      className="flex flex-col gap-2 rounded-lg border border-border/30 bg-muted/40 p-3 text-sm"
    >
      <p className="font-medium">远端结果未知</p>
      <p className="text-muted-foreground">
        {requestIds.length} 次模型调用未取得完整响应，已停止本地等待。可以继续发起任务，原请求不会自动重发。
      </p>
      <details>
        <summary className="cursor-pointer rounded-lg text-xs text-muted-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
          查看请求编号
        </summary>
        <ul className="mt-2 flex flex-col gap-2 font-mono text-xs text-muted-foreground">
          {requestIds.map((id) => (
            <li key={id} className="break-all">{id}</li>
          ))}
        </ul>
      </details>
    </div>
  );
}
