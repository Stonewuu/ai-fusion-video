/** 仅用于系统提示。创作正文、文件名和模型输出不经过这里。 */
export function userFacingError(message: string | null | undefined): string {
  let text = message?.trim();
  if (!text) return "操作遇到问题，请稍后重试。";
  const envelope = text.replace(/^Error:\s*/i, "");
  if (envelope.startsWith("{") || envelope.startsWith("[")) {
    try {
      const detail: unknown = JSON.parse(envelope);
      if (
        detail &&
        typeof detail === "object" &&
        "message" in detail &&
        typeof detail.message === "string"
      ) {
        text = detail.message.trim();
      } else {
        return "操作遇到问题，请稍后重试。";
      }
    } catch {
      return "操作遇到问题，请稍后重试。";
    }
  }
  if (text === "请求失败") return "暂时无法完成请求，请稍后重试。";
  if (/请求结果未确认|提交意图已保存|未取得(?:受理)?回执|调用状态待确认|UNKNOWN_SUBMISSION/.test(text)) {
    if (/video queue is full/i.test(text))
      return "视频服务繁忙，这次请求的受理状态尚未确认。请打开生成详情核实，避免重复生成。";
    return "这次请求的受理状态尚未确认。请打开任务详情核实，避免重复提交。";
  }
  const providerFailure = text.match(/(?:生成失败（HTTP (\d{3})）：|^.+?任务提交失败:\s*HTTP\s+(\d{3})\s+)([\s\S]+)$/);
  if (providerFailure) {
    const status = providerFailure[1] ?? providerFailure[2];
    const reason = providerFailure[3].replace(/\s*\(request[ _]id:\s*[^)]+\)/gi, "").trim();
    const guidance = /video queue is full/i.test(reason)
      ? "供应商队列已满。"
      : ["401", "403"].includes(status)
        ? "供应商拒绝访问，请检查模型配置与访问权限。"
        : status === "402" && /(?:credit|balance).*(?:low|insufficient|exhausted)|insufficient.*(?:credit|balance)/i.test(reason)
          ? "供应商余额不足。"
          : "";
    text = `生成失败（HTTP ${status}）：${guidance ? `${guidance}服务返回：` : ""}${reason}`;
  }
  if (
    text.includes("原模型调用仍需核实") ||
    text.includes("智能体模型调用未完整返回")
  )
    return "模型连接已中断，暂时无法确认本次调用是否结束。请打开本节点的任务详情处理。";
  if (text.includes("Expected canvas tool is unavailable"))
    return "所选模型未能处理当前智能体的工具。请在本节点选择支持当前任务的思考模型后重新运行。";
  const unsupportedMedia = text.match(/当前模型不支持 (image|video|audio|file) 的 (?:url|base64) 输入/);
  if (unsupportedMedia) {
    const names: Record<string, string> = { image: "图片", video: "视频", audio: "音频", file: "文件" };
    const name = names[unsupportedMedia[1]];
    return `当前模型无法读取这份${name}，请选择支持${name}输入的模型后继续。`;
  }
  const modelHttpStatus = text.match(
    /HTTP (?:request failed with status|(?:transport error.*?status))\s*(\d{3})/i,
  )?.[1];
  if (modelHttpStatus) {
    if (["401", "403"].includes(modelHttpStatus))
      return `模型服务拒绝了访问（HTTP ${modelHttpStatus}），请检查所选模型的访问权限。`;
    if (modelHttpStatus === "429")
      return "模型服务暂时限流（HTTP 429），请稍后重新运行。";
    return `模型服务返回错误（HTTP ${modelHttpStatus}），本次调用已结束，可以重新运行。`;
  }

  if (
    /network error|failed to fetch|fetch failed|ECONNREFUSED|ECONNRESET|ENOTFOUND|ERR_NETWORK/i.test(
      text,
    )
  ) {
    return "暂时无法连接服务，请检查网络后重试。";
  }
  if (/timeout|timed out|ETIMEDOUT/i.test(text)) {
    return "等待响应超时，请先查看当前操作状态，避免重复提交。";
  }
  if (/尚未实现|暂未实现|尚未接通|执行器未接通|真实验证|占位实现/.test(text)) {
    return "这项操作当前不可用，请选择其他创作方式。";
  }
  if (
    /无效.*工具.*参数|工具.*参数.*无效|图操作结构|节点协议|本批别名|不支持的画布节点类型|invalid.*tool.*(?:argument|json)/i.test(
      text,
    )
  ) {
    return "助手未能完成这一步，请补充要求后重新尝试。";
  }
  if (/节点.*(?:已删除|不存在)/.test(text)) {
    return "这个节点已被移除，请刷新画布后重新选择。";
  }
  if (/https?:\/\/\S+/i.test(text))
    return "无法访问所需资源，请检查链接或访问权限后重试。";
  if (
    /schemaVersion|DTO|Outbox|fencing|\b(?:epoch|nodeRunId|canvasRunId|commandId|planHash|expectedRevision|ownerEpoch|leaseUntil|toolCallId|configVersion|layoutVersion)\b|checksum|SQL(?:Syntax|Integrity|State)|jdbc:|java\.|org\.springframework|com\.stonewu|(?:Type|Reference|Syntax|Range)Error|Exception\b|stack trace|<!doctype|<html|Bearer\s|api[_-]?key\s*[:=]|\bsk-[a-z\d]{12,}|[A-Z]:[\\/]|[a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12}/i.test(
      text,
    )
  ) {
    return "操作遇到问题，请检查当前内容；若仍无法继续，请联系管理员。";
  }
  if (!/[\u3400-\u9fff]/.test(text) || text.length > 600) {
    return "操作遇到问题，请稍后重试。";
  }
  return text
    .replaceAll("真实业务对象", "项目内容")
    .replaceAll("真实业务内容", "项目内容")
    .replaceAll("正式业务对象", "项目内容")
    .replaceAll("正式写回", "应用结果")
    .replaceAll("冻结的", "本次运行使用的")
    .replaceAll("上游", "前置步骤")
    .replaceAll("下游", "后续步骤");
}
