/** 服务端根据被冻结的命令与当前授权生成，不能用模型叙述代替此预览。 */
export interface ToolApprovalReview {
  title: string;
  summary: string;
  rows: Array<{ label: string; value: string }>;
  changes: Array<{ action: string; object: string; detail: string }>;
  warnings: string[];
}
