/** 按 SSE 行边界解码；CRLF 可以跨网络分片，不能把尾部 CR 和下一片 LF 当成两个换行。 */
export class SseEventDecoder {
  private line = "";
  private lines: string[] = [];
  private skipLf = false;

  constructor(private readonly onBlock: (block: string) => void) {}

  push(value: string) {
    for (const character of value) {
      if (this.skipLf) {
        this.skipLf = false;
        if (character === "\n") continue;
      }
      if (character === "\r" || character === "\n") {
        this.skipLf = character === "\r";
        if (this.line === "") {
          const block = this.lines.join("\n");
          this.lines = [];
          if (block) this.onBlock(block);
        } else {
          this.lines.push(this.line);
          this.line = "";
        }
      } else {
        this.line += character;
      }
    }
  }

  finish() {
    if ([...this.lines, this.line].some((line) => line.startsWith("data:") || line.startsWith("id:"))) {
      throw new Error("事件连接在完整事件到达前中断");
    }
  }
}
