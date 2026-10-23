/** 浏览器对 Markdown/CSV 的 MIME 并不一致；声明受支持的文件格式，服务端仍校验实际内容。 */
export function canvasUploadFile(file: File): File {
  const extension = file.name.split(".").pop()?.toLowerCase();
  const types: Record<string, string> = {
    md: "text/markdown",
    txt: "text/plain",
    csv: "text/csv",
    json: "application/json",
  };
  const type = extension ? types[extension] : undefined;
  return type && type !== file.type
    ? new File([file], file.name, { type, lastModified: file.lastModified })
    : file;
}
