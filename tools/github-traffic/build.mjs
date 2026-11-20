import { readFile, writeFile } from "node:fs/promises";
import { validateArchive } from "./archive.mjs";

const [historyFile, outputFile] = process.argv.slice(2);
if (!historyFile || !outputFile) {
  throw new Error("用法：node build.mjs 历史文件 报告文件");
}
const data = JSON.parse(await readFile(historyFile, "utf8"));
validateArchive(data, data.repository);
if (!data.snapshots.length) {
  throw new Error("报告至少需要一次成功采集的统计。");
}
const files = ["index.html", "report.css", "chart.js", "report.js"];
const [template, styles, chart, app] = await Promise.all(files.map(file =>
  readFile(new URL("./report/" + file, import.meta.url), "utf8")
));
const html = template
  .replace("<!--STYLES-->", () => "<style>\n" + styles + "\n</style>")
  .replace("<!--DATA-->", () => '<script id="traffic-data" type="application/json">' +
    JSON.stringify(data).replaceAll("<", "\\u003c") + "</script>")
  .replace("<!--APP-->", () => '<script type="module">\n' + chart + "\n" + app + "\n</script>");
await writeFile(outputFile, html);
