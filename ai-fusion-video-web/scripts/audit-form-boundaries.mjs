import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, relative, join } from "node:path";
import ts from "typescript";

// Static inventory for visual review, not a substitute for browser focus checks.
const root = process.cwd();
const controls = new Set([
  "input", "textarea", "select", "Input", "Textarea", "SelectTrigger",
  "Checkbox", "Switch", "RadioGroupItem", "CanvasSelect",
]);
const inventory = [];

function paths(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return relative(root, path).replaceAll("\\", "/") === "components/ui" ? [] : paths(path);
    return entry.name.endsWith(".tsx") ? [path] : [];
  });
}

for (const path of [...paths(resolve(root, "app")), ...paths(resolve(root, "components"))]) {
  const source = ts.createSourceFile(path, readFileSync(path, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const line = (node) => source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;
  const attribute = (opening, name) => opening.attributes.properties.find((prop) => ts.isJsxAttribute(prop) && prop.name.text === name)?.initializer?.getText(source);
  function visit(node, ancestors = []) {
    if (ts.isJsxElement(node) || ts.isJsxSelfClosingElement(node)) {
      const opening = ts.isJsxElement(node) ? node.openingElement : node;
      const tag = opening.tagName.getText(source);
      const entry = { tag, line: line(opening), className: attribute(opening, "className") ?? null };
      if (controls.has(tag) || attribute(opening, "contentEditable")) {
        inventory.push({ file: relative(root, path).replaceAll("\\", "/"), ...entry,
          ancestors: ancestors.slice(-10),
          clippingAncestors: ancestors.filter((item) => /overflow-(?:[xy]-)?(?:auto|hidden|clip|scroll)/.test(item.className ?? "")),
          safeScrollBody: ancestors.some((item) => item.tag === "DialogScrollBody"),
        });
      }
      if (ts.isJsxElement(node)) node.children.forEach((child) => visit(child, [...ancestors, entry]));
      return;
    }
    ts.forEachChild(node, (child) => visit(child, ancestors));
  }
  visit(source);
}

const result = {
  generatedAt: new Date().toISOString(),
  sourceFiles: new Set(inventory.map((item) => item.file)).size,
  controlSites: inventory.length,
  controls: inventory,
};
if (process.argv[2]) writeFileSync(resolve(process.argv[2]), JSON.stringify(result, null, 2));
const risks = inventory.filter((item) => item.clippingAncestors.length && !item.safeScrollBody);
console.log(JSON.stringify({ sourceFiles: result.sourceFiles, controlSites: result.controlSites,
  reviewSites: risks.map(({ file, line, tag, clippingAncestors }) => ({ file, line, tag, clippingAncestors })) }, null, 2));
