import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const currentFile = fileURLToPath(import.meta.url);
const appRoot = path.resolve(path.dirname(currentFile), "..");
const indexPath = path.join(appRoot, "dist", "index.html");
const indexHtml = readFileSync(indexPath, "utf8");

const assetRefs = Array.from(
  indexHtml.matchAll(/(?:src|href)="(\.\/assets\/[^"]+)"/g),
  (match) => match[1]
);

if (assetRefs.length === 0) {
  throw new Error("Electron build must reference renderer assets with ./assets/ paths.");
}

for (const assetRef of assetRefs) {
  const assetPath = path.join(appRoot, "dist", assetRef);
  if (!existsSync(assetPath)) {
    throw new Error(`Electron build references a missing asset: ${assetRef}`);
  }
}
