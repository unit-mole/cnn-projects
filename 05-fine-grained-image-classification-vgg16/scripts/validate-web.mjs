import { access, readFile, stat } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import process from "node:process";

const projectRoot = path.resolve(import.meta.dirname, "..");
const required = [
  "web/index.html",
  "web/style.css",
  "web/app.js",
  "web/metadata.json",
  "web/tfjs_model/model.json",
];

for (const relativePath of required) {
  await access(path.join(projectRoot, relativePath), constants.R_OK);
}

const modelPath = path.join(projectRoot, "web/tfjs_model/model.json");
const model = JSON.parse(await readFile(modelPath, "utf8"));
if (model.format !== "layers-model") {
  throw new Error(`Unexpected TensorFlow.js model format: ${model.format}`);
}

let shardBytes = 0;
let shardCount = 0;
for (const group of model.weightsManifest ?? []) {
  for (const relativePath of group.paths ?? []) {
    const shardPath = path.join(path.dirname(modelPath), relativePath);
    const details = await stat(shardPath);
    shardBytes += details.size;
    shardCount += 1;
  }
}
if (shardCount === 0 || shardBytes === 0) {
  throw new Error("TensorFlow.js model has no readable weight shards.");
}

const html = await readFile(path.join(projectRoot, "web/index.html"), "utf8");
if (!html.includes("@tensorflow/tfjs@4.22.0")) {
  throw new Error("index.html must pin the tested TensorFlow.js runtime version.");
}

console.log(`Web validation passed: ${shardCount} shards, ${shardBytes.toLocaleString()} weight bytes.`);
