import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const root = process.cwd();
const web = path.join(root, "web");
const modelDir = path.join(web, "tfjs_model");
const required = [
  "index.html",
  "style.css",
  "app.js",
  "metadata.json",
  "tfjs_model/model.json",
  "tfjs_model/weights_manifest.json",
  "tfjs_model/weights.bin",
  "tfjs_model/model_metadata.json"
];

for (const relative of required) {
  const target = path.join(web, relative);
  if (!fs.existsSync(target)) {
    throw new Error(`Missing Vercel web file: ${relative}`);
  }
}

const manifestPath = path.join(modelDir, "weights_manifest.json");
const descriptorPath = path.join(modelDir, "model.json");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const descriptor = JSON.parse(fs.readFileSync(descriptorPath, "utf8"));
const weightsPath = path.join(modelDir, manifest.weight_file);
const weights = fs.readFileSync(weightsPath);
const digest = crypto.createHash("sha256").update(weights).digest("hex");
const parameterCount = manifest.weights.reduce(
  (sum, item) => sum + item.value_count,
  0
);

if (
  weights.byteLength !== manifest.weight_bytes ||
  weights.byteLength !== descriptor.weight_bytes
) {
  throw new Error("TensorFlow.js weight byte count mismatch");
}

if (
  digest !== manifest.weight_sha256 ||
  digest !== descriptor.weight_sha256
) {
  throw new Error("TensorFlow.js weight checksum mismatch");
}

if (parameterCount !== 470977 || manifest.weights.length !== 22) {
  throw new Error(
    `Unexpected model manifest: ${manifest.weights.length} tensors, ${parameterCount} parameters`
  );
}

const html = fs.readFileSync(path.join(web, "index.html"), "utf8");
const app = fs.readFileSync(path.join(web, "app.js"), "utf8");

if (!html.includes("@tensorflow/tfjs@4.22.0")) {
  throw new Error("TensorFlow.js CDN reference missing from index.html");
}

if (!app.includes("buildCompactUnet")) {
  throw new Error("Compact U-Net browser architecture missing from app.js");
}

console.log(
  `Vercel static validation passed: ${manifest.weights.length} tensors, ` +
    `${parameterCount.toLocaleString()} parameters, ` +
    `${weights.byteLength.toLocaleString()} weight bytes.`
);
