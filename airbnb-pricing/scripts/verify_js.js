// Checks that interface/pricing.js reproduces the final model exactly:
//  1. stored feature rows -> same prediction as Python (tree export and missing-value handling)
//  2. raw listing inputs  -> same 141 features and price as the Python pipeline (scripts/final_cases.json)
// Run: node scripts/verify_js.js
const path = require("path");
const fs = require("fs");
global.window = {};
require(path.join(__dirname, "../interface/data.js"));
require(path.join(__dirname, "../interface/models.js"));
const P = window.PRICING, MP = window.PRICING_MODELS;
const E = require(path.join(__dirname, "../interface/pricing.js")).create(P, MP);
let ok = true;

let maxErr = 0;
MP.final_reference.X.forEach((x, i) => { maxErr = Math.max(maxErr, Math.abs(E.predictLog(x) - MP.final_reference.pred[i])); });
console.log(`Tree export        ${MP.final_reference.X.length} rows, max |log error| = ${maxErr.toExponential(2)}`);
if (maxErr > 1e-4) ok = false;

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, "final_cases.json"), "utf8"));
let bad = 0, maxP = 0;
cases.forEach((c, i) => {
  const x = E.features(c.input);
  const diff = c.x.map((v, j) => {
    const a = v === null ? NaN : v, b = x[j];
    const same = (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a));
    return same ? null : `${E.featureNames[j]}: py ${v} js ${b}`;
  }).filter(Boolean);
  if (diff.length) { bad++; if (bad <= 40) console.error(`case ${i}: ${diff.slice(0, 8).join("; ")}`); }
  maxP = Math.max(maxP, Math.abs(Math.exp(E.predictLog(x)) / Math.exp(c.pred) - 1));
});
console.log(`Feature parity     ${cases.length - bad}/${cases.length} raw listings build identical features; max price difference ${(maxP * 100).toFixed(4)}%`);
if (bad || maxP > 1e-4) ok = false;
if (!ok) { console.error("FAIL: browser engine differs from Python"); process.exit(1); }
console.log("OK: browser predictions match the notebook");
