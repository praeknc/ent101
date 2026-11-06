// Checks that interface/pricing.js reproduces the Python predictions stored in data.js.
// Run: node scripts/verify_js.js
const path = require("path");
global.window = {};
require(path.join(__dirname, "../interface/data.js"));
require(path.join(__dirname, "../interface/models.js"));
const P = window.PRICING, MP = window.PRICING_MODELS;
const engine = require(path.join(__dirname, "../interface/pricing.js")).create(P, MP);

let worst = 0;
for (const name of Object.keys(engine.predictLog)) {
  let maxErr = 0;
  MP.snapshot_reference.X.forEach((x, i) => {
    const err = Math.abs(engine.predictLog[name](x) - MP.snapshot_reference.pred[name][i]);
    maxErr = Math.max(maxErr, err);
  });
  worst = Math.max(worst, maxErr);
  console.log(`${name.padEnd(18)} max |log error| = ${maxErr.toExponential(2)}  (≈ ${((Math.exp(maxErr) - 1) * 100).toFixed(3)}% of price)`);
}
if (MP.quarterly) {
  let maxErr = 0;
  MP.quarterly_reference.X.forEach((x, i) => { maxErr = Math.max(maxErr, Math.abs(engine.quarterlyLog(x) - MP.quarterly_reference.pred[i])); });
  worst = Math.max(worst, maxErr);
  console.log(`${"Quarterly RF".padEnd(18)} max |log error| = ${maxErr.toExponential(2)}  (≈ ${((Math.exp(maxErr) - 1) * 100).toFixed(3)}% of price)`);
}
// Feature parity: raw listing inputs -> same feature vector in Python and JavaScript
const fs = require("fs");
const casesPath = path.join(__dirname, "feature_cases.json");
if (fs.existsSync(casesPath)) {
  const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));
  let bad = 0;
  cases.forEach((c, i) => {
    if (c.features.join() !== P.features.join()) { bad++; if (bad < 3) console.error("feature list differs"); return; }
    const x = engine.features(c.input);
    const diff = c.x.map((v, j) => Math.abs(v - x[j]) > 1e-9 ? c.features[j] + ": py " + v + " js " + x[j] : null).filter(Boolean);
    if (diff.length) { bad++; if (bad < 4) console.error("case " + i + ": " + diff.join("; ")); }
  });
  console.log(`Feature parity       ${cases.length - bad}/${cases.length} raw listings build identical features`);
  if (bad) worst = Infinity;
}
if (worst > 0.01) { console.error("FAIL: JS port differs from Python"); process.exit(1); }
console.log("OK: browser predictions match the notebook");
