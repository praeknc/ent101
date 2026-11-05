// Checks that interface/pricing.js reproduces the Python predictions stored in data.js.
// Run: node scripts/verify_js.js
const path = require("path");
global.window = {};
require(path.join(__dirname, "../interface/data.js"));
const P = window.PRICING;
const engine = require(path.join(__dirname, "../interface/pricing.js")).create(P);

let worst = 0;
for (const name of Object.keys(engine.predictLog)) {
  let maxErr = 0;
  P.reference.X.forEach((x, i) => {
    const err = Math.abs(engine.predictLog[name](x) - P.reference.pred[name][i]);
    maxErr = Math.max(maxErr, err);
  });
  worst = Math.max(worst, maxErr);
  console.log(`${name.padEnd(18)} max |log error| = ${maxErr.toExponential(2)}  (≈ ${((Math.exp(maxErr) - 1) * 100).toFixed(3)}% of price)`);
}
if (worst > 0.01) { console.error("FAIL: JS port differs from Python"); process.exit(1); }
console.log("OK: browser predictions match the notebook");
