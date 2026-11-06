/* Pricing engine: a JavaScript port of build_features() and the five trained models
   from notebooks/boston_airbnb_pricing.ipynb. Works in the browser (window.PricingEngine)
   and in Node (module.exports) so scripts/verify_js.js can check it against Python. */
(function (root) {
  "use strict";

  /* P = window.PRICING (data.js), MP = window.PRICING_MODELS (models.js) */
  function create(P, MP) {
    var F = P.features;
    var idx = {};
    F.forEach(function (f, i) { idx[f] = i; });
    var CENTER = P.meta.center;
    var KW = Object.keys(P.keywords);

    var HOST_BANDS = P.host_bands;               // upper bounds: 1 | 3 | 10 | 50 -> bands 0..4
    var ROOM_MERGE = P.room_merge || {}, HOOD_MERGE = P.hood_merge || {};

    /* Same rules as src/pricing_features.py (pre-check merges, caps and build_features).
       listing: {neighbourhood, room_type, bedrooms, minimum_nights, number_of_reviews,
       reviews_per_month, number_of_reviews_ltm, host_listings, availability_365,
       license: "licensed"|"exempt"|"none", lat, lon, kw: {kw_parking: bool, ...}} */
    function featureDict(L) {
      var room = ROOM_MERGE[L.room_type] || L.room_type;
      var hood = HOOD_MERGE[L.neighbourhood] || L.neighbourhood;
      var mn = Math.min(365, L.minimum_nights > 0 ? L.minimum_nights : 1);
      var reviews = Math.max(0, L.number_of_reviews || 0);
      var rpm = reviews === 0 ? 0 : Math.min(15, Math.max(0, L.reviews_per_month || 0));
      var host = Math.max(1, L.host_listings || 1), band = 0;
      while (band < HOST_BANDS.length - 1 && host > HOST_BANDS[band + 1]) band++;
      var d = {
        bedrooms: L.bedrooms, bedrooms_known: 1,
        private_room: room === "Private room" ? 1 : 0, hotel_room: room === "Hotel room" ? 1 : 0,
        shared_room: room === "Shared room" ? 1 : 0,
        log_min_nights: Math.log1p(mn), monthly: mn >= 28 ? 1 : 0,
        log_reviews: Math.log1p(reviews), reviews_per_month: rpm,
        number_of_reviews_ltm: Math.min(reviews, Math.max(0, L.number_of_reviews_ltm || 0)),
        host_size_band: band, log_host_listings: Math.log1p(host),
        availability_365: L.availability_365,
        licensed: L.license === "licensed" ? 1 : 0, exempt: L.license === "exempt" ? 1 : 0,
        dist_km: Math.sqrt(Math.pow((L.lat - CENTER[0]) * 111.0, 2) + Math.pow((L.lon - CENTER[1]) * 82.0, 2)),
        latitude: L.lat, longitude: L.lon
      };
      KW.forEach(function (k) { d[k] = L.kw && L.kw[k] ? 1 : 0; });
      d["nb_" + hood] = 1;
      return d;
    }
    function features(L) {
      var d = featureDict(L);
      return F.map(function (f) { return d[f] || 0; });
    }

    // Trees are stored in pre-order (left child of node i is i+1); rebuild right-child indexes once.
    function prepare(T) {
      if (T.r) return T;
      var n = T.f.length, r = new Int32Array(n), stack = [];
      for (var i = 0; i < n; i++) {
        if (i > 0 && T.f[i - 1] === -1) r[stack.pop()] = i;
        if (T.f[i] !== -1) stack.push(i);
      }
      T.r = r;
      return T;
    }
    ["Random Forest", "Gradient Boosting"].forEach(function (n) { MP.snapshot[n].trees.forEach(prepare); });
    if (MP.quarterly) MP.quarterly.trees.forEach(prepare);

    // sklearn compares float32 features against float64 thresholds
    var f32 = Math.fround;
    function tree(T, x) {
      var i = 0;
      while (T.f[i] !== -1) i = f32(x[T.f[i]]) <= T.x[i] ? i + 1 : T.r[i];
      return T.x[i];
    }

    function linear(M, x) {
      var s = M.intercept;
      for (var i = 0; i < x.length; i++) s += M.coef[i] * (x[i] - M.mean[i]) / M.scale[i];
      return s;
    }

    function baseline(M, x) {
      var hood = null;
      F.forEach(function (f, i) { if (f.indexOf("nb_") === 0 && x[i] === 1) hood = f; });
      var room = x[idx.private_room] ? "private" : x[idx.hotel_room] ? "hotel" : "entire";
      var key = hood + "|" + room;
      if (key in M.cell) return M.cell[key];
      return room in M.room ? M.room[room] : M.all;
    }

    var M = MP.snapshot;
    // Each returns log(price)
    var predictLog = {
      "Baseline": function (x) { return baseline(M["Baseline"], x); },
      "Linear Regression": function (x) { return linear(M["Linear Regression"], x); },
      "Lasso": function (x) { return linear(M["Lasso"], x); },
      "Random Forest": function (x) {
        var s = 0, T = M["Random Forest"].trees;
        for (var i = 0; i < T.length; i++) s += tree(T[i], x);
        return s / T.length;
      },
      "Gradient Boosting": function (x) {
        var s = M["Gradient Boosting"].init, T = M["Gradient Boosting"].trees;
        for (var i = 0; i < T.length; i++) s += tree(T[i], x);
        return s;
      }
    };

    // Lasso explanation: contribution of each feature (in log points) vs the average listing
    function lassoDrivers(x) {
      var L = M["Lasso"], out = [];
      for (var i = 0; i < x.length; i++) {
        var c = L.coef[i] * (x[i] - L.mean[i]) / L.scale[i];
        if (c !== 0) out.push({ feature: F[i], logpts: c });
      }
      return out;
    }

    function priceAll(listing) {
      var x = features(listing), out = {};
      Object.keys(predictLog).forEach(function (n) { out[n] = Math.exp(predictLog[n](x)); });
      var best = P.meta.best;
      out.low = out[best] * Math.exp(P.meta.band_log[0]);
      out.high = out[best] * Math.exp(P.meta.band_log[1]);
      out.best = best;
      return out;
    }

    /* Phase 2: pooled quarterly model (snapshot features + time + zone).
       Extra listing fields: quarter ("2026Q2" | "2026Q3"), tenure_q (0-3), days_since_review. */
    var Q = P.quarterly && MP.quarterly ? P.quarterly : null;
    var QT = Q ? MP.quarterly.trees : null;
    function quarterlyFeatures(L) {
      var x = features(L);
      var dist = x[idx.dist_km];
      var ky = (L.lat - CENTER[0]) * 111.0, kx = (L.lon - CENTER[1]) * 82.0, zone = 0, bestD = Infinity;
      Q.zones.centers_km.forEach(function (c, z) { var d = (c[0] - ky) * (c[0] - ky) + (c[1] - kx) * (c[1] - kx); if (d < bestD) { bestD = d; zone = z; } });
      var extra = {
        q_sep: L.quarter === "2026Q3" ? 1 : 0, tenure_q: L.tenure_q || 0,
        days_since_review: Math.min(1095, Math.max(0, L.days_since_review == null ? 1095 : L.days_since_review)),
        ring_core: dist < 2 ? 1 : 0, ring_inner: dist >= 2 && dist <= 5 ? 1 : 0, ring_outer: dist > 5 ? 1 : 0
      };
      for (var z = 0; z < Q.n_zones; z++) extra["zone_" + z] = z === zone ? 1 : 0;
      return Q.features.map(function (f) { return f in extra ? extra[f] : x[idx[f]]; });
    }
    function quarterlyLog(xq) {
      var s = 0, T = QT;
      for (var i = 0; i < T.length; i++) s += tree(T[i], xq);
      return s / T.length;
    }
    function priceQuarter(L) {
      var xq = quarterlyFeatures(L), v = Math.exp(quarterlyLog(xq));
      return { price: v, low: v * Math.exp(Q.band_log[0]), high: v * Math.exp(Q.band_log[1]), zone: xq.slice(-Q.n_zones).indexOf(1) };
    }

    return { features: features, predictLog: predictLog, priceAll: priceAll, lassoDrivers: lassoDrivers, featureNames: F,
             quarterlyFeatures: Q ? quarterlyFeatures : null, quarterlyLog: Q ? quarterlyLog : null, priceQuarter: Q ? priceQuarter : null };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { create: create };
  else root.PricingEngine = { create: create };
})(this);
