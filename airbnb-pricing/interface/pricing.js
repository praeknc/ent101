/* Pricing engine for the final model (LightGBM, 141 features), exported by
   notebooks/boston_airbnb_final_pipeline.ipynb. Rebuilds the features from a listing typed into the
   form with the same rules as src/pricing_features.py and src/enriched_features.py, then runs the trees.
   Works in the browser (window.PricingEngine) and in Node (scripts/verify_js.js). */
(function (root) {
  "use strict";

  /* P = window.PRICING (data.js), MP = window.PRICING_MODELS (models.js) */
  function create(P, MP) {
    var C = P.final, F = MP.final.features, TREES = MP.final.trees;
    var CENTER = C.center, BANDS = C.host_bands, LM = C.landmarks, ZONES = C.zones;
    var LM_NAMES = Object.keys(LM);
    var KW = ["kw_parking", "kw_luxury", "kw_view", "kw_amenities"];
    var SCORES = ["rating", "accuracy", "cleanliness", "checkin", "communication", "location", "value"];
    var DPM = 365.25 / 12;

    // right-child index for each internal node of a pre-order tree (left child = i + 1)
    TREES.forEach(function (T) {
      var n = T.f.length, r = new Int32Array(n), stack = [];
      for (var i = 0; i < n; i++) {
        if (i > 0 && T.f[i - 1] === -1) r[stack.pop()] = i;
        if (T.f[i] !== -1) stack.push(i);
      }
      T.r = r;
    });

    function km(lat, lon) { return [(lat - CENTER[0]) * 111.0, (lon - CENTER[1]) * 82.0]; }
    function dist(a, b) { return Math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])); }
    var SUPPLY = {};
    Object.keys(C.supply).forEach(function (q) { SUPPLY[q] = C.supply[q].map(function (p) { return km(p[0], p[1]); }); });

    /* L: {neighbourhood, lat, lon, room_type, property_group, accommodates, bedrooms, beds, baths, baths_shared,
          amenities: {am_*: bool}, am_count, title_kw: {kw_*: bool}, minimum_nights, availability_365,
          license: "licensed"|"exempt"|"none", host_listings, superhost, identity_verified,
          number_of_reviews, reviews_per_month, number_of_reviews_ltm, scores: {rating..value},
          months_since_first_review, days_since_review, quarter: "2026Q2"|"2026Q3", tenure_q, quote_dow: 0-6 (Mon=0), desc_len} */
    function featureDict(L) {
      var room = C.room_merge[L.room_type] || L.room_type;
      var hood = C.hood_merge[L.neighbourhood] || L.neighbourhood;
      var mn = Math.min(365, L.minimum_nights > 0 ? L.minimum_nights : 1);
      var nrev = Math.max(0, Math.round(L.number_of_reviews || 0)), hasRev = nrev > 0;
      var rpm = hasRev ? Math.min(15, Math.max(0, L.reviews_per_month || 0)) : 0;
      var host = Math.max(1, L.host_listings || 1), band = 0;
      while (band < BANDS.length - 1 && host > BANDS[band + 1]) band++;
      var p = km(L.lat, L.lon), dkm = dist(p, [0, 0]);
      var d = {
        bedrooms: Math.min(8, Math.max(0, Math.round(L.bedrooms))), bedrooms_known: 1,
        private_room: room === "Private room" ? 1 : 0, hotel_room: room === "Hotel room" ? 1 : 0,
        log_min_nights: Math.log1p(mn), log_reviews: Math.log1p(nrev), reviews_per_month: rpm,
        number_of_reviews_ltm: Math.min(nrev, Math.max(0, Math.round(L.number_of_reviews_ltm || 0))),
        host_size_band: band, availability_365: L.availability_365,
        licensed: L.license === "licensed" ? 1 : 0, exempt: L.license === "exempt" ? 1 : 0,
        dist_km: dkm, latitude: L.lat, longitude: L.lon,
        q_sep: L.quarter === "2026Q3" ? 1 : 0, tenure_q: L.tenure_q || 0,
        days_since_review: hasRev ? Math.min(1095, Math.max(0, L.days_since_review)) : 1095,
        ring_core: dkm < 2 ? 1 : 0, ring_inner: dkm >= 2 && dkm <= 5 ? 1 : 0, ring_outer: dkm > 5 ? 1 : 0,
        accommodates: L.accommodates, bedrooms_d: L.bedrooms, beds: L.beds, baths: L.baths, baths_shared: L.baths_shared ? 1 : 0,
        am_count: Math.max(L.am_count, 0), has_review_scores: hasRev ? 1 : 0,
        superhost: L.superhost ? 1 : 0, identity_verified: L.identity_verified ? 1 : 0, desc_len: L.desc_len
      };
      KW.forEach(function (k) { d[k] = L.title_kw && L.title_kw[k] ? 1 : 0; });
      d["nb_" + hood] = 1;
      var zone = 0, best = Infinity;
      ZONES.forEach(function (c, z) { var dz = dist(p, c); if (dz < best) { best = dz; zone = z; } });
      d["zone_" + zone] = 1;
      C.property_groups.forEach(function (g) { d["prop_" + g.replace(/ /g, "_").replace("&", "and")] = g === L.property_group ? 1 : 0; });
      Object.keys(C.amenities).forEach(function (k) { d[k] = L.amenities && L.amenities[k] ? 1 : 0; });
      if (d.am_central_ac) d.am_any_ac = 1;
      SCORES.forEach(function (s) { d["rs_" + s] = hasRev ? L.scores[s] : NaN; });
      var nearest = Infinity, within2 = 0;
      LM_NAMES.forEach(function (n) {
        var dl = dist(p, km(LM[n][0], LM[n][1]));
        d["d_" + n] = dl; nearest = Math.min(nearest, dl); if (dl < 2) within2++;
      });
      d.d_nearest_landmark = nearest; d.n_landmarks_2km = within2;
      var s300 = 0, s1k = 0, S = SUPPLY[L.quarter] || [];
      for (var i = 0; i < S.length; i++) { var ds = dist(p, S[i]); if (ds < 0.3) s300++; if (ds < 1.0) s1k++; }
      d.supply_300m = s300; d.supply_1km = s1k;
      d.months_since_first_review = hasRev ? Math.round(L.months_since_first_review * DPM) / DPM : NaN;
      for (var w = 0; w < 7; w++) d["quote_dow_" + w] = w === L.quote_dow ? 1 : 0;
      return d;
    }
    function features(L) {
      var d = featureDict(L);
      return F.map(function (f) { return f in d ? d[f] : 0; });
    }

    // LightGBM numerical split: missing type 0 = none (NaN -> 0), 1 = zero, 2 = NaN
    function tree(T, x) {
      var i = 0;
      while (T.f[i] !== -1) {
        var v = x[T.f[i]], m = T.m[i], left;
        if (v === null || v === undefined) v = NaN;
        if (m === 2 && v !== v) left = T.d[i] === 1;
        else {
          if (v !== v) v = 0;
          if (m === 1 && Math.abs(v) <= 1e-35) left = T.d[i] === 1;
          else left = v <= T.x[i];
        }
        i = left ? i + 1 : T.r[i];
      }
      return T.x[i];
    }
    function predictLog(x) {
      var s = 0;
      for (var i = 0; i < TREES.length; i++) s += tree(TREES[i], x);
      return s;
    }
    function price(L) {
      var v = Math.exp(predictLog(features(L)));
      return { price: v, low: v * Math.exp(C.band_log[0]), high: v * Math.exp(C.band_log[1]) };
    }
    return { features: features, predictLog: predictLog, price: price, featureNames: F };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { create: create };
  else root.PricingEngine = { create: create };
})(this);
