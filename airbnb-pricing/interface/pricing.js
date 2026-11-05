/* Pricing engine: a JavaScript port of build_features() and the five trained models
   from notebooks/boston_airbnb_pricing.ipynb. Works in the browser (window.PricingEngine)
   and in Node (module.exports) so scripts/verify_js.js can check it against Python. */
(function (root) {
  "use strict";

  function create(P) {
    var F = P.features;
    var idx = {};
    F.forEach(function (f, i) { idx[f] = i; });
    var CENTER = P.meta.center;
    var KW = Object.keys(P.keywords);

    /* listing: {neighbourhood, room_type, bedrooms, minimum_nights, number_of_reviews,
       reviews_per_month, number_of_reviews_ltm, host_listings, availability_365,
       license: "licensed"|"exempt"|"none", lat, lon, kw: {kw_parking: bool, ...}} */
    function features(L) {
      var x = new Array(F.length).fill(0);
      var mn = L.minimum_nights > 0 ? L.minimum_nights : 1;
      x[idx.bedrooms] = L.bedrooms;
      x[idx.bedrooms_known] = 1;
      x[idx.private_room] = L.room_type === "Private room" ? 1 : 0;
      x[idx.hotel_room] = L.room_type === "Hotel room" ? 1 : 0;
      x[idx.shared_room] = L.room_type === "Shared room" ? 1 : 0;
      x[idx.log_min_nights] = Math.log1p(mn);
      x[idx.monthly] = mn >= 28 ? 1 : 0;
      x[idx.log_reviews] = Math.log1p(L.number_of_reviews);
      x[idx.reviews_per_month] = L.reviews_per_month;
      x[idx.number_of_reviews_ltm] = L.number_of_reviews_ltm;
      x[idx.log_host_listings] = Math.log1p(L.host_listings);
      x[idx.availability_365] = L.availability_365;
      x[idx.licensed] = L.license === "licensed" ? 1 : 0;
      x[idx.exempt] = L.license === "exempt" ? 1 : 0;
      x[idx.dist_km] = Math.sqrt(Math.pow((L.lat - CENTER[0]) * 111.0, 2) + Math.pow((L.lon - CENTER[1]) * 82.0, 2));
      x[idx.latitude] = L.lat;
      x[idx.longitude] = L.lon;
      KW.forEach(function (k) { x[idx[k]] = L.kw && L.kw[k] ? 1 : 0; });
      var nb = idx["nb_" + L.neighbourhood];
      if (nb !== undefined) x[nb] = 1;
      return x;
    }

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
      var room = x[idx.private_room] ? "private" : x[idx.hotel_room] ? "hotel" : x[idx.shared_room] ? "shared" : "entire";
      var key = hood + "|" + room;
      if (key in M.cell) return M.cell[key];
      return room in M.room ? M.room[room] : M.all;
    }

    var M = P.models;
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

    return { features: features, predictLog: predictLog, priceAll: priceAll, lassoDrivers: lassoDrivers, featureNames: F };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { create: create };
  else root.PricingEngine = { create: create };
})(this);
