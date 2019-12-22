/**
 * Flow functions
 *
 * - Identity
 * - Planar
 * - Radial
 */
IdentityFlow = function() {};

IdentityFlow.prototype.transform = function(data) {
  return data;
};

PlanarFlow = function(_w, _u, _b) {
  this.w0 = _w[0];
  this.w1 = _w[1];
  var uhat = getUhat(_w, _u);
  this.u0 = uhat[0];
  this.u1 = uhat[1];
  this.b = _b;
};
PlanarFlow.prototype.transform = function(z) {
  var flow = this;

  flow.w = [flow.w0, flow.w1];
  flow.u = [flow.u0, flow.u1];

  return z.map(function(d) {
    return math.add(d, math.multiply(flow.u, math.tanh(math.add(math.dot(flow.w, d), flow.b))));
  })
};
PlanarFlow.prototype.updateParams = function(newParams) {
  for (var param in newParams) {
    if (newParams.hasOwnProperty(param)) {
      this[param] = newParams[param];
    }
    var uhat = getUhat([this.w0, this.w1], [this.u0, this.u1]);
    this.u0 = uhat[0];
    this.u1 = uhat[1];
  }
};
function getUhat(w, u) {
  if ((math.norm(u) === 0) || (math.norm(w) === 0)) {
    return u;
  } else {
    var dotProd = math.dot(w, u);
    var mVal = m(dotProd);
    var wNorm2 = math.dot(w, w);
    var wUnit = math.divide(w, wNorm2);
    var res = math.add(u, math.multiply(mVal - dotProd, wUnit));
    return res;
  }
}
function m(x) {
  return -1 + math.log(1 + math.exp(x));
}