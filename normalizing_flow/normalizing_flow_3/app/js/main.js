
// Colors
var yellow = d3.interpolateYlGn(0), // "rgb(255, 255, 229)"
    yellowGreen = d3.interpolateYlGn(0.5), // "rgb(120, 197, 120)"
    green = d3.interpolateYlGn(1); // "rgb(0, 69, 41)"

// Formater
var format1d = d3.format(".1f");

function generateData(N, d, mu, sigma) {
  if (isNaN(mu)) {
    mu = 0;
  }
  if (isNaN(sigma)) {
    sigma = 1;

  }
  var normSamples = d3.randomNormal(mu, sigma);
  var data = [];
  d3.range(0, N).forEach(function() {
    obs = [];
    d3.range(0, d).forEach(function() {
      obs.push(normSamples(2));
    });
    data.push(obs);
  });
  return data;
}


var data = generateData(1000, 2, 0, 1);

var w = [0, 0],
    u = [0, 0],
    b = 0;
var flow0 = new PlanarFlow(w, u, b);
var flow1 = new PlanarFlow(w, u, b);
var flow2 = new PlanarFlow(w, u, b);

var flowVis0 = new NormflowVis('vis-1', data, flow0);
var flowVis1 = new NormflowVis('vis-2', flowVis0.transformedData, flow1, flowVis0);
var flowVis2 = new NormflowVis('vis-3', flowVis1.transformedData, flow2, flowVis1);
