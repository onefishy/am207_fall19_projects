// *********** main.js ************ //
define('flowVis', ['d3'], function (d3) {
  function draw(container) {
    var svg = d3.select(container).append("svg");

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

  }
  return draw;
});

// *********** flows.js ************ //
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
  this.u0 = _u[0];
  this.u1 = _u[1];
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
  }
};

// *********** normflow-vis.js ************ //
NormflowVis = function(_offset, _width, _data, _flow, _parentVis) {
  this.offset = _offset;
  this.width = _width;
  this.height = this.width;
  this.data = _data;
  this.transformedData = _data;
  this.displayData = _data;
  this.flow = _flow;
  this.childVis = null;
  if (_parentVis !== undefined) {
    _parentVis.childVis = this;
  }

  this.initVis();
};

NormflowVis.prototype.initVis = function() {
  var vis = this;

  vis.margin = { top: 40, right: 0, bottom: 60, left: 60 };

  // vis.width = 300;
  // vis.height = 300;

  vis.extent = 10;

  // SVG drawing area
  vis.svg = d3.select("#" + vis.parentElement).append("svg")
      .attr("width", vis.width + vis.margin.left + vis.margin.right)
      .attr("height", vis.height + vis.margin.top + vis.margin.bottom)
      .append("g")
      .attr("transform", "translate(" + vis.margin.left + "," + vis.margin.top + ")");

  vis.contours = vis.svg.append('g');

  vis.colorScale = d3.scaleSequential()
      .interpolator(d3.interpolateYlGnBu);

  vis.density = d3.contourDensity()
      .x(function(d) { return vis.x(d[0]); })
      .y(function(d) { return vis.y(d[1]); })
      .size([vis.width, vis.height])
      .bandwidth(45);

  vis.x = d3.scaleLinear()
      .domain([-vis.extent, vis.extent]) // Set this to dynamically update with data
      .range([0, vis.width]);

  vis.svg.append("g")
      .attr('class', 'axis x')
      .attr("transform", "translate(0," + vis.height + ")")
      .call(d3.axisBottom(vis.x));

  // Add Y axis
  vis.y = d3.scaleLinear()
      .domain([-vis.extent, vis.extent])
      .range([vis.height, 0]);
  vis.svg.append("g")
      .attr('class', 'axis y')
      .call(d3.axisLeft(vis.y));

  // Adding sliders
  vis.sliderW0 = new Slider('w0', vis.parentElement, vis.width, -5, 5, val => vis.updateFlow({'w0': val}));
  vis.sliderW1 = new Slider('w1', vis.parentElement, vis.width, -5, 5, val => vis.updateFlow({'w1': val}));
  vis.sliderU0 = new Slider('u0', vis.parentElement, vis.width, -5, 5, val => vis.updateFlow({'u0': val}));
  vis.sliderU1 = new Slider('u1', vis.parentElement, vis.width, -5, 5, val => vis.updateFlow({'u1': val}));
  vis.sliderB = new Slider('b', vis.parentElement, vis.width, -5, 5, val => vis.updateFlow({'b': val}));

  // Adding label of dot product
  vis.dotProdLab = vis.svg.append('text')
      .attr('class', 'label')
      .attr('x', 20)
      .attr('y', 0);

  // Adding label for flow parameters
  vis.flowLabs = vis.svg.append('g')
      .attr('transform', 'translate(0,' + (vis.height + vis.margin.bottom - 20) + ')')
      .attr('class', 'flow-labs');
  vis.wLab = vis.flowLabs.append('text')
      .attr('transform', 'translate(0,0)');
  vis.uLab = vis.flowLabs.append('text')
      .attr('transform', 'translate(' + vis.width / 3 + ',0)');
  vis.bLab = vis.flowLabs.append('text')
      .attr('transform', 'translate(' + vis.width * 2 / 3 + ',0)');

  vis.wrangleData();
};

NormflowVis.prototype.wrangleData = function() {
  var vis = this;

  vis.transformedData = vis.flow.transform(vis.data);
  vis.displayData = vis.density(vis.transformedData);

  vis.updateChildData(vis.transformedData);
  vis.updateVis();
};

NormflowVis.prototype.updateVis = function() {
  var vis = this;

  vis.colorScale.domain(d3.extent(vis.displayData, d => d.value));

  var contours = vis.contours
      .selectAll("path")
      .data(vis.displayData, (d, i) => i);

  contours.enter().append("path")
      .merge(contours)
      .attr("d", d3.geoPath())
      .attr("fill", d => vis.colorScale(d.value));

  contours.exit().remove();

  var dotProd = vis.dotProd();
  vis.dotProdLab.text('w^Tu = ' + format1d(dotProd))
      .attr('fill', () => dotProd >= -1 ? 'black' : 'red');
  vis.wLab.text('w = [' + format1d(vis.flow.w0) + ', ' + format1d(vis.flow.w1) + ']');
  vis.uLab.text('u = [' + format1d(vis.flow.u0) + ', ' + format1d(vis.flow.u1) + ']');
  vis.bLab.text('b = ' + format1d(vis.flow.b));

};
NormflowVis.prototype.updateFlow = function(newParams) {
  var vis = this;
  vis.flow.updateParams(newParams);
  vis.wrangleData();
};
NormflowVis.prototype.updateData = function(newData) {
  var vis = this;

  vis.data = newData;
  vis.wrangleData();
};
NormflowVis.prototype.dotProd = function() {
  var vis = this;

  return vis.flow.w0 * vis.flow.u0 + vis.flow.w1 * vis.flow.u1;
};
NormflowVis.prototype.updateChildData = function(newData) {
  var vis = this;
  if (vis.childVis) {
    vis.childVis.updateData(newData);
  }
};

// *********** slider.js ************ //
Slider = function(_name, _parent, _width, _min, _max, _callback) {
  this.name = _name;
  this.parentElement = _parent;
  this.min = _min;
  this.max = _max;
  this.callback = _callback;
  this.current = 0;
  this.width = _width;

  this.initVis();
};

Slider.prototype.initVis = function() {
  var vis = this;

  vis.margin = { top: 15, right: 40, bottom: 5, left: 40 };

  vis.height = 40;

  // SVG drawing area
  vis.svg = d3.select("#" + vis.parentElement).append("svg")
      .attr("width", vis.width + vis.margin.left + vis.margin.right)
      .attr("height", vis.height + vis.margin.top + vis.margin.bottom)
      .append("g")
      .attr("transform", "translate(" + vis.margin.left + "," + vis.margin.top + ")");

  vis.slider = d3.sliderHorizontal()
      .min(vis.min)
      .max(vis.max)
      .step(0.1)
      .width(vis.width)
      .height(vis.height)
      .displayValue(true)
      .on('onchange', function(val) {
        vis.current = val;
        vis.callback(val);
      });

  vis.svg.call(vis.slider);

  vis.svg.append('text')
      .text(vis.name)
      .attr('x', vis.width + 15)
      .attr('y', 0);
};
