NormflowVis = function(_parent, _data, _flow, _parentVis) {
  this.parentElement = _parent;
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

  vis.width = 300;
  vis.height = 300;

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