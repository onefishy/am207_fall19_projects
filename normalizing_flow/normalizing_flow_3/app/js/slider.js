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
