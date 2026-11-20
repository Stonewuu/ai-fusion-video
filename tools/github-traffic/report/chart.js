const dayMs = 86400000;
const number = new Intl.NumberFormat("zh-CN");
const fullDate = new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" });
const unitNames = {
  count: { clones: "克隆次数", views: "浏览次数" },
  uniques: { clones: "独立克隆者", views: "独立访客" }
};
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const dateTime = day => Date.parse(day + "T00:00:00Z");
const isoDay = time => new Date(time).toISOString().slice(0, 10);
const dateLabel = day => fullDate.format(dateTime(day));
const formatValue = value => value == null ? "—" : number.format(value);

function segments(points) {
  const groups = [];
  let group = [];
  for (const point of points) {
    if (point === null) {
      if (group.length) {
        groups.push(group);
        group = [];
      }
    } else {
      group.push(point);
    }
  }
  if (group.length) {
    groups.push(group);
  }
  return groups;
}

function linePath(points) {
  const slopes = points.slice(1).map(([x, y], i) => (y - points[i][1]) / (x - points[i][0]));
  const tangents = points.map((_, i) => {
    if (i === 0 || i === points.length - 1) {
      return slopes[i === 0 ? 0 : i - 1];
    }
    const a = slopes[i - 1];
    const b = slopes[i];
    return a * b <= 0 ? 0 : 2 * a * b / (a + b);
  });
  let path = "M" + points[0].join(" ");
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    const width = (x1 - x0) / 3;
    path += "C" + [x0 + width, y0 + width * tangents[i - 1], x1 - width, y1 - width * tangents[i], x1, y1].join(" ");
  }
  return path;
}

function yScale(max) {
  const target = Math.max(1, max / 4);
  const base = 10 ** Math.floor(Math.log10(target));
  const step = [1, 2, 2.5, 5, 10].map(n => n * base).find(n => n >= target && Number.isInteger(n));
  const ceiling = Math.max(step, Math.ceil(max / step) * step);
  return { ceiling, step };
}

function svgNode(tag, attrs, content = "") {
  const attributes = Object.entries(attrs).map(([key, value]) => key + '="' + value + '"').join(" ");
  return "<" + tag + " " + attributes + ">" + content + "</" + tag + ">";
}

class TrafficChart {
  constructor(svg, tip, reading) {
    this.svg = svg;
    this.tip = tip;
    this.reading = reading;
    this.index = -1;
    this.resize = new ResizeObserver(() => {
      const bounds = svg.getBoundingClientRect();
      if (bounds.width !== this.width || bounds.height !== this.height) {
        this.draw(false);
      }
    });
    this.resize.observe(svg);
    const pointAt = event => {
      if (!this.rows.length) {
        return;
      }
      const offset = event.clientX - svg.getBoundingClientRect().left;
      const fraction = (offset - this.left) / (this.right - this.left);
      this.show(Math.round(Math.min(1, Math.max(0, fraction)) * (this.rows.length - 1)), false);
    };
    svg.addEventListener("pointermove", pointAt);
    svg.addEventListener("pointerdown", pointAt);
    svg.addEventListener("pointerleave", event => {
      if (event.pointerType === "mouse") {
        this.clear();
      }
    });
    svg.addEventListener("focus", () => {
      if (this.rows.length) {
        this.show(this.rows.length - 1, true);
      }
    });
    svg.addEventListener("blur", () => this.clear());
    svg.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        this.clear();
        return;
      }
      if (!this.rows.length) {
        return;
      }
      const positions = { ArrowLeft: Math.max(0, this.index - 1), ArrowRight: Math.min(this.rows.length - 1, this.index + 1), Home: 0, End: this.rows.length - 1 };
      if (Object.hasOwn(positions, event.key)) {
        event.preventDefault();
        this.show(positions[event.key], true);
      }
    });
  }

  set(rows, unit, series) {
    this.rows = rows;
    this.unit = unit;
    this.series = series;
    this.draw(true);
  }

  x(index) {
    if (this.rows.length === 1) {
      return (this.left + this.right) / 2;
    }
    return this.left + index * (this.right - this.left) / (this.rows.length - 1);
  }

  y(value) {
    return this.bottom - (this.bottom - this.top) * value / this.ceiling;
  }

  draw(animate) {
    if (!this.rows) {
      return;
    }
    this.clear();
    const { width, height } = this.svg.getBoundingClientRect();
    if (width < 1) {
      return;
    }
    this.width = width;
    this.height = height;
    this.top = 17;
    this.bottom = height - 30;
    this.right = width - 12;
    let max = 0;
    let hasValues = false;
    for (const row of this.rows) {
      for (const key of this.series) {
        const value = row[key]?.[this.unit];
        if (value != null) {
          max = Math.max(max, value);
          hasValues = true;
        }
      }
    }
    const { ceiling, step } = yScale(max);
    this.ceiling = ceiling;
    this.left = Math.max(30, number.format(ceiling).length * 7 + 12);
    this.svg.setAttribute("viewBox", [0, 0, width, height].join(" "));
    document.querySelector("#empty-chart").hidden = hasValues;
    if (!hasValues) {
      this.svg.replaceChildren();
      return;
    }
    const axis = [];
    for (let value = 0; value <= ceiling; value += step) {
      const y = this.y(value);
      axis.push(svgNode("line", { x1: this.left, x2: this.right, y1: y, y2: y, class: value ? "grid" : "baseline" }));
      axis.push(svgNode("text", { x: this.left - 8, y: y + 4, class: "axis", "text-anchor": "end" }, number.format(value)));
    }
    const size = Math.min(this.rows.length, Math.max(2, Math.floor((this.right - this.left) / 78) + 1));
    for (let tick = 0; tick < size; tick++) {
      const index = size === 1 ? 0 : Math.round(tick * (this.rows.length - 1) / (size - 1));
      const day = this.rows[index].date;
      const label = this.rows.length > 365 ? day.slice(0, 7) : day.slice(5).replace("-", "/");
      const anchor = size === 1 ? "middle" : tick === 0 ? "start" : tick === size - 1 ? "end" : "middle";
      axis.push(svgNode("text", { x: this.x(index), y: height - 7, class: "axis", "text-anchor": anchor }, label));
    }
    const shapes = [];
    for (const key of this.series) {
      const points = this.rows.map((row, i) => row[key] === null ? null : [this.x(i), this.y(row[key][this.unit])]);
      for (const segment of segments(points)) {
        if (segment.length === 1) {
          shapes.push(svgNode("circle", { cx: segment[0][0], cy: segment[0][1], r: 3.2, fill: "var(--" + (key === "clones" ? "violet" : "green") + ")" }));
        } else {
          const path = linePath(segment);
          if (key === this.series[0]) {
            const area = path + "L" + segment.at(-1)[0] + " " + this.bottom + "L" + segment[0][0] + " " + this.bottom + "Z";
            shapes.push(svgNode("path", { d: area, fill: "url(#area-" + key + ")" }));
          }
          shapes.push(svgNode("path", { d: path, class: "plot-line " + key + "-line" }));
        }
      }
    }
    const gradients = ["clones", "views"].map((key, i) => {
      const color = "var(--" + (i ? "green" : "violet") + ")";
      return svgNode("linearGradient", { id: "area-" + key, x1: 0, x2: 0, y1: 0, y2: 1 },
        svgNode("stop", { offset: "0%", "stop-color": color, "stop-opacity": ".17" }) +
        svgNode("stop", { offset: "100%", "stop-color": color, "stop-opacity": ".015" }));
    }).join("");
    this.svg.innerHTML = "<defs>" + gradients +
      '<clipPath id="reveal"><rect id="reveal-rect" x="0" y="0" width="' + width + '" height="' + height + '"/></clipPath></defs>' +
      axis.join("") + '<g clip-path="url(#reveal)">' + shapes.join("") + '</g><g id="markers" aria-hidden="true"></g>';
    if (animate && !reducedMotion.matches) {
      this.svg.querySelector("#reveal-rect").animate([{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 340, easing: "cubic-bezier(.2,.65,.35,1)" });
    }
  }

  show(index, announce) {
    const markers = this.svg.querySelector("#markers");
    if (!markers) {
      return;
    }
    this.index = index;
    const row = this.rows[index];
    const x = this.x(index);
    const entries = this.series.map(key => ({ key, label: unitNames[this.unit][key], value: row[key]?.[this.unit] }));
    this.tip.innerHTML = "<strong>" + dateLabel(row.date) + "</strong>" + entries.map(({ key, label, value }) =>
      '<div class="tip-row"><span><i class="' + key + '"></i>' + label + "</span><b>" + formatValue(value) + "</b></div>"
    ).join("");
    markers.innerHTML = svgNode("line", { x1: x, x2: x, y1: this.top, y2: this.bottom, class: "guide" }) +
      entries.filter(entry => entry.value != null).map(({ key, value }) =>
        svgNode("circle", { cx: x, cy: this.y(value), r: 4.5, class: "dot", fill: "var(--" + (key === "clones" ? "violet" : "green") + ")" })
      ).join("");
    this.tip.hidden = false;
    const w = this.tip.offsetWidth;
    this.tip.style.left = Math.max(0, Math.min(this.width - w, x + w + 14 < this.width ? x + 12 : x - w - 12)) + "px";
    this.tip.style.top = this.top + 3 + "px";
    if (announce) {
      this.reading.textContent = dateLabel(row.date) + "，" + entries.map(v => v.label + " " + formatValue(v.value)).join("，");
    }
  }

  clear() {
    this.index = -1;
    this.tip.hidden = true;
    this.svg.querySelector("#markers")?.replaceChildren();
  }
}
