const $ = selector => document.querySelector(selector);
const history = JSON.parse($("#traffic-data").textContent);
const snapshot = history.snapshots.at(-1);
const byDay = new Map(history.days.map(row => [row.date, row]));
const keys = ["clones", "views"];
const dimensions = [["clones", "count"], ["clones", "uniques"], ["views", "count"], ["views", "uniques"]];
const earliest = history.days.at(0)?.date;
const latest = history.days.at(-1)?.date;
const perPage = 7;
const controls = { range: "14", unit: "count", active: new Set(keys), page: 0, customStart: earliest, customEnd: latest };
const chart = new TrafficChart($("#plot"), $("#tooltip"), $("#chart-reading"));
let periodRows = [];
let recordedRows = [];

function eachDay(start, end) {
  const rows = [];
  for (let time = dateTime(start); time <= dateTime(end); time += dayMs) {
    const date = isoDay(time);
    rows.push(byDay.get(date) ?? { date, clones: null, views: null });
  }
  return rows;
}

function shortRange(start, end) {
  if (start === end) {
    return dateLabel(start);
  }
  let last = dateLabel(end);
  if (start.slice(0, 7) === end.slice(0, 7)) {
    last = Number(end.slice(8)) + "日";
  } else if (start.slice(0, 4) === end.slice(0, 4)) {
    last = last.slice(last.indexOf("年") + 1);
  }
  return dateLabel(start) + " – " + last;
}

function sparkline(rows, key, unit) {
  const values = rows.map(row => row[key]?.[unit] ?? null);
  const observed = values.filter(v => v !== null);
  if (!observed.length) {
    return "";
  }
  const min = Math.min(...observed);
  const max = Math.max(...observed);
  const points = values.map((v, i) => {
    if (v === null) {
      return null;
    }
    return [values.length === 1 ? 34 : 2 + 66 * i / (values.length - 1), max === min ? 13 : 23 - (v - min) / (max - min) * 20];
  });
  const color = key === "clones" ? "#bbadff" : "#7ddfcb";
  const paths = segments(points).map(group => {
    if (group.length === 1) {
      return svgNode("circle", { cx: group[0][0], cy: group[0][1], r: 2, fill: color });
    }
    return svgNode("path", { d: linePath(group), fill: "none", stroke: color, "stroke-width": 1.8, "stroke-linecap": "round" });
  });
  return '<svg class="spark" viewBox="0 0 70 26" aria-hidden="true">' + paths.join("") + "</svg>";
}

function overview() {
  const [owner, repo] = history.repository.split("/");
  $("#repo-owner").textContent = owner;
  $("#repo-name").textContent = repo;
  $("#repo-link").href = "https://github.com/" + history.repository;
  $("#archive-link").href = $("#repo-link").href + "/tree/traffic-history";
  document.title = "仓库流量 · " + repo;
  const dates = keys.flatMap(key => [snapshot[key].first, snapshot[key].last]).filter(Boolean).sort();
  $("#summary-range").textContent = dates.length ? shortRange(dates[0], dates.at(-1)) + " · UTC" : "采集于 " + dateLabel(snapshot.observedOn) + " · UTC";
  $("#metrics").innerHTML = dimensions.map(([key, unit]) => {
    const data = snapshot[key];
    const rows = data.first === null ? [] : eachDay(data.first, data.last);
    return '<div class="kpi"><div class="kpi-label">' + unitNames[unit][key] +
      '</div><div class="kpi-value-row"><strong class="kpi-value">' + formatValue(data[unit]) +
      "</strong>" + sparkline(rows, key, unit) + "</div></div>";
  }).join("");
  $("#updated").textContent = "采集于 " + dateLabel(snapshot.observedOn) + " UTC";
}

function renderLegend() {
  const container = $("#legend");
  const isCount = controls.unit === "count";
  $("#legend-basis").textContent = isCount ? "区间合计" : "单日最高";
  for (const key of keys) {
    let button = container.querySelector('[data-series="' + key + '"]');
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.className = "series-key";
      button.dataset.series = key;
      container.append(button);
    }
    const active = controls.active.has(key);
    const label = unitNames[controls.unit][key];
    button.setAttribute("aria-pressed", String(active));
    button.setAttribute("aria-disabled", String(active && controls.active.size === 1));
    button.title = active && controls.active.size === 1 ? "至少显示一项指标" : (active ? "隐藏" : "显示") + label;
    let total = null;
    for (const row of periodRows) {
      if (row[key] !== null) {
        const value = row[key][controls.unit];
        total = isCount ? (total ?? 0) + value : Math.max(total ?? 0, value);
      }
    }
    button.innerHTML = '<i class="swatch ' + key + '" aria-hidden="true"></i>' + label + " <b>" + formatValue(total) + "</b>";
  }
}

function renderChart() {
  renderLegend();
  chart.set(periodRows, controls.unit, keys.filter(key => controls.active.has(key)));
  for (const button of $("#units").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(controls.unit === button.dataset.unit));
  }
}

function renderTable() {
  const count = recordedRows.length;
  const pages = Math.max(1, Math.ceil(count / perPage));
  const start = controls.page * perPage;
  const pageRows = recordedRows.slice(start, start + perPage);
  $("#table-body").innerHTML = pageRows.length ? pageRows.map(row =>
    '<tr><th scope="row" class="table-date">' + row.date + "</th>" +
    dimensions.map(([key, unit]) => "<td>" + formatValue(row[key]?.[unit]) + "</td>").join("") + "</tr>"
  ).join("") : '<tr><td colspan="5">这段时间暂无记录</td></tr>';
  $("#record-count").textContent = count + " 天";
  $("#table-range").textContent = count ? "第 " + (start + 1) + "–" + (start + pageRows.length) + " 条，共 " + count + " 条" : "0 条记录";
  $("#page-count").textContent = (controls.page + 1) + " / " + pages;
  $("#previous").disabled = controls.page === 0;
  $("#next").disabled = controls.page === pages - 1;
  $("#export").disabled = !count;
}

function refreshRange() {
  let start = earliest;
  let end = latest;
  if (earliest && controls.range === "custom") {
    start = controls.customStart;
    end = controls.customEnd;
  } else if (earliest && controls.range !== "all") {
    start = isoDay(Math.max(dateTime(earliest), dateTime(latest) - (Number(controls.range) - 1) * dayMs));
  }
  periodRows = start ? eachDay(start, end) : [];
  recordedRows = periodRows.filter(row => row.clones !== null || row.views !== null).reverse();
  $("#trend-range").textContent = start ? shortRange(start, end) : "暂无归档";
  for (const button of $("#ranges").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.range === controls.range));
  }
  controls.page = 0;
  renderChart();
  renderTable();
}

function closeCalendar() {
  $("#custom-range").hidden = true;
  $('#ranges [data-range="custom"]').setAttribute("aria-expanded", "false");
}

$("#ranges").addEventListener("click", event => {
  const button = event.target.closest("button");
  if (!button) {
    return;
  }
  if (button.dataset.range === "custom") {
    const form = $("#custom-range");
    form.hidden = !form.hidden;
    button.setAttribute("aria-expanded", String(!form.hidden));
    if (!form.hidden) {
      $("#from-date").focus();
    }
    return;
  }
  closeCalendar();
  controls.range = button.dataset.range;
  refreshRange();
});

$("#custom-range").addEventListener("submit", event => {
  event.preventDefault();
  const from = $("#from-date").value;
  const to = $("#to-date").value;
  if (from > to) {
    $("#to-date").setCustomValidity("结束日期须等于或晚于开始日期");
    $("#to-date").reportValidity();
    return;
  }
  controls.customStart = from;
  controls.customEnd = to;
  controls.range = "custom";
  closeCalendar();
  $('#ranges [data-range="custom"]').focus();
  refreshRange();
});
$("#custom-range").addEventListener("input", () => $("#to-date").setCustomValidity(""));
$("#custom-range").addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeCalendar();
    $('#ranges [data-range="custom"]').focus();
  }
});

$("#units").addEventListener("click", event => {
  const button = event.target.closest("button");
  if (button && button.dataset.unit !== controls.unit) {
    controls.unit = button.dataset.unit;
    renderChart();
  }
});
$("#legend").addEventListener("click", event => {
  const key = event.target.closest("button")?.dataset.series;
  if (!key || (controls.active.has(key) && controls.active.size === 1)) {
    return;
  }
  if (controls.active.has(key)) {
    controls.active.delete(key);
  } else {
    controls.active.add(key);
  }
  renderChart();
});

$("#previous").addEventListener("click", () => {
  controls.page--;
  renderTable();
});
$("#next").addEventListener("click", () => {
  controls.page++;
  renderTable();
});
$("#export").addEventListener("click", () => {
  const lines = [["日期（UTC）", ...dimensions.map(([key, unit]) => unitNames[unit][key])].join(",")];
  for (const row of [...recordedRows].reverse()) {
    lines.push([row.date, ...dimensions.map(([key, unit]) => row[key]?.[unit] ?? "")].join(","));
  }
  const blob = new Blob(["\uFEFF" + lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = history.repository.split("/")[1] + "-traffic-" + periodRows[0].date + "_" + periodRows.at(-1).date + ".csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("#theme").addEventListener("click", () => {
  const isDark = getComputedStyle(document.documentElement).colorScheme === "dark";
  document.documentElement.dataset.theme = isDark ? "light" : "dark";
});

overview();
const calendarButton = $('#ranges [data-range="custom"]');
calendarButton.disabled = !earliest;
if (earliest) {
  for (const input of [$("#from-date"), $("#to-date")]) {
    input.min = earliest;
    input.max = latest;
  }
  $("#from-date").value = earliest;
  $("#to-date").value = latest;
}
refreshRange();
