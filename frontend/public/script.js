const BENGALURU_CENTER = [12.9716, 77.5946];
const API = {
  health: "/api/health",
  forecastHour: (hour) => `/api/forecast/hour/${hour}`,
  forecastStreet: (lat, lon) => `/api/forecast/street?lat=${lat}&lon=${lon}`,
  route: "/api/route",
};

// ── Topbar progress ──────────────────────────────────
const topbar = document.getElementById('topbar');
function progressStart() {
  topbar.style.width = '0%';
  topbar.style.opacity = '1';
  requestAnimationFrame(() => { topbar.style.width = '70%'; });
}
function progressDone() {
  topbar.style.width = '100%';
  setTimeout(() => { topbar.style.opacity = '0'; topbar.style.width = '0%'; }, 400);
}

let map;
let markerLayer;
let routeLayer;
let heatLayer;
let originMarker;
let destinationMarker;
let activeTab = "routingTab";

// Nominatim Geocoding API
const NOMINATIM_URL = "https://nominatim.openstreetmap.org/search?format=json&limit=5&q=";

// DOM Elements
const datetimeInput = document.getElementById("datetimeInput");
const routeBtn = document.getElementById("routeBtn");
const originInput = document.getElementById("originInput");
const destinationInput = document.getElementById("destinationInput");
const profileSelect = document.getElementById("profileSelect");
const transportModeSelect = document.getElementById("transportModeSelect");
const statusEl = document.getElementById("status");
const routeSummary = document.getElementById("routeSummary");

// XAI
const xaiInsightEl = document.getElementById("xaiInsight");
const xaiTextEl = document.getElementById("xaiText");
const showAttentionBtn = document.getElementById("showAttentionBtn");
let currentHighlights = [];
let attentionLayer;

// Autocomplete DOM
const originSuggestions = document.getElementById("originSuggestions");
const destinationSuggestions = document.getElementById("destinationSuggestions");

// Charts
let chart24 = null;
let chart7D = null;
let chartCompare = null;
const comparePoints = []; // Stores {lat, lon, label, data: []} up to 3

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

// Calculate hour offset (1-168) from now
function getHourOffset() {
  if (!datetimeInput.value) return 1;
  const selectedTime = new Date(datetimeInput.value);
  const now = new Date();
  const diffHours = Math.floor((selectedTime - now) / (1000 * 60 * 60));
  if (diffHours < 1) return 1;
  if (diffHours > 168) return 168;
  return diffHours;
}

function parseLatLon(value) {
  const parts = value.split(",").map((part) => Number.parseFloat(part.trim()));
  if (parts.length !== 2 || parts.some((part) => Number.isNaN(part))) {
    throw new Error("Use latitude, longitude or select an address");
  }
  return { lat: parts[0], lon: parts[1] };
}

function formatLatLon(latlng) {
  return `${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}`;
}

function aqiRadius(aqi) {
  return Math.max(8, Math.min(28, Number(aqi) / 8));
}

function markerHtml(point) {
  return `
    <strong>${point.station_name}</strong><br>
    AQI ${point.aqi} - ${point.category}<br>
    <span>${point.timestamp}</span>
  `;
}

// Map Setup
function ensureMap() {
  if (map) return;

  map = L.map("map", { zoomControl: false }).setView(BENGALURU_CENTER, 11);
  L.control.zoom({ position: "bottomleft" }).addTo(map);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  }).addTo(map);

  markerLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);

  map.on("click", handleMapClick);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return payload;
}

// Map Click Handler depending on active tab
async function handleMapClick(event) {
  const latlng = event.latlng;
  
  if (activeTab === "routingTab") {
    // Determine if setting origin or destination based on which is empty
    if (!originInput.value || (originInput.value && destinationInput.value)) {
      originInput.value = formatLatLon(latlng);
      if (originMarker) originMarker.remove();
      originMarker = L.marker(latlng).addTo(map).bindPopup("Origin").openPopup();
      if (originInput.value && destinationInput.value) {
          destinationInput.value = "";
          if (destinationMarker) destinationMarker.remove();
      }
    } else {
      destinationInput.value = formatLatLon(latlng);
      if (destinationMarker) destinationMarker.remove();
      destinationMarker = L.marker(latlng).addTo(map).bindPopup("Destination").openPopup();
    }
  } else if (activeTab === "forecast24Tab") {
    await loadStreetForecast(latlng, 24);
  } else if (activeTab === "forecast7DTab") {
    await loadStreetForecast(latlng, 168);
  } else if (activeTab === "compareTab") {
    await addCompareLocation(latlng);
  }
}

// Heatmap / Station loader
async function loadForecastMap(hour) {
  ensureMap();
  setStatus("Loading map data...");

  const points = await fetchJson(API.forecastHour(hour));
  markerLayer.clearLayers();

  const heatPoints = points.map((point) => [
    point.lat,
    point.lon,
    Math.max(0.15, Math.min(1, Number(point.aqi) / 300)),
  ]);

  points.forEach((point) => {
    L.circleMarker([point.lat, point.lon], {
      radius: aqiRadius(point.aqi),
      color: "#111827",
      weight: 1,
      fillColor: point.color,
      fillOpacity: 0.82,
    })
      .bindPopup(markerHtml(point))
      .addTo(markerLayer);
  });

  if (heatLayer) heatLayer.remove();
  if (L.heatLayer) {
    heatLayer = L.heatLayer(heatPoints, {
      radius: 42,
      blur: 34,
      maxZoom: 13,
      gradient: { 0.2: "#00e400", 0.45: "#ffff00", 0.7: "#ff7e00", 1: "#7e0023" },
    }).addTo(map);
  }

  setStatus(`${points.length} stations loaded for hour ${hour}.`);
}

// Routing Logic
function routePopup(title, route) {
  return `
    <strong>${title}</strong><br>
    ${route.distance_km} km - ${route.travel_time_minutes} min<br>
    Avg AQI ${route.average_aqi}
  `;
}

function drawRoute(route, color, title) {
  // 1. Soft wide glow
  L.polyline(route.coordinates, {
    color: color,
    weight: 16,
    opacity: 0.15,
    lineCap: 'round',
    lineJoin: 'round',
    className: 'premium-route-glow'
  }).addTo(routeLayer);

  // 2. Dark sharp outline to separate from map
  L.polyline(route.coordinates, {
    color: '#0a0d14',
    weight: 8,
    opacity: 0.85,
    lineCap: 'round',
    lineJoin: 'round'
  }).addTo(routeLayer);

  // 3. Bright vibrant core line
  const polyline = L.polyline(route.coordinates, {
    color: color,
    weight: 4,
    opacity: 1,
    lineCap: 'round',
    lineJoin: 'round',
    className: 'premium-route-core'
  })
    .bindPopup(routePopup(title, route))
    .addTo(routeLayer);

  // Draw end-cap dots
  if (route.coordinates.length > 0) {
    const start = route.coordinates[0];
    const end = route.coordinates[route.coordinates.length - 1];
    
    L.circleMarker(start, { radius: 5, color: '#0a0d14', weight: 2, fillColor: color, fillOpacity: 1 }).addTo(routeLayer);
    L.circleMarker(end, { radius: 5, color: '#0a0d14', weight: 2, fillColor: color, fillOpacity: 1 }).addTo(routeLayer);
  }

  return polyline;
}

// findBestTravelTime removed; optimal route departure is now handled by the backend

async function findRoutes() {
  routeLayer.clearLayers();
  routeSummary.style.display = 'none';
  routeSummary.innerHTML = '';
  setStatus("Finding routes…");
  progressStart();

  const routeBtn = document.getElementById('routeBtn');
  routeBtn.disabled = true;
  routeBtn.innerHTML = '<span class="spinner"></span> Finding Routes…';

  try {
    const origin = parseLatLon(originInput.value);
    
    const destInputs = document.querySelectorAll('.dest-input');
    const waypoints = [ {lat: origin.lat, lon: origin.lon} ];
    destInputs.forEach(inp => {
        if (inp.value.trim()) {
            const pt = parseLatLon(inp.value);
            waypoints.push({lat: pt.lat, lon: pt.lon});
        }
    });
    
    const isTSP = waypoints.length > 2;
    const hour = getHourOffset();

    const selectedTime = datetimeInput.value ? new Date(datetimeInput.value) : new Date();
    const midnight = new Date(selectedTime);
    midnight.setHours(24, 0, 0, 0);
    const hoursToScan = Math.max(1, Math.floor((midnight - selectedTime) / (1000 * 60 * 60)));

    let payload, endpoint;
    if (isTSP) {
        payload = {
            waypoints: waypoints,
            profile: profileSelect.value,
            hour: hour,
            transport_mode: transportModeSelect.value
        };
        endpoint = "/api/route/tsp";
    } else {
        payload = {
            orig_lat: origin.lat,
            orig_lon: origin.lon,
            dest_lat: waypoints[1].lat,
            dest_lon: waypoints[1].lon,
            profile: profileSelect.value,
            hour: hour,
            transport_mode: transportModeSelect.value,
            hours_to_scan: hoursToScan,
        };
        endpoint = API.route;
    }

    const result = await fetchJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (isTSP) {
      drawRoute(result.data.cleanest, '#3d7a5c', 'Cleanest TSP route');
      const cleanest = result.data.cleanest;
      routeSummary.style.display = 'block';
      routeSummary.innerHTML = `
        <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:10px">Multi-Stop Route Analysis</div>
        <div style="padding:10px;background:rgba(61,122,92,0.1);border:1px solid rgba(61,122,92,0.2);border-radius:4px">
          <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:#3d7a5c;margin-bottom:4px">Optimal Sequence</div>
          <div style="font-size:0.9rem;font-weight:600">${cleanest.travel_time_minutes} min</div>
          <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">${cleanest.distance_km} km &middot; AQI ${cleanest.average_aqi}</div>
        </div>
      `;
      map.fitBounds(routeLayer.getBounds(), { padding: [40, 40] });
    } else {
      const fastestLine = drawRoute(result.data.fastest, '#3b82f6', 'Fastest route'); // Bright Google-like blue
      drawRoute(result.data.cleanest, '#10b981', 'Cleanest route'); // Vibrant emerald green
      map.fitBounds(fastestLine.getBounds(), { padding: [40, 40] });

      const optimal = result.data.optimal_departure;
      const fastest = result.data.fastest;
      const cleanest = result.data.cleanest;

      const savingsUg = Math.max(0, (fastest.exposure_aqi_hours - cleanest.exposure_aqi_hours) * 1000);

      let bestTimeHtml = '';
      if (optimal && optimal.hourly_route_aqi.length > 0) {
        const bestTime = new Date(selectedTime.getTime() + optimal.best_hour_offset * 60 * 60 * 1000);
        bestTime.setMinutes(0, 0, 0);
        const timeStr = bestTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true });

        const hourly = optimal.hourly_route_aqi;
        const scanHours = hourly.length;
        const h0 = selectedTime.getHours();

        let morning = { sum: 0, count: 0 }, midday = { sum: 0, count: 0 }, evening = { sum: 0, count: 0 };
        
        for (let i = 0; i < scanHours; i++) {
           const h = (h0 + i) % 24;
           if (h >= 0 && h < 12) {
               morning.sum += hourly[i]; morning.count++;
           } else if (h >= 12 && h < 18) {
               midday.sum += hourly[i]; midday.count++;
           } else {
               evening.sum += hourly[i]; evening.count++;
           }
        }
        
        const segments = [];
        if (morning.count > 0) segments.push({ label: 'Morning', avg: (morning.sum / morning.count).toFixed(0) });
        if (midday.count > 0) segments.push({ label: 'Midday', avg: (midday.sum / midday.count).toFixed(0) });
        if (evening.count > 0) segments.push({ label: 'Evening', avg: (evening.sum / evening.count).toFixed(0) });

        const segRows = segments.map(s =>
          `<span style="margin-right:14px;color:var(--text-secondary)">${s.label} <b style="color:var(--text-primary)">${s.avg}</b></span>`
        ).join('');

        bestTimeHtml = `
          <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border)">
            <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:8px">Optimal Departure Today</div>
            <div style="font-size:1.05rem;font-weight:600;color:var(--gold-light);letter-spacing:0.02em">${timeStr}</div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">Projected Route AQI <b style="color:var(--text-primary)">${optimal.best_avg_aqi.toFixed(0)}</b> &mdash; ${optimal.best_hour_offset === 0 ? 'Depart now' : `In ${optimal.best_hour_offset}h`}</div>
            <div style="margin-top:10px;font-size:0.72rem">${segRows}</div>
          </div>`;
      }

      routeSummary.style.display = 'block';
      routeSummary.innerHTML = `
        <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:10px">Route Analysis</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <div style="padding:10px;background:rgba(74,111,165,0.1);border:1px solid rgba(74,111,165,0.2);border-radius:4px">
            <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a6fa5;margin-bottom:4px">Fastest</div>
            <div style="font-size:0.9rem;font-weight:600">${fastest.travel_time_minutes} min</div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">${fastest.distance_km} km &middot; AQI ${fastest.average_aqi}</div>
          </div>
          <div style="padding:10px;background:rgba(61,122,92,0.1);border:1px solid rgba(61,122,92,0.2);border-radius:4px">
            <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:#3d7a5c;margin-bottom:4px">Cleanest</div>
            <div style="font-size:0.9rem;font-weight:600">${cleanest.travel_time_minutes} min</div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">${cleanest.distance_km} km &middot; AQI ${cleanest.average_aqi}</div>
          </div>
        </div>
        <button onclick="saveToWallet(${savingsUg})" class="secondary-btn" style="margin-top:10px;width:100%">Save Route to Health Wallet</button>
        ${bestTimeHtml}
      `;
    }

    try {
      const xaiResult = await fetchJson("/api/xai/route_insight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      xaiInsightEl.style.display = "block";
      xaiTextEl.textContent = xaiResult.insight;
      currentHighlights = xaiResult.highlights || [];
    } catch (_) {
      xaiInsightEl.style.display = "none";
    }

    setStatus("Routes ready ✓");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    routeBtn.disabled = false;
    routeBtn.innerHTML = 'Find Optimal Routes';
    progressDone();
  }
}

// Geocoding Autocomplete
let debounceTimeout;
async function fetchGeocode(query, listEl, inputEl) {
  if (query.length < 3) {
    listEl.style.display = 'none';
    return;
  }
  
  clearTimeout(debounceTimeout);
  debounceTimeout = setTimeout(async () => {
    try {
      const response = await fetch(NOMINATIM_URL + encodeURIComponent(query + ", Bengaluru"));
      const data = await response.json();
      listEl.innerHTML = '';
      if (data.length > 0) {
        data.forEach(item => {
          const li = document.createElement('li');
          li.textContent = item.display_name;
          li.addEventListener('click', () => {
            inputEl.value = `${item.lat}, ${item.lon}`;
            listEl.style.display = 'none';
            // update map markers
            if (inputEl.id === 'originInput') {
              if (originMarker) originMarker.remove();
              originMarker = L.marker([item.lat, item.lon]).addTo(map).bindPopup("Origin").openPopup();
              map.setView([item.lat, item.lon], 13);
            } else {
              if (destinationMarker) destinationMarker.remove();
              destinationMarker = L.marker([item.lat, item.lon]).addTo(map).bindPopup("Destination").openPopup();
              map.setView([item.lat, item.lon], 13);
            }
          });
          listEl.appendChild(li);
        });
        listEl.style.display = 'block';
      } else {
        listEl.style.display = 'none';
      }
    } catch (e) {
      console.error("Geocode error", e);
    }
  }, 500);
}

originInput.addEventListener('input', (e) => fetchGeocode(e.target.value, originSuggestions, originInput));
destinationInput.addEventListener('input', (e) => fetchGeocode(e.target.value, destinationSuggestions, destinationInput));

const addStopBtn = document.getElementById("addStopBtn");
const destinationsContainer = document.getElementById("destinationsContainer");

if (addStopBtn) {
  addStopBtn.addEventListener("click", () => {
    const wrapper = document.createElement("div");
    wrapper.className = "autocomplete-wrapper dest-wrapper";
    wrapper.style.marginBottom = "8px";
    const newId = "dest" + Date.now();
    wrapper.innerHTML = `
      <input id="${newId}" class="dest-input" type="text" placeholder="Stop..." autocomplete="off" />
      <ul class="suggestions-list" style="display:none;"></ul>
    `;
    destinationsContainer.appendChild(wrapper);
    
    const input = wrapper.querySelector("input");
    const suggestions = wrapper.querySelector("ul");
    input.addEventListener("input", (e) => fetchGeocode(e.target.value, suggestions, input));
  });
}

document.addEventListener('click', (e) => {
  if (!e.target.closest('.autocomplete-wrapper')) {
    document.querySelectorAll('.suggestions-list').forEach(el => el.style.display = 'none');
  }
});

// Chart Integration (Street Level)
function getChartConfig(labels, datasets, title) {
  return {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#5a6270',
            font: { family: "'IBM Plex Mono', monospace", size: 10 },
            boxWidth: 12,
            padding: 14
          }
        },
        title: {
          display: true,
          text: title,
          color: '#e4e1d9',
          font: { family: "'Inter', sans-serif", size: 11, weight: '600' },
          padding: { bottom: 12 }
        }
      },
      scales: {
        y: {
          grid: { color: 'rgba(255,255,255,0.04)', lineWidth: 1 },
          border: { color: 'rgba(255,255,255,0.07)' },
          ticks: {
            color: '#5a6270',
            font: { family: "'IBM Plex Mono', monospace", size: 10 }
          },
          title: { display: true, text: 'AQI', color: '#5a6270', font: { size: 10 } }
        },
        x: {
          grid: { color: 'rgba(255,255,255,0.03)', lineWidth: 1 },
          border: { color: 'rgba(255,255,255,0.07)' },
          ticks: {
            color: '#5a6270',
            font: { family: "'IBM Plex Mono', monospace", size: 9 },
            maxTicksLimit: 10
          }
        }
      }
    }
  };
}

function generateLabels(hours) {
  const labels = [];
  let d = new Date();
  d.setMinutes(0,0,0);
  for (let i = 0; i < hours; i++) {
    labels.push(d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
    d.setHours(d.getHours() + 1);
  }
  return labels;
}

async function loadStreetForecast(latlng, hours) {
  setStatus(`Fetching ${hours === 24 ? '24h' : '7-Day'} forecast...`);
  try {
    const data = await fetchJson(API.forecastStreet(latlng.lat, latlng.lng));
    
    let aqiData = [];
    let labels = [];
    let titleStr = '';

    if (hours === 24) {
      aqiData = data.hourly_aqi.slice(0, 24);
      labels = generateLabels(24);
      titleStr = 'Street Forecast (24h)';
    } else {
      // Aggregate into 7 days
      let d = new Date();
      for (let day = 0; day < 7; day++) {
        let slice = data.hourly_aqi.slice(day * 24, (day + 1) * 24);
        if (slice.length === 0) break;
        let avg = slice.reduce((a, b) => a + b, 0) / slice.length;
        aqiData.push(avg);
        labels.push(d.toLocaleDateString([], {weekday: 'short', month: 'short', day: 'numeric'}));
        d.setDate(d.getDate() + 1);
      }
      titleStr = 'Street Forecast (7-Day Daily Avg)';
    }
    
    L.popup()
      .setLatLng(latlng)
      .setContent(`<strong>Street Selected</strong><br>${formatLatLon(latlng)}`)
      .openOn(map);

    const canvasId = hours === 24 ? "streetForecastChart24" : "streetForecastChart7D";
    const detailsId = hours === 24 ? "forecastDetails24" : "forecastDetails7D";
    const ctx = document.getElementById(canvasId).getContext("2d");
    
    let targetChart = hours === 24 ? chart24 : chart7D;
    
    if (targetChart) {
      targetChart.destroy();
    }
    
    const newChart = new Chart(ctx, getChartConfig(labels, [{
      label: `AQI — ${formatLatLon(latlng)}`,
      data: aqiData,
      borderColor: '#b8922a',
      backgroundColor: 'rgba(184, 146, 42, 0.06)',
      borderWidth: 1.5,
      pointRadius: 2,
      pointBackgroundColor: '#b8922a',
      fill: true,
      tension: 0.3
    }], titleStr));

    if (hours === 24) chart24 = newChart;
    else chart7D = newChart;
    
    document.getElementById(detailsId).style.display = "block";
    document.getElementById(detailsId).innerHTML = `Avg AQI: ${(aqiData.reduce((a,b)=>a+b,0)/aqiData.length).toFixed(1)} | Max: ${Math.max(...aqiData).toFixed(1)}`;
    setStatus("Forecast loaded.");
  } catch (e) {
    setStatus(e.message, true);
  }
}

// Compare Logic
const COMPARE_COLORS = ['#b8922a', '#4a7c6f', '#7a6a5a'];

async function addCompareLocation(latlng) {
  if (comparePoints.length >= 3) {
    setStatus("Maximum 3 locations allowed for comparison.", true);
    return;
  }
  
  setStatus("Fetching forecast for comparison...");
  try {
    const data = await fetchJson(API.forecastStreet(latlng.lat, latlng.lng));
    const locationId = comparePoints.length + 1;
    
    comparePoints.push({
      lat: latlng.lat,
      lon: latlng.lng,
      label: `Loc ${locationId}: ${formatLatLon(latlng)}`,
      data: data.hourly_aqi
    });
    
    L.marker(latlng).addTo(map).bindPopup(`Location ${locationId}`).openPopup();
    updateCompareChart();
    setStatus("Location added to comparison.");
  } catch(e) {
    setStatus(e.message, true);
  }
}

function updateCompareChart() {
  const ctx = document.getElementById("compareChart").getContext("2d");
  if (chartCompare) chartCompare.destroy();
  
  if (comparePoints.length === 0) return;
  
  // Aggregate labels and datasets for 7 daily points
  const labels = [];
  let d = new Date();
  for (let day = 0; day < 7; day++) {
    labels.push(d.toLocaleDateString([], {weekday: 'short', month: 'short', day: 'numeric'}));
    d.setDate(d.getDate() + 1);
  }

  const datasets = comparePoints.map((pt, i) => {
    let aggregatedData = [];
    for (let day = 0; day < 7; day++) {
      let slice = pt.data.slice(day * 24, (day + 1) * 24);
      if (slice.length === 0) break;
      let avg = slice.reduce((a, b) => a + b, 0) / slice.length;
      aggregatedData.push(avg);
    }
    return {
      label: pt.label,
      data: aggregatedData,
      borderColor: COMPARE_COLORS[i],
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 2,
      pointBackgroundColor: COMPARE_COLORS[i],
      tension: 0.3
    };
  });
  
  chartCompare = new Chart(ctx, getChartConfig(labels, datasets, "7-Day Location Comparison (Daily Avg)"));
}

document.getElementById('clearCompareBtn').addEventListener('click', () => {
  comparePoints.length = 0;
  if (chartCompare) chartCompare.destroy();
  // Clear map markers (simplistic approach: clear routing/forecast ones too unless we track compare markers separately)
  // For simplicity, let's just clear popups/markers by re-initializing map layers
  ensureMap();
  map.eachLayer((layer) => {
    if (layer instanceof L.Marker && layer !== originMarker && layer !== destinationMarker) {
      map.removeLayer(layer);
    }
  });
  setStatus("Comparison cleared.");
});

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    // UI toggle
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    const tabId = e.target.getAttribute('data-tab');
    e.target.classList.add('active');
    document.getElementById(tabId).classList.add('active');
    
    activeTab = tabId;
  });
});

async function boot() {
  ensureMap();
  // Init datetime input to now
  const now = new Date();
  now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
  datetimeInput.value = now.toISOString().slice(0, 16);

  progressStart();
  try {
    const health = await fetchJson(API.health);
    if (!health.forecast_loaded) {
      setStatus("Forecast data missing. Run the forecast pipeline.", true);
      progressDone();
      return;
    }
    await loadForecastMap(1);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    progressDone();
  }
}

routeBtn.addEventListener("click", () => {
  if (attentionLayer) map.removeLayer(attentionLayer);
  findRoutes().catch((error) => setStatus(error.message, true));
});

showAttentionBtn.addEventListener("click", async () => {
  if (attentionLayer) {
    map.removeLayer(attentionLayer);
    attentionLayer = null;
  }

  // If we have highlights from a recent route, use those; otherwise fetch fresh
  let edges = currentHighlights;
  if (!edges || edges.length === 0) {
    setStatus("Fetching attention weights from model...");
    try {
      const rawWeights = await fetchJson("/api/xai/attention");
      // rawWeights has from_node/to_node keys; map to coords
      edges = rawWeights
        .filter(w => w.from_node !== w.to_node)
        .slice(0, 15)
        .map(w => {
          const STATION_COORDS = {
            "site_1553": [12.9279, 77.6271], "site_162": [12.9174, 77.6235],
            "site_165": [12.9166, 77.6101], "site_1554": [13.0450, 77.5966],
            "site_1555": [12.9250, 77.5938], "site_5729": [13.0289, 77.5199],
            "site_5681": [12.9634, 77.5559], "site_163": [12.9609, 77.5996],
            "site_5678": [12.9774, 77.5713], "site_166": [13.0068, 77.5090],
            "site_5686": [13.0450, 77.5116], "site_1558": [13.0219, 77.5421],
          };
          const from = STATION_COORDS[w.from_node];
          const to = STATION_COORDS[w.to_node];
          return from && to ? { from_lat: from[0], from_lon: from[1], to_lat: to[0], to_lon: to[1], weight: w.weight } : null;
        })
        .filter(Boolean);
    } catch (e) {
      setStatus("Could not load attention weights: " + e.message, true);
      return;
    }
  }

  if (!edges || edges.length === 0) {
    setStatus("No cross-station attention edges to visualize.", true);
    return;
  }

  attentionLayer = L.layerGroup().addTo(map);

  edges.forEach((edge) => {
    const start = [edge.from_lat, edge.from_lon];
    const end = [edge.to_lat, edge.to_lon];
    const w = Math.min(1.0, edge.weight);

    // Animated dashed line (Wind Advection)
    L.polyline([start, end], {
      color: `hsl(${Math.round((1 - w) * 120)}, 90%, 60%)`, // green→red by weight
      weight: 2 + w * 6,
      opacity: 0.4 + w * 0.6,
      dashArray: "8, 6",
      className: "wind-particle-line"
    })
      .bindPopup(`
        <strong>Attention Edge</strong><br>
        Weight: <b>${edge.weight.toFixed(3)}</b><br>
        <span style="color:#f59e0b">Higher weight = stronger wind-driven coupling</span>
      `)
      .addTo(attentionLayer);

    // Arrow head at destination
    L.circleMarker(end, {
      radius: 4 + w * 4,
      color: `hsl(${Math.round((1 - w) * 120)}, 90%, 60%)`,
      fillColor: `hsl(${Math.round((1 - w) * 120)}, 90%, 60%)`,
      fillOpacity: 0.8,
      weight: 1
    })
      .bindPopup(`Destination station — Weight: ${edge.weight.toFixed(3)}`)
      .addTo(attentionLayer);
  });

  map.fitBounds(attentionLayer.getBounds ? attentionLayer.getLayers()
    .filter(l => l.getBounds).reduce((b, l) => b.extend(l.getBounds()), L.latLngBounds()) : undefined);

  setStatus(`✓ Visualized ${edges.length} dynamic graph attention edges (green=low, red=high weight).`);
});

boot();
