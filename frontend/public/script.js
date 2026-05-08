const BENGALURU_CENTER = [12.9716, 77.5946];

// Update this to your Hugging Face Space URL when deploying the backend
// Example: const BASE_URL = "https://yourusername-blr-aqi-api.hf.space";
const BASE_URL = ""; 

const API = {
  health: `${BASE_URL}/api/health`,
  forecastHour: (hour) => `${BASE_URL}/api/forecast/hour/${hour}`,
  forecastSurface: (hour, stride = 3) => `${BASE_URL}/api/forecast/surface/${hour}?stride=${stride}`,
  forecastStreet: (lat, lon) => `${BASE_URL}/api/forecast/street?lat=${lat}&lon=${lon}`,
  route: `${BASE_URL}/api/route`,
  routeTsp: `${BASE_URL}/api/route/tsp`,
  xaiInsight: `${BASE_URL}/api/xai/route_insight`,
  xaiAttention: `${BASE_URL}/api/xai/attention`,
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
let userLocationMarker;
let userAccuracyCircle;
let locationWatchId;
let followUserLocation = true;
let lastLocalAqiUpdate = 0;
let currentRouteOptions = {};
let selectedRouteKey = "least_harm";
let navigationActive = false;

// Nominatim Geocoding API
const NOMINATIM_URL = "https://nominatim.openstreetmap.org/search?format=json&limit=5&q=";

const MAP_ATTRIBUTION = '&copy; OpenStreetMap contributors &copy; CARTO';

// DOM Elements
const datetimeInput = document.getElementById("datetimeInput");
const routeBtn = document.getElementById("routeBtn");
const originInput = document.getElementById("originInput");
const destinationInput = document.getElementById("destinationInput");
const profileSelect = document.getElementById("profileSelect");
const transportModeSelect = document.getElementById("transportModeSelect");
const statusEl = document.getElementById("status");
const routeSummary = document.getElementById("routeSummary");
const myLocationBtn = document.getElementById("myLocationBtn");
const healthTipsContainer = document.getElementById("healthTipsContainer");
const localAqiCard = document.getElementById("localAqiCard");
const localAqiValue = document.getElementById("localAqiValue");
const localAqiCategory = document.getElementById("localAqiCategory");
const localAqiLocation = document.getElementById("localAqiLocation");

// XAI
const xaiInsightEl = document.getElementById("xaiInsight");
const xaiTextEl = document.getElementById("xaiText");
const showAttentionBtn = document.getElementById("showAttentionBtn");
let currentHighlights = [];
let attentionLayer;

// Autocomplete DOM
const originSuggestions = document.getElementById("originSuggestions");
const destinationSuggestions = document.getElementById("destinationSuggestions");
const forecast24Input = document.getElementById("forecast24Input");
const forecast24Suggestions = document.getElementById("forecast24Suggestions");
const forecast7DInput = document.getElementById("forecast7DInput");
const forecast7DSuggestions = document.getElementById("forecast7DSuggestions");

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
    <span>${point.timestamp || "station anchor"}</span>
  `;
}

function metersLabel(value) {
  const meters = Number(value) || 0;
  if (meters >= 1000) return `${(meters / 1000).toFixed(1)} km`;
  return `${Math.max(0, Math.round(meters))} m`;
}

function createUserLocationIcon(heading = 0) {
  const rotation = Number.isFinite(heading) ? heading : 0;
  return L.divIcon({
    className: "user-location-wrapper",
    html: `
      <div class="user-location-pulse"></div>
      <div class="user-location-puck" style="transform: rotate(${rotation}deg)">
        <div class="user-location-arrow"></div>
      </div>
    `,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
  });
}

function updateUserLocationMarker(position, { recenter = false } = {}) {
  if (!map || !position?.coords) return;
  const latlng = { lat: position.coords.latitude, lng: position.coords.longitude };
  const heading = Number.isFinite(position.coords.heading) ? position.coords.heading : 0;
  const accuracy = Math.max(5, Number(position.coords.accuracy) || 20);

  if (!userLocationMarker) {
    userLocationMarker = L.marker(latlng, {
      icon: createUserLocationIcon(heading),
      zIndexOffset: 1200,
      interactive: false,
    }).addTo(map);
  } else {
    userLocationMarker.setLatLng(latlng);
    userLocationMarker.setIcon(createUserLocationIcon(heading));
  }

  if (!userAccuracyCircle) {
    userAccuracyCircle = L.circle(latlng, {
      radius: accuracy,
      color: "#1a73e8",
      weight: 1,
      fillColor: "#1a73e8",
      fillOpacity: 0.08,
      interactive: false,
    }).addTo(map);
  } else {
    userAccuracyCircle.setLatLng(latlng);
    userAccuracyCircle.setRadius(accuracy);
  }

  if (recenter || followUserLocation) {
    map.setView(latlng, Math.max(map.getZoom(), 16), { animate: true });
  }

  return latlng;
}

function setLocalAqiState({ value = "--", category = "Waiting", location = "Use location for present prediction", color = "#5a6270", loading = false } = {}) {
  if (!localAqiCard) return;
  localAqiValue.textContent = value;
  localAqiCategory.textContent = category;
  localAqiLocation.textContent = location;
  localAqiCard.style.borderLeftColor = color;
  localAqiCard.classList.toggle("loading", loading);
}

function syncLocalAqiCardFromForecast(data) {
  if (!data?.hourly_aqi?.length) return;
  const currentAqi = Number(data.current?.aqi ?? data.hourly_aqi[0]);
  const uncertainty = Number(data.current?.uncertainty ?? data.hourly_uncertainty?.[0] ?? 0);
  const insight = getHealthTips(currentAqi, profileSelect.value);
  const snapped = {
    lat: Number(data.actual_lat ?? data.requested_lat ?? 0),
    lng: Number(data.actual_lon ?? data.requested_lon ?? 0),
  };
  setLocalAqiState({
    value: currentAqi.toFixed(0),
    category: uncertainty > 0 ? `${insight.category} +/- ${uncertainty.toFixed(0)}` : insight.category,
    location: `Predicted now near ${formatLatLon(snapped)}`,
    color: insight.color,
    loading: false,
  });
}

async function selectForecastLocation(latlng, hours) {
  map.setView(latlng, 14);
  await loadStreetForecast(latlng, hours);
}

function updateOriginFromLatLng(latlng, label = "My Location") {
  originInput.value = formatLatLon(latlng);
  if (originMarker) originMarker.remove();
  originMarker = L.marker(latlng).addTo(map).bindPopup(label).openPopup();
  map.setView(latlng, 14);
}

async function updateLocalStreetAqi(latlng, { setOrigin = false } = {}) {
  if (setOrigin) updateOriginFromLatLng(latlng);

  setLocalAqiState({
    value: "--",
    category: "Predicting",
    location: formatLatLon(latlng),
    color: "#b8922a",
    loading: true,
  });

  const data = await fetchJson(API.forecastStreet(latlng.lat, latlng.lng));
  if (!data?.hourly_aqi?.length) {
    throw new Error("No street AQI forecast is available for this location.");
  }

  const currentAqi = Number(data.hourly_aqi[0]);
  const uncertainty = Number(data.current?.uncertainty ?? data.hourly_uncertainty?.[0] ?? 0);
  const insight = getHealthTips(currentAqi, profileSelect.value);
  const snapped = { lat: data.actual_lat, lng: data.actual_lon };
  const explainer = data.explanation || {};
  const reasons = Array.isArray(explainer.reasons) ? explainer.reasons : [];
  const tips = Array.isArray(explainer.tips) ? explainer.tips : [];

  setLocalAqiState({
    value: currentAqi.toFixed(0),
    category: uncertainty > 0 ? `${insight.category} ±${uncertainty.toFixed(0)}` : insight.category,
    location: `Predicted now near ${formatLatLon(snapped)}`,
    color: insight.color,
    loading: false,
  });

  if (healthTipsContainer) {
    healthTipsContainer.style.display = "block";
    healthTipsContainer.style.borderLeftColor = insight.color;
    healthTipsContainer.innerHTML = `
      <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:6px">Current Location AQI: <strong style="color:${insight.color}; font-size:1rem; margin-left:4px;">${currentAqi.toFixed(0)}${uncertainty > 0 ? ` ± ${uncertainty.toFixed(0)}` : ''}</strong></div>
      <div style="font-size:0.8rem; font-weight:600; color:var(--text-primary); margin-bottom:4px;">${insight.category}</div>
      <div style="font-size:0.75rem; color:var(--text-secondary); line-height: 1.5;">${explainer.summary || insight.tip}</div>
      ${reasons.length ? `<div style="margin-top:8px;font-size:0.72rem;color:var(--text-secondary);line-height:1.45">${reasons.slice(0, 2).map(item => `• ${item}`).join('<br>')}</div>` : ''}
      ${tips.length ? `<div style="margin-top:8px;font-size:0.72rem;color:var(--text-primary);line-height:1.45">${tips.slice(0, 2).map(item => `• ${item}`).join('<br>')}</div>` : ''}
    `;
  }

  return { currentAqi, insight, data };
}

function requestCurrentLocation({ setOrigin = true, quiet = false } = {}) {
  if (!navigator.geolocation) {
    setLocalAqiState({ category: "Unavailable", location: "Geolocation is not supported", color: "#8b3a3a" });
    if (!quiet) setStatus("Geolocation not supported.", true);
    return;
  }

  if (!quiet) setStatus("Acquiring location...");
  setLocalAqiState({ category: "Locating", location: "Waiting for browser permission", color: "#b8922a", loading: true });

  navigator.geolocation.getCurrentPosition(
    async (position) => {
      updateUserLocationMarker(position, { recenter: setOrigin });
      const latlng = { lat: position.coords.latitude, lng: position.coords.longitude };
      try {
        await updateLocalStreetAqi(latlng, { setOrigin });
        if (!quiet) setStatus("Location AQI ready");
      } catch (error) {
        setLocalAqiState({ category: "Error", location: error.message, color: "#8b3a3a" });
        if (!quiet) setStatus(error.message, true);
      }
    },
    (error) => {
      const message = error.code === error.PERMISSION_DENIED
        ? "Location permission was denied"
        : "Could not read your location";
      setLocalAqiState({ category: "Permission needed", location: "Tap the location button to try again", color: "#8b3a3a" });
      if (!quiet) setStatus(message, true);
    },
    { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 }
  );
}

function startLocationWatch() {
  if (!navigator.geolocation || locationWatchId != null) return;
  locationWatchId = navigator.geolocation.watchPosition(
    async (position) => {
      const latlng = updateUserLocationMarker(position);
      const now = Date.now();
      if (latlng && now - lastLocalAqiUpdate > 90000) {
        lastLocalAqiUpdate = now;
        try {
          await updateLocalStreetAqi(latlng, { setOrigin: false });
        } catch (_) {
          // Keep the moving navigation puck alive even if AQI refresh fails.
        }
      }
    },
    () => {},
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 3000 }
  );
}

// Map Setup
function ensureMap() {
  if (map) return;

  map = L.map("map", { zoomControl: false }).setView(BENGALURU_CENTER, 11);
  L.control.zoom({ position: "bottomleft" }).addTo(map);
  const detailedMap = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  const highContrastMap = L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
    maxZoom: 20,
    attribution: MAP_ATTRIBUTION,
  });
  const lightMap = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 20,
    attribution: MAP_ATTRIBUTION,
  });
  L.control.layers(
    {
      "Street labels": detailedMap,
      "Clear labels": highContrastMap,
      "Light mode": lightMap,
    },
    {},
    { position: "bottomleft", collapsed: true }
  ).addTo(map);

  markerLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);

  map.on("click", handleMapClick);
  map.on("dragstart zoomstart", () => { followUserLocation = false; });
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
    await selectForecastLocation(latlng, 24);
  } else if (activeTab === "forecast7DTab") {
    await selectForecastLocation(latlng, 168);
  } else if (activeTab === "compareTab") {
    await addCompareLocation(latlng);
  }
}

// Heatmap / Station loader
async function loadForecastMap(hour) {
  ensureMap();
  setStatus("Loading station AQI markers...");

  const points = await fetchJson(API.forecastHour(hour));
  markerLayer.clearLayers();

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
  heatLayer = null;

  setStatus(`${points.length} station AQI markers loaded for hour ${hour}.`);
}

// Routing Logic
function routePopup(title, route) {
  return `
    <strong>${title}</strong><br>
    ${route.distance_km} km - ${route.travel_time_minutes} min<br>
    Avg AQI ${route.average_aqi}
  `;
}

const ROUTE_META = {
  fastest: { label: "Fastest", color: "#3b82f6", bg: "rgba(59,130,246,0.10)", border: "rgba(59,130,246,0.30)" },
  balanced: { label: "Balanced", color: "#f59e0b", bg: "rgba(245,158,11,0.10)", border: "rgba(245,158,11,0.30)" },
  least_harm: { label: "Least Harm", color: "#8b5cf6", bg: "rgba(139,92,246,0.12)", border: "rgba(139,92,246,0.34)" },
  cleanest: { label: "Cleanest", color: "#10b981", bg: "rgba(16,185,129,0.10)", border: "rgba(16,185,129,0.28)" },
};

function routeLabel(key) {
  return ROUTE_META[key]?.label || key.replace(/_/g, " ");
}

function getSelectedRoute() {
  return currentRouteOptions[selectedRouteKey] || currentRouteOptions.least_harm || currentRouteOptions.fastest;
}

function drawRoute(route, color, title, { selected = false } = {}) {
  const routeGroup = L.layerGroup().addTo(routeLayer);
  const glowWeight = selected ? 20 : 10;
  const outlineWeight = selected ? 10 : 6;
  const coreWeight = selected ? 6 : 3;
  const coreOpacity = selected ? 1 : 0.42;
  // 1. Soft wide glow
  L.polyline(route.coordinates, {
    color: color,
    weight: glowWeight,
    opacity: selected ? 0.22 : 0.08,
    lineCap: 'round',
    lineJoin: 'round',
    className: 'premium-route-glow'
  }).addTo(routeGroup);

  // 2. Dark sharp outline to separate from map
  L.polyline(route.coordinates, {
    color: '#0a0d14',
    weight: outlineWeight,
    opacity: selected ? 0.85 : 0.38,
    lineCap: 'round',
    lineJoin: 'round'
  }).addTo(routeGroup);

  // 3. Bright vibrant core line
  const polyline = L.polyline(route.coordinates, {
    color: color,
    weight: coreWeight,
    opacity: coreOpacity,
    lineCap: 'round',
    lineJoin: 'round',
    className: 'premium-route-core'
  })
    .bindPopup(routePopup(title, route))
    .addTo(routeGroup);

  // Draw end-cap dots
  if (selected && route.coordinates.length > 0) {
    const start = route.coordinates[0];
    const end = route.coordinates[route.coordinates.length - 1];
    
    L.circleMarker(start, { radius: 5, color: '#0a0d14', weight: 2, fillColor: color, fillOpacity: 1 }).addTo(routeGroup);
    L.circleMarker(end, { radius: 5, color: '#0a0d14', weight: 2, fillColor: color, fillOpacity: 1 }).addTo(routeGroup);
  }

  return polyline;
}

function addNavigationMarkers(route, color) {
  const navigation = route.navigation || {};
  (navigation.signals || []).forEach((signal) => {
    L.circleMarker([signal.lat, signal.lon], {
      radius: 6,
      color: "#ffffff",
      weight: 2,
      fillColor: "#ef4444",
      fillOpacity: 0.95,
      className: "nav-signal-marker",
    }).bindPopup(`<strong>Signal ahead</strong><br>${metersLabel(signal.distance_m)} from start`).addTo(routeLayer);
  });

  (navigation.aqi_alerts || []).forEach((alert) => {
    L.circleMarker([alert.lat, alert.lon], {
      radius: 8,
      color: "#ffffff",
      weight: 2,
      fillColor: "#b91c1c",
      fillOpacity: 0.88,
      className: "nav-aqi-marker",
    }).bindPopup(`<strong>High AQI ahead</strong><br>AQI ${alert.aqi}<br>${metersLabel(alert.distance_m)} from start`).addTo(routeLayer);
  });

  // Keep navigation steps in the card, but do not draw waypoint dots over the route.
}

function renderNavigationCard(route) {
  const navigation = route.navigation || {};
  const traffic = navigation.traffic || { level: "Unknown", summary: "Traffic signal unavailable" };
  const nextStep = (navigation.steps || [])[0];
  const signal = (navigation.signals || [])[0];
  const aqiAlert = (navigation.aqi_alerts || [])[0];
  const stepRows = (navigation.steps || []).slice(0, 5).map((step) => `
    <div class="nav-step">
      <span>${metersLabel(step.distance_m)}</span>
      <b>${step.instruction}</b>
    </div>
  `).join("");

  return `
    <div class="navigation-card">
      <div class="nav-primary">
        <div>
          <span>${navigationActive ? "Navigating" : "Live Guidance"}</span>
          <strong>${nextStep ? `${nextStep.instruction} in ${metersLabel(nextStep.distance_m)}` : "Continue on route"}</strong>
        </div>
        <div class="traffic-pill ${traffic.level.toLowerCase()}">${traffic.level} traffic</div>
      </div>
      <div class="nav-alert-grid">
        <div><span>Traffic</span><b>${traffic.summary}</b></div>
        <div><span>AQI</span><b>${aqiAlert ? `${aqiAlert.message} in ${metersLabel(aqiAlert.distance_m)}` : "No major AQI hotspot on this route"}</b></div>
        <div><span>Signal</span><b>${signal ? `Signal in ${metersLabel(signal.distance_m)}` : "No mapped signal on route"}</b></div>
      </div>
      ${stepRows ? `<div class="nav-steps">${stepRows}</div>` : ""}
      <button class="start-navigation-btn${navigationActive ? " active" : ""}" data-start-navigation="true">
        ${navigationActive ? "Navigation Started" : `Start ${routeLabel(selectedRouteKey)}`}
      </button>
    </div>
  `;
}

function routeCard(key, route) {
  const meta = ROUTE_META[key];
  const active = key === selectedRouteKey ? " active" : "";
  const signalCount = route.quality?.signal_count ?? 0;
  const stressScore = route.quality?.stress_score ?? 0;
  const subline = key === "least_harm"
    ? `Stress ${stressScore} · ${signalCount} signals`
    : `Dose ${route.dose_index}`;
  return `
    <button class="route-choice ${key}${active}" data-route-key="${key}" style="--route-color:${meta.color};--route-bg:${meta.bg};--route-border:${meta.border}">
      <span>${meta.label}</span>
      <strong>${route.travel_time_minutes} min</strong>
      <small>${route.distance_km} km · AQI ${route.average_aqi}</small>
      <em>${subline}</em>
    </button>
  `;
}

function renderTradeoffRows(tradeoffCurve = []) {
  return tradeoffCurve.map(item => `
    <button class="tradeoff-row ${item.route === selectedRouteKey ? "active" : ""}" data-route-key="${item.route}">
      <span>${routeLabel(item.route)}</span>
      <small>${item.travel_time_minutes} min · dose ${item.dose_index} · stress ${item.stress_score ?? 0}</small>
      <b>${item.dose_reduction_percent_vs_fastest.toFixed(1)}%</b>
    </button>
  `).join("");
}

function renderResearchPanel(route) {
  const uncertainty = route.uncertainty || {};
  const signalForecast = route.signal_forecast || {};
  const personalizedDose = route.personalized_dose || {};
  const counterfactuals = route.counterfactuals || [];
  const signalRows = (signalForecast.top_signals || []).map((item) => `
    <div class="research-row">
      <span>${metersLabel(item.distance_m)}</span>
      <b>${item.expected_wait_seconds}s wait · AQI ${item.aqi}</b>
    </div>
  `).join("");

  return `
    <div class="research-panel">
      <div class="research-card">
        <span>Uncertainty-Aware Routing</span>
        <strong>${uncertainty.confidence || "Unknown"} confidence</strong>
        <small>Mean segment +/- ${uncertainty.mean_segment_uncertainty ?? 0} AQI · Peak +/- ${uncertainty.peak_segment_uncertainty ?? 0}</small>
        <p>${uncertainty.confidence_reason || "No uncertainty summary available yet."}</p>
      </div>
      <div class="research-card">
        <span>Personalized Inhaled Dose</span>
        <strong>${personalizedDose.strain_band || "Unknown"} strain band</strong>
        <small>${personalizedDose.dose_per_km ?? 0} dose/km · sensitivity x${personalizedDose.sensitivity_multiplier ?? 1}</small>
        <p>${personalizedDose.summary || "Dose interpretation unavailable."}</p>
      </div>
      <div class="research-card">
        <span>Signal Exposure Forecast</span>
        <strong>${signalForecast.count ?? 0} mapped signals</strong>
        <small>${signalForecast.total_expected_wait_seconds ?? 0}s expected waiting · dose ${signalForecast.expected_dose_at_signals ?? 0}</small>
        <p>${signalForecast.summary || "No signal forecast available."}</p>
        ${signalRows ? `<div class="research-list">${signalRows}</div>` : ""}
      </div>
      <div class="research-card">
        <span>Counterfactual Interventions</span>
        <strong>What to change</strong>
        <div class="research-list">
          ${(counterfactuals.length ? counterfactuals : ["No intervention suggestions available."]).map((item) => `
            <div class="research-row">
              <span>Step</span>
              <b>${item}</b>
            </div>
          `).join("")}
        </div>
      </div>
    </div>
  `;
}

function renderRouteSummary(resultData, selectedRoute, bestTimeHtml = "") {
  const fastest = resultData.fastest;
  const balanced = resultData.balanced;
  const cleanest = resultData.cleanest;
  const leastHarm = resultData.least_harm;
  const savingsUg = Math.max(0, (fastest.dose_index - selectedRoute.dose_index) * 1000);

  routeSummary.style.display = "block";
  routeSummary.innerHTML = `
    <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:10px">Route Analysis</div>
    ${renderNavigationCard(selectedRoute)}
    <div class="route-choice-grid">
      ${routeCard("fastest", fastest)}
      ${routeCard("balanced", balanced)}
      ${routeCard("least_harm", leastHarm)}
      ${routeCard("cleanest", cleanest)}
    </div>
    <div style="margin-top:12px">
      <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:4px">Time-Exposure Trade-off</div>
      <div class="tradeoff-list">${renderTradeoffRows(resultData.tradeoff_curve || [])}</div>
    </div>
    <div style="margin-top:12px">
      <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:6px">Research Features</div>
      ${renderResearchPanel(selectedRoute)}
    </div>
    <button onclick="saveToWallet(${savingsUg})" class="secondary-btn" style="margin-top:10px;width:100%">Save Selected Route to Health Wallet</button>
    ${bestTimeHtml}
  `;

}

function renderSelectedRoute(bestTimeHtml = "") {
  const selectedRoute = getSelectedRoute();
  if (!selectedRoute) return;
  routeLayer.clearLayers();

  const routeKeys = navigationActive
    ? [selectedRouteKey]
    : ["fastest", "balanced", "cleanest", "least_harm"]
        .filter((key) => key !== selectedRouteKey)
        .concat(selectedRouteKey);

  routeKeys.forEach((key) => {
    const route = currentRouteOptions[key];
    if (!route) return;
    const meta = ROUTE_META[key];
    drawRoute(route, meta.color, `${meta.label} route`, { selected: key === selectedRouteKey });
  });

  const selectedMeta = ROUTE_META[selectedRouteKey] || ROUTE_META.least_harm;
  addNavigationMarkers(selectedRoute, selectedMeta.color);
  renderRouteSummary(currentRouteOptions, selectedRoute, bestTimeHtml || currentRouteOptions.bestTimeHtml || "");
}

function selectRouteOption(key) {
  if (!currentRouteOptions[key]) return;
  selectedRouteKey = key;
  navigationActive = false;
  renderSelectedRoute();
  const selectedRoute = getSelectedRoute();
  if (selectedRoute?.coordinates?.length) {
    map.fitBounds(L.latLngBounds(selectedRoute.coordinates), { padding: [42, 42] });
  }
  setStatus(`${routeLabel(key)} route selected. Press Start to begin navigation.`);
}

function startSelectedRouteNavigation(event) {
  if (event) event.preventDefault();
  const route = getSelectedRoute();
  if (!route?.coordinates?.length) {
    setStatus("Select a route before starting navigation.", true);
    return;
  }

  navigationActive = true;
  followUserLocation = true;
  startLocationWatch();
  requestCurrentLocation({ setOrigin: false, quiet: true });
  renderSelectedRoute();
  map.fitBounds(L.latLngBounds(route.coordinates), { padding: [28, 28] });
  setStatus(`Navigation started on the ${routeLabel(selectedRouteKey)} route.`);
}

// findBestTravelTime removed; optimal route departure is now handled by the backend

async function loadRouteInsight(payload) {
  try {
    xaiInsightEl.style.display = "block";
    xaiTextEl.innerHTML = '<div style="color:var(--text-secondary)">Preparing route insight...</div>';
    const xaiResult = await fetchJson(API.xaiInsight, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const hotspotHtml = (xaiResult.hotspots || []).slice(0, 2).map(item => `â€¢ ${item}`).join('<br>');
    const tipsHtml = (xaiResult.tips || []).slice(0, 2).map(item => `â€¢ ${item}`).join('<br>');
    xaiTextEl.innerHTML = `
      <div style="color:var(--text-primary);line-height:1.6">${xaiResult.summary || xaiResult.insight}</div>
      ${hotspotHtml ? `<div style="margin-top:8px;line-height:1.5">${hotspotHtml}</div>` : ''}
      ${tipsHtml ? `<div style="margin-top:8px;color:var(--text-primary);line-height:1.5">${tipsHtml}</div>` : ''}
    `;
    currentHighlights = xaiResult.highlights || [];
  } catch (_) {
    xaiInsightEl.style.display = "none";
  }
}

async function findRoutes() {
  navigationActive = false;
  routeLayer.clearLayers();
  routeSummary.style.display = 'none';
  routeSummary.innerHTML = '';
  setStatus("Finding routes...");
  progressStart();

  const routeBtn = document.getElementById('routeBtn');
  routeBtn.disabled = true;
  routeBtn.innerHTML = '<span class="spinner"></span> Finding Routes...';

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
        endpoint = API.routeTsp;
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
      currentRouteOptions = { cleanest: result.data.cleanest };
      selectedRouteKey = "cleanest";
      drawRoute(result.data.cleanest, '#3d7a5c', 'Cleanest TSP route');
      addNavigationMarkers(result.data.cleanest, '#3d7a5c');
      const cleanest = result.data.cleanest;
      routeSummary.style.display = 'block';
      routeSummary.innerHTML = `
        <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:10px">Multi-Stop Route Analysis</div>
        <div style="padding:10px;background:rgba(61,122,92,0.1);border:1px solid rgba(61,122,92,0.2);border-radius:4px">
          <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:#3d7a5c;margin-bottom:4px">Optimal Sequence</div>
          <div style="font-size:0.9rem;font-weight:600">${cleanest.travel_time_minutes} min</div>
          <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">${cleanest.distance_km} km &middot; AQI ${cleanest.average_aqi}</div>
        </div>
        ${renderNavigationCard(cleanest)}
      `;
      map.fitBounds(routeLayer.getBounds(), { padding: [40, 40] });
    } else {
      const optimal = result.data.optimal_departure;
      const fastest = result.data.fastest;
      const balanced = result.data.balanced;
      const cleanest = result.data.cleanest;
      const leastHarm = result.data.least_harm;
      const savingsUg = Math.max(0, (fastest.dose_index - leastHarm.dose_index) * 1000);

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
      const tradeoffRows = (result.data.tradeoff_curve || []).map(item => `
        <div style="display:flex;justify-content:space-between;gap:8px;padding:6px 0;border-top:1px solid var(--border);font-size:0.72rem">
          <span style="text-transform:capitalize;color:var(--text-primary)">${item.route}</span>
          <span style="color:var(--text-secondary)">${item.travel_time_minutes} min · dose ${item.dose_index}</span>
          <span style="color:${item.dose_reduction_percent_vs_fastest >= 0 ? '#10b981' : '#c97777'}">${item.dose_reduction_percent_vs_fastest.toFixed(1)}%</span>
        </div>
      `).join('');
      routeSummary.innerHTML = `
        <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:10px">Route Analysis</div>
        ${renderNavigationCard(leastHarm)}
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <div style="padding:10px;background:rgba(74,111,165,0.1);border:1px solid rgba(74,111,165,0.2);border-radius:4px">
            <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a6fa5;margin-bottom:4px">Fastest</div>
            <div style="font-size:0.9rem;font-weight:600">${fastest.travel_time_minutes} min</div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">${fastest.distance_km} km &middot; AQI ${fastest.average_aqi}</div>
            <div style="font-size:0.68rem;color:var(--text-muted);margin-top:2px">Dose ${fastest.dose_index}</div>
          </div>
          <div style="padding:10px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.22);border-radius:4px">
            <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:#f59e0b;margin-bottom:4px">Balanced</div>
            <div style="font-size:0.9rem;font-weight:600">${balanced.travel_time_minutes} min</div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">${balanced.distance_km} km &middot; AQI ${balanced.average_aqi}</div>
            <div style="font-size:0.68rem;color:var(--text-muted);margin-top:2px">Dose ${balanced.dose_index}</div>
          </div>
          <div style="padding:10px;background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.28);border-radius:4px">
            <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:#a78bfa;margin-bottom:4px">Least Harm</div>
            <div style="font-size:0.9rem;font-weight:600">${leastHarm.travel_time_minutes} min</div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">${leastHarm.distance_km} km &middot; AQI ${leastHarm.average_aqi}</div>
            <div style="font-size:0.68rem;color:var(--text-muted);margin-top:2px">Stress ${leastHarm.quality?.stress_score ?? 0} &middot; ${leastHarm.quality?.signal_count ?? 0} signals</div>
          </div>
          <div style="padding:10px;background:rgba(61,122,92,0.1);border:1px solid rgba(61,122,92,0.2);border-radius:4px">
            <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;color:#3d7a5c;margin-bottom:4px">Cleanest</div>
            <div style="font-size:0.9rem;font-weight:600">${cleanest.travel_time_minutes} min</div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:2px">${cleanest.distance_km} km &middot; AQI ${cleanest.average_aqi}</div>
            <div style="font-size:0.68rem;color:var(--text-muted);margin-top:2px">Dose ${cleanest.dose_index}</div>
          </div>
        </div>
        <div style="margin-top:12px">
          <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:4px">Time-Exposure Trade-off</div>
          ${tradeoffRows}
        </div>
        <button onclick="saveToWallet(${savingsUg})" class="secondary-btn" style="margin-top:10px;width:100%">Save Route to Health Wallet</button>
        ${bestTimeHtml}
      `;
      currentRouteOptions = {
        ...result.data,
        bestTimeHtml,
      };
      selectedRouteKey = result.data.least_harm ? "least_harm" : "fastest";
      renderSelectedRoute(bestTimeHtml);
      const selectedRoute = currentRouteOptions[selectedRouteKey];
      if (selectedRoute?.coordinates?.length) {
        map.fitBounds(L.latLngBounds(selectedRoute.coordinates), { padding: [40, 40] });
      }
    }

    setStatus("Routes ready âœ“");
    if (!isTSP) {
      loadRouteInsight(payload);
    }
    return;

    try {
      const xaiResult = await fetchJson("/api/xai/route_insight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      xaiInsightEl.style.display = "block";
      const hotspotHtml = (xaiResult.hotspots || []).slice(0, 2).map(item => `• ${item}`).join('<br>');
      const tipsHtml = (xaiResult.tips || []).slice(0, 2).map(item => `• ${item}`).join('<br>');
      xaiTextEl.innerHTML = `
        <div style="color:var(--text-primary);line-height:1.6">${xaiResult.summary || xaiResult.insight}</div>
        ${hotspotHtml ? `<div style="margin-top:8px;line-height:1.5">${hotspotHtml}</div>` : ''}
        ${tipsHtml ? `<div style="margin-top:8px;color:var(--text-primary);line-height:1.5">${tipsHtml}</div>` : ''}
      `;
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
async function fetchGeocode(query, listEl, inputEl, onSelect) {
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
            const latlng = { lat: Number(item.lat), lng: Number(item.lon) };
            inputEl.value = onSelect ? item.display_name : `${item.lat}, ${item.lon}`;
            listEl.style.display = 'none';
            if (onSelect) {
              onSelect(latlng, item);
            } else if (inputEl.id === 'originInput') {
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
if (forecast24Input && forecast24Suggestions) {
  forecast24Input.addEventListener('input', (e) => fetchGeocode(
    e.target.value,
    forecast24Suggestions,
    forecast24Input,
    (latlng) => selectForecastLocation(latlng, 24).catch((error) => setStatus(error.message, true))
  ));
}
if (forecast7DInput && forecast7DSuggestions) {
  forecast7DInput.addEventListener('input', (e) => fetchGeocode(
    e.target.value,
    forecast7DSuggestions,
    forecast7DInput,
    (latlng) => selectForecastLocation(latlng, 168).catch((error) => setStatus(error.message, true))
  ));
}

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
    syncLocalAqiCardFromForecast(data);
    
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
    const uncertainty = data.current?.uncertainty ?? data.hourly_uncertainty?.[0];
    document.getElementById(detailsId).innerHTML = `Avg AQI: ${(aqiData.reduce((a,b)=>a+b,0)/aqiData.length).toFixed(1)} | Max: ${Math.max(...aqiData).toFixed(1)}${uncertainty ? ` | Now: ${data.current.aqi.toFixed(1)} ± ${uncertainty.toFixed(1)}` : ''}`;
    const explanation = data.explanation || {};
    const reasons = Array.isArray(explanation.reasons) ? explanation.reasons : [];
    const tips = Array.isArray(explanation.tips) ? explanation.tips : [];
    const road = data.road_context || {};
    document.getElementById(detailsId).innerHTML = `
      <div>Avg AQI: ${(aqiData.reduce((a,b)=>a+b,0)/aqiData.length).toFixed(1)} | Max: ${Math.max(...aqiData).toFixed(1)}${uncertainty ? ` | Now: ${data.current.aqi.toFixed(1)} Â± ${uncertainty.toFixed(1)}` : ''}</div>
      <div style="margin-top:8px;font-size:0.75rem;color:var(--text-secondary);line-height:1.5;">${explanation.summary || ''}</div>
      ${reasons.length ? `<div style="margin-top:8px;font-size:0.72rem;color:var(--text-secondary);line-height:1.45">${reasons.slice(0, 2).map(item => `• ${item}`).join('<br>')}</div>` : ''}
      ${tips.length ? `<div style="margin-top:8px;font-size:0.72rem;color:var(--text-primary);line-height:1.45">${tips.slice(0, 2).map(item => `• ${item}`).join('<br>')}</div>` : ''}
      <div style="margin-top:8px;font-size:0.7rem;color:var(--text-muted);line-height:1.45;">Major-road share ${((road.major_road_share || 0) * 100).toFixed(0)}%${road.nearest_major_road_m != null ? ` | nearest major road ${road.nearest_major_road_m.toFixed(0)} m` : ''}${data.current?.nearest_station_km != null ? ` | nearest station ${data.current.nearest_station_km.toFixed(2)} km` : ''}</div>
    `;
    document.getElementById(detailsId).innerHTML = document.getElementById(detailsId).innerHTML
      .replaceAll("Ã‚Â±", "+/-")
      .replaceAll("Â±", "+/-")
      .replaceAll("â€¢", "-")
      .replaceAll("â€”", "-");
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

function getWalletEntries() {
  try {
    return JSON.parse(localStorage.getItem("bluruExposureWallet") || "[]");
  } catch (_) {
    return [];
  }
}

function renderWallet() {
  const totalEl = document.getElementById("walletTotal");
  const historyEl = document.getElementById("walletHistory");
  if (!totalEl || !historyEl) return;

  const entries = getWalletEntries();
  const total = entries.reduce((sum, entry) => sum + Number(entry.savedDose || 0), 0);
  totalEl.textContent = `${total.toFixed(0)} dose pts`;

  historyEl.innerHTML = entries.length
    ? entries.slice(-8).reverse().map(entry => `
      <div style="display:flex;justify-content:space-between;gap:10px;padding:9px 10px;background:var(--surface);border:1px solid var(--border);border-radius:4px;font-size:0.75rem">
        <span style="color:var(--text-primary)">${entry.label}</span>
        <span style="color:#10b981">-${Number(entry.savedDose).toFixed(0)}</span>
      </div>
    `).join('')
    : '<div style="font-size:0.75rem;color:var(--text-secondary);text-align:center;padding:12px">No saved routes yet.</div>';
}

function saveToWallet(savedDose) {
  const entries = getWalletEntries();
  entries.push({
    savedDose: Math.max(0, Number(savedDose) || 0),
    label: new Date().toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
  });
  localStorage.setItem("bluruExposureWallet", JSON.stringify(entries.slice(-30)));
  renderWallet();
  setStatus("Exposure savings added to wallet.");
}

window.saveToWallet = saveToWallet;

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
    if (activeTab === "walletTab") renderWallet();
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
    startLocationWatch();
    requestCurrentLocation({ setOrigin: true, quiet: true });
    renderWallet();
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

routeSummary.addEventListener("click", (event) => {
  const routeChoice = event.target.closest("[data-route-key]");
  if (routeChoice) {
    selectRouteOption(routeChoice.dataset.routeKey);
    return;
  }

  if (event.target.closest("[data-start-navigation]")) {
    startSelectedRouteNavigation(event);
  }
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
      const rawWeights = await fetchJson(API.xaiAttention);
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
            "site_5686": [13.0450, 77.5116],
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

function getHealthTips(aqi, profile) {
  let category = "Good";
  let color = "#3d7a5c"; // green
  if (aqi > 50 && aqi <= 100) { category = "Moderate"; color = "#b8922a"; } // gold
  else if (aqi > 100 && aqi <= 150) { category = "Unhealthy for Sensitive Groups"; color = "#ff7e00"; } // orange
  else if (aqi > 150 && aqi <= 200) { category = "Unhealthy"; color = "#8b3a3a"; } // red
  else if (aqi > 200) { category = "Very Unhealthy/Hazardous"; color = "#7e0023"; } // dark red

  let tip = "";
  if (profile === "asthmatic" || profile === "copd") {
    if (aqi <= 50) tip = "Air quality is good. Enjoy outdoor activities, but keep your inhaler handy.";
    else if (aqi <= 100) tip = "Air quality is acceptable. Consider limiting prolonged outdoor exertion if symptoms appear.";
    else if (aqi <= 150) tip = "Unhealthy for you. Reduce heavy outdoor exertion. Keep windows closed.";
    else tip = "Poor air quality. Avoid outdoor activities. Use air purifiers and keep medications nearby.";
  } else if (profile === "elderly" || profile === "children") {
    if (aqi <= 50) tip = "Great air quality! Perfect time for outdoor activities.";
    else if (aqi <= 100) tip = "Moderate air quality. Limit heavy exertion if sensitive.";
    else if (aqi <= 150) tip = "Unhealthy for sensitive groups. Reduce prolonged outdoor activities.";
    else tip = "Unhealthy air. Stay indoors, keep windows closed, and avoid physical exertion outside.";
  } else {
    // healthy
    if (aqi <= 50) tip = "Air quality is excellent. Ideal conditions for outdoor workouts!";
    else if (aqi <= 100) tip = "Air quality is acceptable. No major health risk for the general public.";
    else if (aqi <= 150) tip = "Air quality is poor for sensitive groups, but generally fine for healthy adults. You may want to reduce very heavy exertion.";
    else tip = "Air quality is poor. Consider moving workouts indoors and limiting prolonged exposure.";
  }

  return { category, tip, color };
}

if (myLocationBtn) {
  myLocationBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopImmediatePropagation();
    followUserLocation = true;
    startLocationWatch();
    requestCurrentLocation({ setOrigin: true });
  }, true);
}

boot();
