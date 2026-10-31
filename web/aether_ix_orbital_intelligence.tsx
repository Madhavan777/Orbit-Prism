import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import {
  Satellite,
  Radio,
  Globe,
  Compass,
  Activity,
  ShieldCheck,
  Cpu,
  Play,
  RotateCcw,
  FastForward,
  Eye,
  Layers,
  Crosshair,
  Sliders,
  ChevronRight,
  Sparkles,
  Wifi,
  Terminal,
  Navigation,
  Maximize2,
  Menu,
  X,
  Info,
  Zap,
  Lock,
  CloudRain,
  AlertTriangle,
  ArrowUp,
  CheckCircle2,
  BarChart3,
  Radar
} from 'lucide-react';

// ============================================================================
// COLOR SYSTEM & CONSTANTS
// ============================================================================
const COLORS = {
  midnightBlack: '#05070D',
  deepNavy: '#0A1020',
  cobaltBlue: '#245BFF',
  acidLime: '#C6FF00',
  white: '#F4F7FF',
  pureWhite: '#FFFFFF',
  mutedGrey: '#8B95A7',
};

const INDIA_COORDS = {
  lat: 20.5937,
  lon: 78.9629,
  alt: 408,
};

// 5-Step Intro Sequence
const INTRO_STEPS = [
  { id: 1, code: 'INTRO 01', name: 'SATELLITE REVEAL', range: [0.0, 0.20] },
  { id: 2, code: 'INTRO 02', name: 'ANTENNA GIMBAL LOCK', range: [0.20, 0.40] },
  { id: 3, code: 'INTRO 03', name: 'WHITE OPTICAL LASER LINK', range: [0.40, 0.62] },
  { id: 4, code: 'INTRO 04', name: 'SUBCONTINENT SCAN', range: [0.62, 0.82] },
  { id: 5, code: 'INTRO 05', name: 'ORBITAL HANDOFF', range: [0.82, 1.0] },
];

// 8-Section Continuous Scroll Mission Architecture
const MISSION_SECTIONS = [
  { id: 'sec-hero', index: 0, code: '01', label: 'HERO', title: 'ORBITAL INTELLIGENCE' },
  { id: 'sec-mission', index: 1, code: '02', label: 'MISSION', title: 'ORBIT-PRISM ARCHITECTURE' },
  { id: 'sec-observation', index: 2, code: '03', label: 'OBSERVATION', title: 'SUBCONTINENT SWATH' },
  { id: 'sec-flood', index: 3, code: '04', label: 'FLOOD CHIP', title: '512×512 FLOOD GRID' },
  { id: 'sec-sensors', index: 4, code: '05', label: 'SAR / OPTICAL', title: 'DUAL-SENSOR FUSION' },
  { id: 'sec-downlink', index: 5, code: '06', label: 'DOWNLINK', title: 'L0 / L1 / L2 TELEMETRY' },
  { id: 'sec-calibration', index: 6, code: '07', label: 'CALIBRATION', title: 'CONFORMAL RISK CONTROL' },
  { id: 'sec-footer', index: 7, code: '08', label: 'HORIZON', title: 'MISSION CONTINUITY' },
];

// 4x4 Tile Grid for Section 4 (512x512 Satellite Image Chip)
const FLOOD_TILES = Array.from({ length: 16 }, (_, idx) => {
  const row = Math.floor(idx / 4);
  const col = idx % 4;
  // Critical flood tiles along the Brahmaputra / Ganga delta channel
  const isCritical = [5, 6, 9, 10].includes(idx);
  const isModerate = [1, 2, 8, 11, 14].includes(idx);
  const waterPct = isCritical
    ? 78 + ((idx * 7) % 18)
    : isModerate
    ? 31 + ((idx * 5) % 19)
    : 4 + ((idx * 3) % 11);
  return {
    id: idx,
    coord: `T-${row}${col}`,
    pxBounds: `${col * 128}..${(col + 1) * 128}px`,
    isCritical,
    isModerate,
    waterPct,
    priority: isCritical ? 'L2 CRITICAL' : isModerate ? 'L1 BUFFER' : 'L0 ARCHIVE',
    confidence: isCritical ? (0.96 + (idx % 3) * 0.012).toFixed(3) : (0.88 + (idx % 4) * 0.02).toFixed(3),
  };
});

// Calibration Miss-Rate Targets for Section 7
const CALIBRATION_TARGETS = {
  '1%': {
    alpha: '0.010',
    empiricalMiss: '0.84%',
    thresholdTau: '0.214',
    l2TilesFlagged: '6 / 16 Tiles',
    bandwidthSaved: '62.5%',
    desc: 'Ultra-conservative emergency disaster posture. Guarantees at least 99% flood pixel recall across monsoon cloud cover.',
    curvePoints: [99.2, 99.1, 99.3, 99.16, 99.24, 99.18],
  },
  '5%': {
    alpha: '0.050',
    empiricalMiss: '4.62%',
    thresholdTau: '0.382',
    l2TilesFlagged: '4 / 16 Tiles',
    bandwidthSaved: '75.0%',
    desc: 'Balanced operational mission profile. Prioritizes the 4 highest-severity riverine surge tiles for sub-second L2 laser downlink.',
    curvePoints: [95.8, 95.3, 95.5, 95.4, 95.38, 95.42],
  },
  '10%': {
    alpha: '0.100',
    empiricalMiss: '9.18%',
    thresholdTau: '0.519',
    l2TilesFlagged: '2 / 16 Tiles',
    bandwidthSaved: '87.5%',
    desc: 'High-compression bandwidth conservation mode for low-elevation ground passes with constrained optical link budgets.',
    curvePoints: [91.1, 90.7, 90.9, 90.82, 90.85, 90.82],
  },
};

// ============================================================================
// MATHEMATICAL & PROCEDURAL TEXTURE HELPERS
// ============================================================================
function latLonToVector3(lat, lon, radius) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);
  return new THREE.Vector3(x, y, z);
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Catmull-Rom 1D scalar interpolation across keyframe array
function sampleScalarSpline(values, t) {
  const n = values.length - 1;
  const scaled = clamp(t, 0, 1) * n;
  const idx = Math.min(Math.floor(scaled), n - 1);
  const frac = scaled - idx;

  const p0 = values[Math.max(0, idx - 1)];
  const p1 = values[idx];
  const p2 = values[Math.min(n, idx + 1)];
  const p3 = values[Math.min(n, idx + 2)];

  const f2 = frac * frac;
  const f3 = f2 * frac;
  return (
    0.5 *
    (2 * p1 +
      (-p0 + p2) * frac +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * f3)
  );
}

// Procedural Kapton Gold Foil Texture
function createKaptonTextures() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  const grad = ctx.createLinearGradient(0, 0, 512, 512);
  grad.addColorStop(0, '#D99B26');
  grad.addColorStop(0.5, '#F5BD41');
  grad.addColorStop(1, '#B37412');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 512);

  for (let i = 0; i < 850; i++) {
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const size = 8 + Math.random() * 24;
    ctx.fillStyle =
      Math.random() > 0.5
        ? 'rgba(255, 235, 160, 0.16)'
        : 'rgba(90, 48, 4, 0.18)';
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + size * (Math.random() - 0.3), y + size * (Math.random() - 0.3));
    ctx.lineTo(x + size * (Math.random() - 0.3), y + size * (Math.random() - 0.3));
    ctx.closePath();
    ctx.fill();
  }

  ctx.strokeStyle = 'rgba(70, 38, 5, 0.45)';
  ctx.lineWidth = 2;
  for (let i = 0; i <= 512; i += 64) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }

  const map = new THREE.CanvasTexture(canvas);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(2, 2);
  return map;
}

// Procedural Solar Panel Array Texture
function createSolarPanelTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#05112E';
  ctx.fillRect(0, 0, 512, 1024);

  const cols = 8;
  const rows = 24;
  const pad = 4;
  const cellW = (512 - pad * (cols + 1)) / cols;
  const cellH = (1024 - pad * (rows + 1)) / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = pad + c * (cellW + pad);
      const y = pad + r * (cellH + pad);
      const g = ctx.createLinearGradient(x, y, x + cellW, y + cellH);
      g.addColorStop(0, '#0E2D73');
      g.addColorStop(0.5, '#1D52CC');
      g.addColorStop(1, '#0B2259');
      ctx.fillStyle = g;
      ctx.fillRect(x, y, cellW, cellH);

      ctx.strokeStyle = 'rgba(180, 215, 255, 0.35)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + cellW * 0.25, y);
      ctx.lineTo(x + cellW * 0.25, y + cellH);
      ctx.moveTo(x + cellW * 0.75, y);
      ctx.lineTo(x + cellW * 0.75, y + cellH);
      ctx.stroke();

      ctx.fillStyle = '#245BFF';
      ctx.fillRect(x, y, 3, 3);
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 4;
  return tex;
}

// Procedural High-Contrast Earth Day, Night & Cloud Textures
function createEarthTextures() {
  const width = 2048;
  const height = 1024;

  const toXY = (lat, lon) => {
    const x = ((lon + 180) / 360) * width;
    const y = ((90 - lat) / 180) * height;
    return [x, y];
  };

  const indiaPolygon = [
    [35.5, 74.0], [34.5, 77.8], [32.0, 78.5], [30.0, 81.0],
    [28.2, 88.2], [27.8, 92.0], [28.0, 97.0], [24.0, 94.5],
    [22.0, 92.5], [22.2, 89.0], [21.5, 87.0], [19.8, 85.8],
    [16.5, 82.2], [13.1, 80.3], [10.3, 79.8], [8.08, 77.55],
    [9.9, 76.2], [12.9, 74.8], [15.4, 73.8], [19.0, 72.8],
    [20.7, 70.9], [22.3, 69.0], [23.8, 68.4], [24.5, 71.0],
    [27.5, 70.5], [30.5, 73.8], [33.5, 73.5],
  ];

  const sriLankaPolygon = [
    [9.8, 80.2], [8.5, 81.8], [6.8, 81.8], [5.9, 80.5], [7.2, 79.8], [9.0, 79.9],
  ];

  const arabiaPolygon = [
    [30.0, 32.5], [29.5, 48.0], [26.2, 56.3], [22.5, 59.8],
    [17.0, 54.0], [12.8, 45.0], [12.6, 43.3], [20.0, 39.5], [27.8, 34.5],
  ];

  const africaPolygon = [
    [37.0, 10.0], [32.0, 32.0], [12.0, 44.0], [11.8, 51.3],
    [2.0, 45.0], [-10.0, 40.5], [-26.0, 33.0], [-34.8, 20.0],
    [-30.0, 17.0], [-17.0, 11.8], [-5.0, 12.0], [4.0, 9.5],
    [5.0, -5.0], [15.0, -17.5], [25.0, -15.0], [35.8, -5.8],
  ];

  const eurasiaMainPolygon = [
    [36.0, -9.0], [43.0, -9.0], [48.0, -4.5], [54.0, 8.5],
    [60.0, 5.0], [70.0, 25.0], [70.0, 60.0], [73.0, 100.0],
    [70.0, 140.0], [65.0, 170.0], [55.0, 160.0], [50.0, 140.0],
    [40.0, 125.0], [35.0, 120.0], [23.0, 114.0], [20.0, 106.0],
    [10.0, 105.0], [1.3, 103.8], [7.0, 99.0], [16.0, 97.5],
    [22.0, 92.5], [28.0, 97.0], [35.5, 74.0], [25.2, 62.0],
    [26.5, 56.5], [30.0, 48.5], [32.0, 35.0], [36.5, 36.0],
    [38.0, 26.0], [40.0, 20.0], [41.0, 14.0], [44.0, 8.0], [36.0, -5.5],
  ];

  const australiaPolygon = [
    [-12.0, 131.0], [-11.0, 142.5], [-18.0, 146.0], [-28.0, 153.5],
    [-38.0, 147.0], [-35.0, 136.0], [-32.0, 125.0], [-34.0, 115.0],
    [-22.0, 114.0], [-15.0, 124.0],
  ];

  const northAmericaPolygon = [
    [70.0, -160.0], [72.0, -120.0], [65.0, -80.0], [50.0, -55.0],
    [42.0, -70.0], [30.0, -81.0], [25.0, -80.0], [30.0, -90.0],
    [20.0, -105.0], [8.5, -79.5], [16.0, -95.0], [32.0, -117.0],
    [48.0, -124.0], [60.0, -145.0],
  ];

  const southAmericaPolygon = [
    [11.0, -75.0], [8.0, -60.0], [0.0, -50.0], [-7.0, -35.0],
    [-23.0, -43.0], [-35.0, -54.0], [-54.0, -68.0], [-45.0, -75.0],
    [-18.0, -71.0], [-5.0, -81.0], [5.0, -77.0],
  ];

  const indonesiaIslands = [
    [[5.5, 95.3], [-5.8, 105.8], [-3.0, 104.5], [3.5, 98.5]],
    [[-6.0, 106.0], [-7.8, 114.0], [-8.5, 114.0], [-6.8, 106.0]],
    [[7.0, 116.5], [1.0, 118.5], [-3.5, 116.0], [-3.0, 110.0], [2.0, 109.5]],
  ];

  const drawPolygon = (
    ctx,
    poly,
    fillStyle,
    strokeStyle,
    lineWidth = 2,
    shadowBlur = 0,
    shadowColor = 'transparent'
  ) => {
    ctx.save();
    ctx.beginPath();
    poly.forEach(([lat, lon], idx) => {
      const [x, y] = toXY(lat, lon);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    if (shadowBlur > 0) {
      ctx.shadowBlur = shadowBlur;
      ctx.shadowColor = shadowColor;
    }
    if (fillStyle) {
      ctx.fillStyle = fillStyle;
      ctx.fill();
    }
    if (strokeStyle) {
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
    ctx.restore();
  };

  // 1. DAY MAP
  const dayCanvas = document.createElement('canvas');
  dayCanvas.width = width;
  dayCanvas.height = height;
  const dCtx = dayCanvas.getContext('2d');

  const oceanGrad = dCtx.createLinearGradient(0, 0, 0, height);
  oceanGrad.addColorStop(0, '#030814');
  oceanGrad.addColorStop(0.3, '#061534');
  oceanGrad.addColorStop(0.5, '#092456');
  oceanGrad.addColorStop(0.7, '#061534');
  oceanGrad.addColorStop(1, '#030814');
  dCtx.fillStyle = oceanGrad;
  dCtx.fillRect(0, 0, width, height);

  const allLandPolys = [
    eurasiaMainPolygon,
    indiaPolygon,
    sriLankaPolygon,
    arabiaPolygon,
    africaPolygon,
    australiaPolygon,
    northAmericaPolygon,
    southAmericaPolygon,
    ...indonesiaIslands,
  ];

  allLandPolys.forEach((poly) => {
    drawPolygon(dCtx, poly, '#103468', '#1C5BB8', 8, 22, '#245BFF');
  });

  allLandPolys.forEach((poly) => {
    drawPolygon(dCtx, poly, '#17292B', '#284A46', 2, 0);
  });

  // Topographic detail
  dCtx.save();
  for (let i = 0; i < 2000; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    const r = 4 + Math.random() * 22;
    const g = dCtx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, 'rgba(65, 110, 85, 0.09)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    dCtx.fillStyle = g;
    dCtx.beginPath();
    dCtx.arc(x, y, r, 0, Math.PI * 2);
    dCtx.fill();
  }
  dCtx.restore();

  // Prominent India highlight with Electric Cobalt & Acid Lime riverine hydrology
  drawPolygon(dCtx, indiaPolygon, '#1E3A3E', '#245BFF', 4, 18, '#245BFF');
  drawPolygon(dCtx, sriLankaPolygon, '#1E3A3E', '#245BFF', 2.5);

  // Ganga-Brahmaputra Flood Corridor on Day Map
  dCtx.save();
  dCtx.strokeStyle = 'rgba(36, 91, 255, 0.85)';
  dCtx.lineWidth = 3;
  dCtx.beginPath();
  [[30.0, 78.5], [27.5, 81.0], [25.5, 85.0], [24.0, 88.5], [22.2, 89.2]].forEach(
    ([lat, lon], i) => {
      const [x, y] = toXY(lat, lon);
      if (i === 0) dCtx.moveTo(x, y);
      else dCtx.lineTo(x, y);
    }
  );
  dCtx.stroke();
  dCtx.restore();

  // Graticule grid
  dCtx.strokeStyle = 'rgba(36, 91, 255, 0.11)';
  dCtx.lineWidth = 1;
  for (let lat = -60; lat <= 60; lat += 20) {
    const [, y] = toXY(lat, 0);
    dCtx.beginPath();
    dCtx.moveTo(0, y);
    dCtx.lineTo(width, y);
    dCtx.stroke();
  }
  for (let lon = -150; lon <= 180; lon += 20) {
    const [x] = toXY(0, lon);
    dCtx.beginPath();
    dCtx.moveTo(x, 0);
    dCtx.lineTo(x, height);
    dCtx.stroke();
  }

  // 2. NIGHT / EMISSIVE TELEMETRY MAP
  const emissiveCanvas = document.createElement('canvas');
  emissiveCanvas.width = width;
  emissiveCanvas.height = height;
  const eCtx = emissiveCanvas.getContext('2d');
  eCtx.fillStyle = '#000000';
  eCtx.fillRect(0, 0, width, height);

  drawPolygon(
    eCtx,
    indiaPolygon,
    'rgba(36, 91, 255, 0.26)',
    '#245BFF',
    4,
    22,
    '#245BFF'
  );

  const indianCities = [
    [28.6139, 77.209, 9],
    [19.076, 72.8777, 9],
    [12.9716, 77.5946, 8],
    [13.0827, 80.2707, 7],
    [22.5726, 88.3639, 9],
    [17.385, 78.4867, 8],
    [23.0225, 72.5714, 7],
    [26.1445, 91.7362, 8], // Guwahati / Brahmaputra Flood Sensor Node
    [25.5941, 85.1376, 8], // Patna / Ganga Basin Node
    [20.2961, 85.8245, 8], // Bhubaneswar / Mahanadi Delta Node
    [13.7199, 80.2304, 7], // Sriharikota SDSC
    [20.5937, 78.9629, 9], // Central India Calibration Node
  ];

  indianCities.forEach(([lat, lon, radius]) => {
    const [x, y] = toXY(lat, lon);
    const g = eCtx.createRadialGradient(x, y, 0, x, y, radius * 2.8);
    g.addColorStop(0, '#FFFFFF');
    g.addColorStop(0.28, '#C6FF00');
    g.addColorStop(0.65, 'rgba(36, 91, 255, 0.75)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    eCtx.fillStyle = g;
    eCtx.beginPath();
    eCtx.arc(x, y, radius * 2.8, 0, Math.PI * 2);
    eCtx.fill();

    for (let s = 0; s < 28; s++) {
      const sx = x + (Math.random() - 0.5) * 44;
      const sy = y + (Math.random() - 0.5) * 44;
      eCtx.fillStyle =
        Math.random() > 0.35
          ? 'rgba(255, 215, 130, 0.6)'
          : 'rgba(36, 91, 255, 0.75)';
      eCtx.fillRect(sx, sy, 2, 2);
    }
  });

  // 3. CLOUD LAYER
  const cloudCanvas = document.createElement('canvas');
  cloudCanvas.width = 1024;
  cloudCanvas.height = 512;
  const cCtx = cloudCanvas.getContext('2d');
  cCtx.clearRect(0, 0, 1024, 512);

  for (let i = 0; i < 400; i++) {
    const x = Math.random() * 1024;
    const y = 36 + Math.random() * 440;
    const rx = 18 + Math.random() * 64;
    const ry = 5 + Math.random() * 16;
    const g = cCtx.createRadialGradient(x, y, 0, x, y, rx);
    g.addColorStop(0, 'rgba(244, 247, 255, 0.24)');
    g.addColorStop(0.55, 'rgba(180, 215, 255, 0.1)');
    g.addColorStop(1, 'rgba(255, 255, 255, 0)');
    cCtx.fillStyle = g;
    cCtx.save();
    cCtx.translate(x, y);
    cCtx.scale(1, ry / rx);
    cCtx.beginPath();
    cCtx.arc(0, 0, rx, 0, Math.PI * 2);
    cCtx.fill();
    cCtx.restore();
  }

  const dayTex = new THREE.CanvasTexture(dayCanvas);
  const emissiveTex = new THREE.CanvasTexture(emissiveCanvas);
  const cloudTex = new THREE.CanvasTexture(cloudCanvas);
  dayTex.anisotropy = 4;
  emissiveTex.anisotropy = 4;

  return { dayTex, emissiveTex, cloudTex };
}

// ============================================================================
// BUILD HYPER-DETAILED 3D SATELLITE ASSEMBLY
// ============================================================================
function buildSatelliteAssembly(kaptonTex, solarTex) {
  const root = new THREE.Group();

  const kaptonMat = new THREE.MeshStandardMaterial({
    map: kaptonTex,
    roughnessMap: kaptonTex,
    color: new THREE.Color('#FFBE3D'),
    metalness: 0.88,
    roughness: 0.3,
  });

  const titaniumMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#C4D0E2'),
    metalness: 0.92,
    roughness: 0.2,
  });

  const darkAlloyMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#141C2B'),
    metalness: 0.86,
    roughness: 0.32,
  });

  const solarMat = new THREE.MeshStandardMaterial({
    map: solarTex,
    color: new THREE.Color('#4A88FF'),
    metalness: 0.8,
    roughness: 0.15,
    emissive: new THREE.Color('#0B2569'),
    emissiveIntensity: 0.55,
  });

  // Octagonal Gold Kapton Bus
  const busMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.82, 0.82, 2.35, 8),
    kaptonMat
  );
  busMesh.rotation.z = Math.PI / 2;
  root.add(busMesh);

  // Titanium Bulkhead Rings & Bolts
  [-1.18, -0.45, 0.45, 1.18].forEach((xPos) => {
    const ring = new THREE.Mesh(
      new THREE.CylinderGeometry(0.87, 0.87, 0.08, 16),
      titaniumMat
    );
    ring.rotation.z = Math.PI / 2;
    ring.position.x = xPos;
    root.add(ring);

    for (let b = 0; b < 8; b++) {
      const angle = (b / 8) * Math.PI * 2;
      const bolt = new THREE.Mesh(
        new THREE.CylinderGeometry(0.024, 0.024, 0.12, 6),
        titaniumMat
      );
      bolt.rotation.z = Math.PI / 2;
      bolt.position.set(xPos, Math.cos(angle) * 0.86, Math.sin(angle) * 0.86);
      root.add(bolt);
    }
  });

  // Forward Instrumentation Module
  const noseModule = new THREE.Mesh(
    new THREE.CylinderGeometry(0.62, 0.78, 0.7, 12),
    darkAlloyMat
  );
  noseModule.rotation.z = -Math.PI / 2;
  noseModule.position.x = 1.52;
  root.add(noseModule);

  // Aft Thruster Assembly
  const thrusterHousing = new THREE.Mesh(
    new THREE.CylinderGeometry(0.55, 0.72, 0.5, 12),
    darkAlloyMat
  );
  thrusterHousing.rotation.z = Math.PI / 2;
  thrusterHousing.position.x = -1.42;
  root.add(thrusterHousing);

  const mainNozzle = new THREE.Mesh(
    new THREE.CylinderGeometry(0.18, 0.36, 0.42, 20, 1, true),
    titaniumMat
  );
  mainNozzle.rotation.z = Math.PI / 2;
  mainNozzle.position.x = -1.78;
  root.add(mainNozzle);

  const ionGlow = new THREE.Mesh(
    new THREE.TorusGeometry(0.24, 0.025, 12, 24),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.cobaltBlue) })
  );
  ionGlow.rotation.y = Math.PI / 2;
  ionGlow.position.x = -1.82;
  root.add(ionGlow);

  // Dual Solar Array Wings
  const wingsGroup = new THREE.Group();
  [-1, 1].forEach((dir) => {
    const wingPivot = new THREE.Group();
    wingPivot.position.set(0, 0, dir * 0.9);

    const hinge = new THREE.Mesh(
      new THREE.CylinderGeometry(0.14, 0.14, 0.45, 16),
      titaniumMat
    );
    hinge.rotation.x = Math.PI / 2;
    wingPivot.add(hinge);

    const boomLen = 4.6;
    const boom = new THREE.Mesh(
      new THREE.BoxGeometry(0.12, 0.08, boomLen),
      titaniumMat
    );
    boom.position.z = dir * (boomLen / 2 + 0.15);
    wingPivot.add(boom);

    for (let s = 0; s < 3; s++) {
      const segZ = dir * (1.05 + s * 1.38);
      const frame = new THREE.Mesh(
        new THREE.BoxGeometry(1.42, 0.04, 1.32),
        darkAlloyMat
      );
      frame.position.z = segZ;

      const cells = new THREE.Mesh(
        new THREE.BoxGeometry(1.34, 0.052, 1.24),
        solarMat
      );
      cells.position.z = segZ;

      wingPivot.add(frame, cells);
    }

    wingsGroup.add(wingPivot);
  });
  root.add(wingsGroup);

  // Articulated High-Gain Parabolic Dish & White Laser Emitter
  const dishPivot = new THREE.Group();
  dishPivot.position.set(1.15, -0.35, 0.55);

  const turret = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22, 0.28, 0.35, 16),
    titaniumMat
  );
  dishPivot.add(turret);

  const lathePoints = [];
  for (let i = 0; i <= 12; i++) {
    const r = (i / 12) * 0.98;
    const y = r * r * 0.38;
    lathePoints.push(new THREE.Vector2(r, y));
  }
  const dishBowl = new THREE.Mesh(
    new THREE.LatheGeometry(lathePoints, 32),
    new THREE.MeshStandardMaterial({
      color: new THREE.Color('#E2ECFF'),
      metalness: 0.88,
      roughness: 0.18,
      side: THREE.DoubleSide,
    })
  );
  dishBowl.rotation.x = Math.PI / 2;

  const dishHead = new THREE.Group();
  dishHead.add(dishBowl);

  for (let i = 0; i < 3; i++) {
    const a = (i / 3) * Math.PI * 2;
    const strut = new THREE.Mesh(
      new THREE.CylinderGeometry(0.015, 0.015, 0.86, 8),
      titaniumMat
    );
    strut.position.set(Math.cos(a) * 0.36, Math.sin(a) * 0.36, 0.38);
    strut.lookAt(new THREE.Vector3(0, 0, 0.78));
    strut.rotateX(Math.PI / 2);
    dishHead.add(strut);
  }

  const feedHorn = new THREE.Mesh(
    new THREE.CylinderGeometry(0.08, 0.13, 0.22, 16),
    darkAlloyMat
  );
  feedHorn.rotation.x = Math.PI / 2;
  feedHorn.position.z = 0.76;
  dishHead.add(feedHorn);

  // Pure White Optical Emitter Core
  const emitterCoreMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color('#FFFFFF'),
  });
  const emitterCore = new THREE.Mesh(
    new THREE.SphereGeometry(0.065, 16, 16),
    emitterCoreMat
  );
  emitterCore.position.z = 0.87;
  dishHead.add(emitterCore);

  const emitterHalo = new THREE.Mesh(
    new THREE.RingGeometry(0.09, 0.18, 32),
    new THREE.MeshBasicMaterial({
      color: new THREE.Color('#FFFFFF'),
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.0,
    })
  );
  emitterHalo.position.z = 0.88;
  dishHead.add(emitterHalo);

  dishPivot.add(dishHead);
  root.add(dishPivot);

  const beaconLime = new THREE.Mesh(
    new THREE.SphereGeometry(0.042, 12, 12),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.acidLime) })
  );
  beaconLime.position.set(0.55, 1.02, 0.15);
  root.add(beaconLime);

  return {
    root,
    wingsGroup,
    dishPivot,
    emitterCore,
    emitterCoreMat,
    emitterHalo,
    beaconLime,
  };
}

// ============================================================================
// CONTINUOUS 8-SECTION KEYFRAME SPLINE DEFINITION
// ============================================================================
// Sections 0..7 mapped from normalized scroll s in [0, 1]
const SCROLL_KEYFRAMES = {
  // Section:           01 Hero  02 Miss  03 Obs   04 Flood 05 SAR   06 Link  07 Calib 08 Footer
  earthX:              [ 3.10,    0.32,   -2.85,   -2.05,    2.78,    2.25,   -2.35,    1.15 ],
  earthY:              [-0.35,   -0.12,   -0.22,    0.42,   -0.18,   -0.45,    0.15,   -0.25 ],
  earthZ:              [-1.80,    0.35,    1.10,    3.15,    1.45,   -2.85,   -0.35,   -4.20 ],
  earthScale:          [ 1.00,    1.26,    1.34,    1.58,    1.32,    0.86,    1.14,    0.76 ],
  earthYawOffset:      [ 0.00,    0.22,    0.48,    0.16,   -0.28,    0.10,    0.62,    1.15 ],
  earthPitchOffset:    [ 0.00,    0.08,    0.18,    0.26,    0.12,   -0.08,    0.32,    0.05 ],

  satX:                [-3.95,   -6.20,   -8.80,   -9.50,   -8.50,   -3.15,   -7.20,   -2.80 ],
  satY:                [ 0.28,    1.15,    2.10,    2.60,    1.95,    0.65,    1.85,    0.95 ],
  satZ:                [ 8.85,    6.40,    3.50,    2.00,    3.20,    7.80,    4.20,    6.10 ],
  satScale:            [ 1.02,    0.82,    0.60,    0.50,    0.60,    0.94,    0.65,    0.78 ],

  camX:                [-0.45,    0.00,   -0.55,   -0.75,    0.60,   -0.20,   -0.40,    0.15 ],
  camY:                [ 0.35,    0.25,    0.45,    0.58,    0.30,    0.40,    0.85,    0.55 ],
  camZ:                [13.40,   12.10,   11.40,   10.20,   11.50,   13.80,   12.30,   14.60 ],

  lookX:               [ 0.65,    0.18,   -1.45,   -1.35,    1.45,    0.15,   -1.15,    0.35 ],
  lookY:               [-0.08,   -0.05,   -0.10,    0.22,   -0.08,   -0.05,    0.08,   -0.10 ],
  lookZ:               [ 1.80,    0.80,    0.90,    2.50,    1.10,    0.50,    0.20,   -1.20 ],

  orbitRingOpacity:    [ 0.25,    0.72,    0.60,    0.30,    0.50,    0.88,    0.82,    0.92 ],
  chipGridOpacity:     [ 0.00,    0.05,    0.45,    1.00,    0.68,    0.15,    0.25,    0.10 ],
  downlinkArcOpacity:  [ 0.15,    0.20,    0.25,    0.35,    0.40,    1.00,    0.55,    0.45 ],
  atmoIntensity:       [ 1.10,    1.38,    1.32,    1.48,    1.35,    1.15,    1.28,    1.05 ],
};

// ============================================================================
// MAIN APPLICATION COMPONENT
// ============================================================================
export default function App() {
  // Intro & Scroll States
  const [introProgress, setIntroProgress] = useState(0);
  const [introComplete, setIntroComplete] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeSectionIdx, setActiveSectionIdx] = useState(0);
  const [inspect3DMode, setInspect3DMode] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [webglError, setWebglError] = useState(false);

  // Interactive Domain States (Sections 4, 5, 6, 7)
  const [selectedTileId, setSelectedTileId] = useState(5); // Default to a critical Acid Lime tile
  const [sensorMode, setSensorMode] = useState('AUTO'); // 'OPTICAL' | 'SAR' | 'AUTO'
  const [cloudCoverPct, setCloudCoverPct] = useState(48); // >30% triggers SAR in AUTO
  const [activePacketTier, setActivePacketTier] = useState('ALL'); // 'ALL' | 'L2' | 'L1' | 'L0'
  const [calibTarget, setCalibTarget] = useState('5%'); // '1%' | '5%' | '10%'
  const [telemetry, setTelemetry] = useState({
    lat: INDIA_COORDS.lat,
    lon: INDIA_COORDS.lon,
    alt: INDIA_COORDS.alt,
    snr: 99.98,
    downlinkGbps: 44.2,
  });

  // Imperative Refs for Smooth 60FPS Three.js Loop without React Re-render Lag
  const mountRef = useRef(null);
  const introProgressRef = useRef(0);
  const introCompleteRef = useRef(false);
  const targetScrollRef = useRef(0);
  const smoothScrollRef = useRef(0);
  const inspect3DRef = useRef(false);
  const sensorModeRef = useRef('SAR');
  const userDragRef = useRef({
    isDragging: false,
    prevX: 0,
    prevY: 0,
    rotY: 0,
    rotX: 0,
    zoom: 0,
  });

  // Sync sensor mode based on cloud cover threshold (>30% -> Sentinel-1 SAR)
  const effectiveSensor = useMemo(() => {
    if (sensorMode === 'AUTO') {
      return cloudCoverPct > 30 ? 'SAR' : 'OPTICAL';
    }
    return sensorMode;
  }, [sensorMode, cloudCoverPct]);

  useEffect(() => {
    sensorModeRef.current = effectiveSensor;
  }, [effectiveSensor]);

  useEffect(() => {
    inspect3DRef.current = inspect3DMode;
    if (!inspect3DMode) {
      // Smoothly decay manual drag offsets back to scroll choreography
      userDragRef.current.rotX = 0;
      userDragRef.current.rotY = 0;
      userDragRef.current.zoom = 0;
    }
  }, [inspect3DMode]);

  // Native Document Scroll Listener -> Normalized 0..1 Progress
  useEffect(() => {
    const handleScroll = () => {
      const docEl = document.documentElement;
      const maxScroll = Math.max(1, docEl.scrollHeight - window.innerHeight);
      const currentScroll = window.scrollY || docEl.scrollTop || 0;
      const norm = clamp(currentScroll / maxScroll, 0, 1);
      targetScrollRef.current = norm;

      // Fast-complete intro if visitor starts scrolling immediately
      if (currentScroll > 24 && !introCompleteRef.current) {
        introCompleteRef.current = true;
        introProgressRef.current = 1.0;
        setIntroComplete(true);
        setIntroProgress(1.0);
      }

      setScrollProgress(norm);
      const secIdx = Math.min(
        MISSION_SECTIONS.length - 1,
        Math.floor(norm * MISSION_SECTIONS.length + 0.04)
      );
      setActiveSectionIdx(secIdx);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Live Telemetry Micro-Jitter
  useEffect(() => {
    const timer = setInterval(() => {
      setTelemetry({
        lat: +(INDIA_COORDS.lat + (Math.random() - 0.5) * 0.0016).toFixed(4),
        lon: +(INDIA_COORDS.lon + (Math.random() - 0.5) * 0.0016).toFixed(4),
        alt: +(INDIA_COORDS.alt + (Math.random() - 0.5) * 0.12).toFixed(2),
        snr: +(99.95 + Math.random() * 0.04).toFixed(2),
        downlinkGbps: +(43.8 + Math.random() * 0.8).toFixed(1),
      });
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Navigation Helpers
  const scrollToSection = useCallback((id) => {
    setMobileMenuOpen(false);
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  const handleSkipIntro = useCallback(() => {
    introProgressRef.current = 1.0;
    introCompleteRef.current = true;
    setIntroProgress(1.0);
    setIntroComplete(true);
  }, []);

  const handleReplayIntro = useCallback(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    setTimeout(() => {
      introProgressRef.current = 0.0;
      introCompleteRef.current = false;
      setIntroProgress(0.0);
      setIntroComplete(false);
    }, 250);
  }, []);

  // ============================================================================
  // PERSISTENT SINGLE-CANVAS THREE.JS SCENE & UNIFIED ANIMATION LOOP
  // ============================================================================
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: 'high-performance',
        alpha: false,
      });
    } catch (err) {
      setWebglError(true);
      return;
    }

    const isMobile = window.innerWidth < 768;
    const dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.22;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.midnightBlack);
    scene.fog = new THREE.FogExp2(COLORS.midnightBlack, 0.0038);

    const camera = new THREE.PerspectiveCamera(
      40,
      container.clientWidth / container.clientHeight,
      0.1,
      400
    );

    // ------------------------------------------------------------------------
    // LIGHTING
    // ------------------------------------------------------------------------
    const ambientLight = new THREE.AmbientLight('#182B52', 0.72);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight('#FFF9EE', 0.1);
    sunLight.position.set(18, 12, 18);
    scene.add(sunLight);

    const cobaltRimLight = new THREE.PointLight(COLORS.cobaltBlue, 0.0, 32);
    cobaltRimLight.position.set(-7, 4, 13);
    scene.add(cobaltRimLight);

    const goldBounceLight = new THREE.PointLight('#FFB946', 0.0, 22);
    goldBounceLight.position.set(-3, -4, 13);
    scene.add(goldBounceLight);

    // Crisp White Point-Light Flash at India Laser Target
    const indiaImpactLight = new THREE.PointLight('#FFFFFF', 0.0, 6.5);
    scene.add(indiaImpactLight);

    // ------------------------------------------------------------------------
    // DEEP SPACE STARFIELD
    // ------------------------------------------------------------------------
    const starCount = isMobile ? 1500 : 3600;
    const starGeo = new THREE.BufferGeometry();
    const starPositions = new Float32Array(starCount * 3);
    const starColors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      const r = 65 + Math.random() * 145;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      starPositions[i * 3 + 2] = r * Math.cos(phi);

      const c =
        Math.random() > 0.8
          ? new THREE.Color('#78A4FF')
          : Math.random() > 0.94
          ? new THREE.Color('#C6FF00')
          : new THREE.Color('#F4F7FF');
      starColors[i * 3] = c.r;
      starColors[i * 3 + 1] = c.g;
      starColors[i * 3 + 2] = c.b;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

    const starMat = new THREE.PointsMaterial({
      size: isMobile ? 0.24 : 0.19,
      vertexColors: true,
      transparent: true,
      opacity: 0.0,
    });
    const starField = new THREE.Points(starGeo, starMat);
    scene.add(starField);

    // ------------------------------------------------------------------------
    // 3D SATELLITE MODEL
    // ------------------------------------------------------------------------
    const kaptonTex = createKaptonTextures();
    const solarTex = createSolarPanelTexture();
    const satellite = buildSatelliteAssembly(kaptonTex, solarTex);
    satellite.root.position.set(-3.95, 0.28, 8.85);
    satellite.root.rotation.set(0.22, 0.55, -0.18);
    scene.add(satellite.root);

    // ------------------------------------------------------------------------
    // PERSISTENT 3D EARTH SCENE
    // ------------------------------------------------------------------------
    const EARTH_RADIUS = 4.65;
    const earthGroup = new THREE.Group();
    earthGroup.position.set(3.1, -0.35, -1.8);
    scene.add(earthGroup);

    const { dayTex, emissiveTex, cloudTex } = createEarthTextures();

    const earthMat = new THREE.MeshStandardMaterial({
      map: dayTex,
      emissiveMap: emissiveTex,
      emissive: new THREE.Color('#FFFFFF'),
      emissiveIntensity: 1.08,
      roughness: 0.52,
      metalness: 0.2,
    });
    const earthMesh = new THREE.Mesh(
      new THREE.SphereGeometry(EARTH_RADIUS, 64, 64),
      earthMat
    );
    earthGroup.add(earthMesh);

    // Base orientation so India (20.59°N, 78.96°E) faces the camera
    const baseEarthRotY = -((INDIA_COORDS.lon + 98) * Math.PI) / 180;
    const baseEarthRotX = ((INDIA_COORDS.lat - 8) * Math.PI) / 180;
    earthGroup.rotation.set(baseEarthRotX, baseEarthRotY, 0.06);

    // Atmospheric Clouds
    const cloudMat = new THREE.MeshStandardMaterial({
      map: cloudTex,
      transparent: true,
      opacity: 0.48,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const cloudMesh = new THREE.Mesh(
      new THREE.SphereGeometry(EARTH_RADIUS * 1.013, 48, 48),
      cloudMat
    );
    earthGroup.add(cloudMesh);

    // Custom GLSL Fresnel Electric Cobalt Atmospheric Glow
    const atmoMat = new THREE.ShaderMaterial({
      uniforms: {
        uIntensity: { value: 1.15 },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPositionNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPositionNormal = normalize((modelViewMatrix * vec4(position, 1.0)).xyz);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vPositionNormal;
        uniform float uIntensity;
        void main() {
          float fresnel = pow(0.68 - dot(vNormal, vPositionNormal), 2.7);
          vec3 col = mix(vec3(0.141, 0.357, 1.0), vec3(0.65, 0.88, 1.0), fresnel);
          gl_FragColor = vec4(col, clamp(fresnel * uIntensity, 0.0, 0.9));
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
    });
    const atmoMesh = new THREE.Mesh(
      new THREE.SphereGeometry(EARTH_RADIUS * 1.068, 64, 64),
      atmoMat
    );
    earthGroup.add(atmoMesh);

    // ------------------------------------------------------------------------
    // INDIA TARGET OVERLAY, 3D 4x4 FLOOD CHIP GRID & RIPPLES ON EARTH SURFACE
    // ------------------------------------------------------------------------
    const indiaSurfacePos = latLonToVector3(
      INDIA_COORDS.lat,
      INDIA_COORDS.lon,
      EARTH_RADIUS * 1.016
    );
    const indiaTargetGroup = new THREE.Group();
    indiaTargetGroup.position.copy(indiaSurfacePos);
    indiaTargetGroup.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      indiaSurfacePos.clone().normalize()
    );
    earthGroup.add(indiaTargetGroup);

    // Target Center Pin (Pure White + Acid Lime Ring)
    const indiaCorePin = new THREE.Mesh(
      new THREE.CircleGeometry(0.065, 24),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#FFFFFF'),
        side: THREE.DoubleSide,
      })
    );
    indiaTargetGroup.add(indiaCorePin);

    // Expanding Energy Ripples
    const rippleMeshes = [];
    for (let r = 0; r < 3; r++) {
      const rippleMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(r === 0 ? '#FFFFFF' : r === 1 ? COLORS.acidLime : COLORS.cobaltBlue),
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const ripple = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.24, 48), rippleMat);
      indiaTargetGroup.add(ripple);
      rippleMeshes.push(ripple);
    }

    // 3D 4x4 Flood Detection Tile Grid on Earth Surface (Illuminates in Section 4)
    const chip3DGroup = new THREE.Group();
    chip3DGroup.position.z = 0.035;
    indiaTargetGroup.add(chip3DGroup);

    const tile3DMeshes = [];
    const gridSpan = 1.32;
    const cellSpan = gridSpan / 4;
    FLOOD_TILES.forEach((t, idx) => {
      const r = Math.floor(idx / 4);
      const c = idx % 4;
      const tx = (c - 1.5) * cellSpan;
      const ty = (1.5 - r) * cellSpan;

      const tileMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(
          t.isCritical ? COLORS.acidLime : t.isModerate ? COLORS.cobaltBlue : '#1E3A6E'
        ),
        transparent: true,
        opacity: 0.0,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const tileMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(cellSpan * 0.86, cellSpan * 0.86),
        tileMat
      );
      tileMesh.position.set(tx, ty, 0);
      chip3DGroup.add(tileMesh);
      tile3DMeshes.push({ mesh: tileMesh, mat: tileMat, isCritical: t.isCritical });
    });

    // ------------------------------------------------------------------------
    // ORBITAL RINGS & MINIATURE ORBITING SATELLITE
    // ------------------------------------------------------------------------
    const orbitRadius = EARTH_RADIUS * 1.18;
    const orbitPoints = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      orbitPoints.push(
        new THREE.Vector3(
          Math.cos(a) * orbitRadius,
          Math.sin(a) * orbitRadius * 0.42,
          Math.sin(a) * orbitRadius * 0.9
        )
      );
    }
    const orbitLineMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(COLORS.cobaltBlue),
      transparent: true,
      opacity: 0.3,
    });
    const orbitRingLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(orbitPoints),
      orbitLineMat
    );
    earthGroup.add(orbitRingLine);

    // Second Polar Calibration Orbit Ring
    const polarPoints = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      polarPoints.push(
        new THREE.Vector3(
          Math.cos(a) * orbitRadius * 0.35,
          Math.sin(a) * orbitRadius * 1.04,
          Math.cos(a) * orbitRadius * 0.98
        )
      );
    }
    const polarOrbitMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(COLORS.acidLime),
      transparent: true,
      opacity: 0.15,
    });
    const polarOrbitLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(polarPoints),
      polarOrbitMat
    );
    earthGroup.add(polarOrbitLine);

    // Mini Orbiting Satellite
    const miniSatGroup = new THREE.Group();
    miniSatGroup.add(
      new THREE.Mesh(
        new THREE.BoxGeometry(0.11, 0.11, 0.2),
        new THREE.MeshBasicMaterial({ color: new THREE.Color('#FFFFFF') })
      ),
      new THREE.Mesh(
        new THREE.BoxGeometry(0.52, 0.02, 0.12),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.cobaltBlue) })
      )
    );
    earthGroup.add(miniSatGroup);

    // ------------------------------------------------------------------------
    // RAZOR-THIN PURE WHITE OPTICAL LASER CORE + COBALT HALO SHEATH
    // ------------------------------------------------------------------------
    const laserGroup = new THREE.Group();
    scene.add(laserGroup);

    // 1. Needle-Thin Pure White Core (radius = 0.0045)
    const coreGeo = new THREE.CylinderGeometry(0.0045, 0.0045, 1, 12, 1, true);
    coreGeo.translate(0, 0.5, 0);
    coreGeo.rotateX(Math.PI / 2);
    const coreLaserMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uProgress: { value: 0 },
        uOpacity: { value: 0 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform float uTime;
        uniform float uProgress;
        uniform float uOpacity;
        void main() {
          if (vUv.y > uProgress) discard;
          float carrier = 0.88 + 0.12 * sin(vUv.y * 90.0 - uTime * 28.0);
          gl_FragColor = vec4(vec3(1.0, 1.0, 1.0), carrier * uOpacity);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const coreLaserMesh = new THREE.Mesh(coreGeo, coreLaserMat);
    laserGroup.add(coreLaserMesh);

    // 2. Tight Pale-Ice & Electric Cobalt Collimated Outer Sheath (radius = 0.018)
    const sheathGeo = new THREE.CylinderGeometry(0.016, 0.024, 1, 16, 1, true);
    sheathGeo.translate(0, 0.5, 0);
    sheathGeo.rotateX(Math.PI / 2);
    const sheathLaserMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uProgress: { value: 0 },
        uOpacity: { value: 0 },
        uCobalt: { value: new THREE.Color(COLORS.cobaltBlue) },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform float uTime;
        uniform float uProgress;
        uniform float uOpacity;
        uniform vec3 uCobalt;
        void main() {
          if (vUv.y > uProgress) discard;
          float pulse = smoothstep(0.1, 0.0, abs(fract(vUv.y * 4.0 - uTime * 3.6) - 0.5));
          vec3 col = mix(uCobalt, vec3(1.0), pulse * 0.75);
          float alpha = (0.28 + pulse * 0.55) * uOpacity;
          gl_FragColor = vec4(col, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const sheathLaserMesh = new THREE.Mesh(sheathGeo, sheathLaserMat);
    laserGroup.add(sheathLaserMesh);

    // 3. High-Speed Photonic White & Acid-Lime Telemetry Packets
    const packetCount = 28;
    const packetGeo = new THREE.BufferGeometry();
    const packetPos = new Float32Array(packetCount * 3);
    const packetColors = new Float32Array(packetCount * 3);
    for (let i = 0; i < packetCount; i++) {
      // Every 4th packet is an L2 Acid Lime priority flood alert packet; others are Pure White optical pulses
      const c =
        i % 4 === 0
          ? new THREE.Color(COLORS.acidLime)
          : new THREE.Color('#FFFFFF');
      packetColors[i * 3] = c.r;
      packetColors[i * 3 + 1] = c.g;
      packetColors[i * 3 + 2] = c.b;
    }
    packetGeo.setAttribute('position', new THREE.BufferAttribute(packetPos, 3));
    packetGeo.setAttribute('color', new THREE.BufferAttribute(packetColors, 3));
    const packetMat = new THREE.PointsMaterial({
      size: 0.11,
      vertexColors: true,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const laserPackets = new THREE.Points(packetGeo, packetMat);
    scene.add(laserPackets);

    // ------------------------------------------------------------------------
    // INTERACTIVE 3D INSPECT MODE POINTER LISTENERS
    // ------------------------------------------------------------------------
    const onPointerDown = (e) => {
      if (!inspect3DRef.current) return;
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      const cy = e.touches ? e.touches[0].clientY : e.clientY;
      userDragRef.current.isDragging = true;
      userDragRef.current.prevX = cx;
      userDragRef.current.prevY = cy;
    };

    const onPointerMove = (e) => {
      if (!inspect3DRef.current || !userDragRef.current.isDragging) return;
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      const cy = e.touches ? e.touches[0].clientY : e.clientY;
      const dx = cx - userDragRef.current.prevX;
      const dy = cy - userDragRef.current.prevY;
      userDragRef.current.prevX = cx;
      userDragRef.current.prevY = cy;

      userDragRef.current.rotY += dx * 0.0055;
      userDragRef.current.rotX = clamp(
        userDragRef.current.rotX + dy * 0.004,
        -0.65,
        0.65
      );
    };

    const onPointerUp = () => {
      userDragRef.current.isDragging = false;
    };

    const domElem = renderer.domElement;
    domElem.addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);
    domElem.addEventListener('touchstart', onPointerDown, { passive: true });
    window.addEventListener('touchmove', onPointerMove, { passive: true });
    window.addEventListener('touchend', onPointerUp);

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // ------------------------------------------------------------------------
    // UNIFIED 60FPS ANIMATION LOOP (INTRO + SPLINE SCROLL JOURNEY)
    // ------------------------------------------------------------------------
    const clock = new THREE.Clock();
    let animId;
    const emitterWorldPos = new THREE.Vector3();
    const indiaWorldPos = new THREE.Vector3();
    const currentCamLook = new THREE.Vector3(0.65, -0.08, 1.8);

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      const elapsed = clock.getElapsedTime();

      // 1. Advance Intro Sequence if not yet completed
      if (!introCompleteRef.current) {
        introProgressRef.current = clamp(
          introProgressRef.current + dt * 0.085,
          0,
          1
        );
        setIntroProgress(introProgressRef.current);
        if (introProgressRef.current >= 1.0) {
          introCompleteRef.current = true;
          setIntroComplete(true);
        }
      }

      const ip = introProgressRef.current;

      // 2. Exponential Damping of Normalized Scroll Progress (smooths wheel & trackpad steps)
      smoothScrollRef.current = THREE.MathUtils.damp(
        smoothScrollRef.current,
        targetScrollRef.current,
        5.2,
        dt
      );
      const s = smoothScrollRef.current;

      // 3. Evaluate Continuous Cubic-Spline Target Pose for Scroll s in [0, 1]
      const mobileScaleMod = isMobile ? 0.78 : 1.0;
      const mobileXMod = isMobile ? 0.45 : 1.0;

      const scrollEarthX = sampleScalarSpline(SCROLL_KEYFRAMES.earthX, s) * mobileXMod;
      const scrollEarthY = sampleScalarSpline(SCROLL_KEYFRAMES.earthY, s);
      const scrollEarthZ = sampleScalarSpline(SCROLL_KEYFRAMES.earthZ, s);
      const scrollEarthScale =
        sampleScalarSpline(SCROLL_KEYFRAMES.earthScale, s) * mobileScaleMod;
      const scrollEarthYaw = sampleScalarSpline(SCROLL_KEYFRAMES.earthYawOffset, s);
      const scrollEarthPitch = sampleScalarSpline(SCROLL_KEYFRAMES.earthPitchOffset, s);

      const scrollSatX = sampleScalarSpline(SCROLL_KEYFRAMES.satX, s) * mobileXMod;
      const scrollSatY = sampleScalarSpline(SCROLL_KEYFRAMES.satY, s);
      const scrollSatZ = sampleScalarSpline(SCROLL_KEYFRAMES.satZ, s);
      const scrollSatScale =
        sampleScalarSpline(SCROLL_KEYFRAMES.satScale, s) * mobileScaleMod;

      const scrollCamX = sampleScalarSpline(SCROLL_KEYFRAMES.camX, s) * mobileXMod;
      const scrollCamY = sampleScalarSpline(SCROLL_KEYFRAMES.camY, s);
      const scrollCamZ = sampleScalarSpline(SCROLL_KEYFRAMES.camZ, s);

      const scrollLookX = sampleScalarSpline(SCROLL_KEYFRAMES.lookX, s) * mobileXMod;
      const scrollLookY = sampleScalarSpline(SCROLL_KEYFRAMES.lookY, s);
      const scrollLookZ = sampleScalarSpline(SCROLL_KEYFRAMES.lookZ, s);

      // 4. Blend Intro Camera & Lighting smoothly into Section 1 Scroll Pose (Zero Jump!)
      const revealFactor = smoothstep(0.0, 0.22, ip);
      starMat.opacity = revealFactor * 0.92;
      sunLight.intensity = smoothstep(0.02, 0.25, ip) * 2.95;
      cobaltRimLight.intensity = smoothstep(0.05, 0.25, ip) * 3.8;
      goldBounceLight.intensity = smoothstep(0.08, 0.25, ip) * 2.1;

      // During early intro (ip < 1), camera starts slightly closer to satellite and converges to scrollCam at ip == 1.0
      const introBlend = smoothstep(0.0, 1.0, ip);
      const targetCamX = THREE.MathUtils.lerp(-2.35 * mobileXMod, scrollCamX, introBlend);
      const targetCamY = THREE.MathUtils.lerp(0.42, scrollCamY, introBlend);
      const targetCamZ = THREE.MathUtils.lerp(13.95, scrollCamZ, introBlend);

      const targetLookX = THREE.MathUtils.lerp(-0.95 * mobileXMod, scrollLookX, introBlend);
      const targetLookY = THREE.MathUtils.lerp(0.08, scrollLookY, introBlend);
      const targetLookZ = THREE.MathUtils.lerp(4.2, scrollLookZ, introBlend);

      // Apply damped transforms to Earth
      earthGroup.position.x = THREE.MathUtils.damp(earthGroup.position.x, scrollEarthX, 6, dt);
      earthGroup.position.y = THREE.MathUtils.damp(earthGroup.position.y, scrollEarthY, 6, dt);
      earthGroup.position.z = THREE.MathUtils.damp(earthGroup.position.z, scrollEarthZ, 6, dt);

      const currentEarthScale = THREE.MathUtils.damp(
        earthGroup.scale.x,
        scrollEarthScale,
        6,
        dt
      );
      earthGroup.scale.setScalar(currentEarthScale);

      // Continuous orbital rotation + spline geographical focus + optional user inspect drag
      const desiredRotY =
        baseEarthRotY +
        scrollEarthYaw +
        userDragRef.current.rotY +
        Math.sin(elapsed * 0.12) * 0.035;
      const desiredRotX =
        baseEarthRotX + scrollEarthPitch + userDragRef.current.rotX;

      earthGroup.rotation.y = THREE.MathUtils.damp(earthGroup.rotation.y, desiredRotY, 6, dt);
      earthGroup.rotation.x = THREE.MathUtils.damp(earthGroup.rotation.x, desiredRotX, 6, dt);

      // Cloud rotation & sensor mode cloud opacity (SAR mode makes clouds translucent so radar sees through)
      cloudMesh.rotation.y = elapsed * 0.022;
      const targetCloudOp = sensorModeRef.current === 'SAR' ? 0.16 : 0.52;
      cloudMat.opacity = THREE.MathUtils.damp(cloudMat.opacity, targetCloudOp, 4, dt);

      // Atmospheric Fresnel Intensity
      atmoMat.uniforms.uIntensity.value = sampleScalarSpline(
        SCROLL_KEYFRAMES.atmoIntensity,
        s
      );

      // Apply damped transforms to Foreground/Orbital Satellite
      satellite.root.position.x = THREE.MathUtils.damp(
        satellite.root.position.x,
        scrollSatX,
        6,
        dt
      );
      satellite.root.position.y = THREE.MathUtils.damp(
        satellite.root.position.y,
        scrollSatY + Math.sin(elapsed * 0.9) * 0.08,
        6,
        dt
      );
      satellite.root.position.z = THREE.MathUtils.damp(
        satellite.root.position.z,
        scrollSatZ,
        6,
        dt
      );
      satellite.root.scale.setScalar(
        THREE.MathUtils.damp(satellite.root.scale.x, scrollSatScale, 6, dt)
      );
      satellite.beaconLime.scale.setScalar(0.85 + Math.sin(elapsed * 7.0) * 0.35);

      // Apply damped transforms to Camera
      camera.position.x = THREE.MathUtils.damp(camera.position.x, targetCamX, 6, dt);
      camera.position.y = THREE.MathUtils.damp(camera.position.y, targetCamY, 6, dt);
      camera.position.z = THREE.MathUtils.damp(camera.position.z, targetCamZ, 6, dt);

      currentCamLook.x = THREE.MathUtils.damp(currentCamLook.x, targetLookX, 6, dt);
      currentCamLook.y = THREE.MathUtils.damp(currentCamLook.y, targetLookY, 6, dt);
      currentCamLook.z = THREE.MathUtils.damp(currentCamLook.z, targetLookZ, 6, dt);
      camera.lookAt(currentCamLook);

      // Update Orbit Rings & Mini Orbiting Satellite
      const orbitOp = sampleScalarSpline(SCROLL_KEYFRAMES.orbitRingOpacity, s);
      orbitLineMat.opacity = orbitOp;
      polarOrbitMat.opacity = orbitOp * 0.65;

      const miniAng = elapsed * 0.65;
      miniSatGroup.position.set(
        Math.cos(miniAng) * orbitRadius,
        Math.sin(miniAng) * orbitRadius * 0.42,
        Math.sin(miniAng) * orbitRadius * 0.9
      );
      miniSatGroup.lookAt(earthGroup.position);

      // Update 3D 4x4 Flood Tile Grid on Earth Surface (Peaks in Section 4)
      const gridOp = sampleScalarSpline(SCROLL_KEYFRAMES.chipGridOpacity, s);
      tile3DMeshes.forEach((item, idx) => {
        const pulse = item.isCritical
          ? 0.65 + 0.35 * Math.sin(elapsed * 5.0 + idx)
          : 0.35;
        item.mat.opacity = gridOp * pulse;
      });

      // 5. RAZOR-THIN WHITE OPTICAL LASER & TELEMETRY PACKETS (Active in Intro & Section 1 & Section 6)
      satellite.emitterCore.getWorldPosition(emitterWorldPos);
      indiaCorePin.getWorldPosition(indiaWorldPos);

      // Dish always tracks India target
      satellite.dishPivot.lookAt(indiaWorldPos);

      const antennaLock = smoothstep(0.22, 0.38, ip);
      const laserExtend = smoothstep(0.38, 0.6, ip);
      const downlinkBoost = sampleScalarSpline(SCROLL_KEYFRAMES.downlinkArcOpacity, s);
      // Laser stays visible in Hero (s ~ 0), fades slightly during close surface inspection, and surges in Section 6 (Data Transmission)
      const scrollLaserVisibility = clamp(
        1.0 - smoothstep(0.08, 0.25, s) + downlinkBoost * 0.95,
        0.22,
        1.0
      );

      const beamDist = emitterWorldPos.distanceTo(indiaWorldPos);
      coreLaserMesh.position.copy(emitterWorldPos);
      coreLaserMesh.lookAt(indiaWorldPos);
      coreLaserMesh.scale.set(1, 1, beamDist);

      sheathLaserMesh.position.copy(emitterWorldPos);
      sheathLaserMesh.lookAt(indiaWorldPos);
      sheathLaserMesh.scale.set(1, 1, beamDist);

      coreLaserMat.uniforms.uTime.value = elapsed;
      coreLaserMat.uniforms.uProgress.value = laserExtend;
      coreLaserMat.uniforms.uOpacity.value = antennaLock * scrollLaserVisibility;

      sheathLaserMat.uniforms.uTime.value = elapsed;
      sheathLaserMat.uniforms.uProgress.value = laserExtend;
      sheathLaserMat.uniforms.uOpacity.value = antennaLock * scrollLaserVisibility * 0.85;

      satellite.emitterHalo.material.opacity = antennaLock * scrollLaserVisibility * 0.9;
      satellite.emitterHalo.scale.setScalar(1.0 + Math.sin(elapsed * 12.0) * 0.18);

      // Update travelling photonic packets along the razor-thin laser beam
      packetMat.opacity = antennaLock * scrollLaserVisibility * 0.95;
      const pAttr = packetGeo.attributes.position;
      for (let i = 0; i < packetCount; i++) {
        const t = ((i / packetCount + elapsed * 0.72) % 1.0) * laserExtend;
        pAttr.setXYZ(
          i,
          THREE.MathUtils.lerp(emitterWorldPos.x, indiaWorldPos.x, t),
          THREE.MathUtils.lerp(emitterWorldPos.y, indiaWorldPos.y, t),
          THREE.MathUtils.lerp(emitterWorldPos.z, indiaWorldPos.z, t)
        );
      }
      pAttr.needsUpdate = true;

      // Target Point Flash & Concentric Ripples over India
      const impactActive = smoothstep(0.54, 0.68, ip);
      indiaImpactLight.position.copy(indiaWorldPos);
      indiaImpactLight.intensity =
        impactActive * (1.8 + Math.sin(elapsed * 10.0) * 0.6);

      rippleMeshes.forEach((ripple, idx) => {
        const phase = (elapsed * 0.68 + idx * 0.33) % 1.0;
        ripple.scale.setScalar(0.35 + phase * 3.4);
        ripple.material.opacity = (1.0 - phase) * impactActive * 0.85;
      });

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      domElem.removeEventListener('mousedown', onPointerDown);
      window.removeEventListener('mousemove', onPointerMove);
      window.removeEventListener('mouseup', onPointerUp);
      domElem.removeEventListener('touchstart', onPointerDown);
      window.removeEventListener('touchmove', onPointerMove);
      window.removeEventListener('touchend', onPointerUp);
      renderer.dispose();
    };
  }, []);

  // Active Intro Step helper
  const currentIntroStep = useMemo(() => {
    for (const st of INTRO_STEPS) {
      if (introProgress >= st.range[0] && introProgress <= st.range[1]) return st;
    }
    return INTRO_STEPS[INTRO_STEPS.length - 1];
  }, [introProgress]);

  const selectedTileObj = useMemo(
    () => FLOOD_TILES.find((t) => t.id === selectedTileId) || FLOOD_TILES[5],
    [selectedTileId]
  );

  const activeCalibData = CALIBRATION_TARGETS[calibTarget];

  return (
    <div
      className="relative w-full min-h-screen overflow-x-hidden font-sans selection:bg-[#C6FF00] selection:text-[#05070D]"
      style={{ backgroundColor: COLORS.midnightBlack, color: COLORS.white }}
    >
      {/* ====================================================================
          PERSISTENT FIXED 3D WEBGL VIEWPORT CANVAS
          pointer-events-none by default so native scrolling never gets hijacked
      ==================================================================== */}
      {!webglError ? (
        <div
          ref={mountRef}
          className={`fixed inset-0 w-full h-screen z-0 transition-opacity duration-500 ${
            inspect3DMode
              ? 'pointer-events-auto cursor-grab active:cursor-grabbing'
              : 'pointer-events-none'
          }`}
        />
      ) : (
        <div
          className="fixed inset-0 z-0 flex items-center justify-center p-8 text-center"
          style={{
            background:
              'radial-gradient(circle at 70% 45%, #0E2356 0%, #05070D 75%)',
          }}
        >
          <div className="max-w-md p-6 rounded-xl border border-[#245BFF]/40 bg-[#0A1020]/90 backdrop-blur-md">
            <Satellite className="w-10 h-10 mx-auto mb-3 text-[#C6FF00]" />
            <h2 className="text-lg font-bold tracking-wider mb-1">
              ORBIT-PRISM STATIC TELEMETRY MODE
            </h2>
            <p className="text-xs text-[#8B95A7]">
              WebGL 3D acceleration is disabled on this device. All 8 mission
              sections and flood-intelligence controls remain interactive below.
            </p>
          </div>
        </div>
      )}

      {/* Subtle Ambient Space Vignette & Technical Grid Overlay */}
      <div
        className="pointer-events-none fixed inset-0 z-[1]"
        style={{
          background:
            'radial-gradient(circle at 50% 50%, transparent 52%, rgba(5, 7, 13, 0.82) 100%)',
        }}
      />

      {/* ====================================================================
          TOP NAVIGATION BAR + CONTINUOUS SCROLL PROGRESS BAR
      ==================================================================== */}
      <header className="fixed top-0 left-0 right-0 z-40 border-b border-[#245BFF]/30 bg-[#05070D]/75 backdrop-blur-xl">
        {/* Top Acid-Lime / Cobalt Scroll Progress Ribbon */}
        <div className="w-full h-[2px] bg-[#0A1020]">
          <div
            className="h-full bg-gradient-to-r from-[#245BFF] via-[#5E8BFF] to-[#C6FF00] transition-all duration-150"
            style={{ width: `${Math.max(2, scrollProgress * 100)}%` }}
          />
        </div>

        <div className="max-w-7xl mx-auto flex items-center justify-between px-4 sm:px-8 py-3">
          {/* Brand Identity */}
          <button
            onClick={() => scrollToSection('sec-hero')}
            className="flex items-center gap-3 text-left group"
          >
            <div className="relative flex items-center justify-center w-9 h-9 rounded-lg border border-[#245BFF]/70 bg-[#0A1020] shadow-[0_0_18px_rgba(36,91,255,0.45)] group-hover:border-[#C6FF00] transition-colors">
              <Satellite className="w-5 h-5 text-[#C6FF00]" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#C6FF00] animate-ping" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold tracking-[0.24em] text-sm sm:text-base text-[#F4F7FF]">
                  ORBIT-PRISM
                </span>
                <span className="hidden sm:inline-block text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#245BFF]/25 text-[#C6FF00] border border-[#245BFF]/50">
                  EO-SAR-IND
                </span>
              </div>
              <p className="text-[9px] font-mono tracking-[0.18em] text-[#8B95A7] hidden sm:block">
                DETECT THE FLOOD • PRIORITIZE THE DATA
              </p>
            </div>
          </button>

          {/* Section Navigation Links */}
          <nav className="hidden xl:flex items-center gap-5 text-[11px] font-mono tracking-[0.16em]">
            {MISSION_SECTIONS.slice(0, 7).map((sec) => {
              const isActive = activeSectionIdx === sec.index;
              return (
                <button
                  key={sec.id}
                  onClick={() => scrollToSection(sec.id)}
                  className={`relative py-1 transition-colors ${
                    isActive
                      ? 'text-[#C6FF00] font-bold'
                      : 'text-[#F4F7FF]/75 hover:text-[#F4F7FF]'
                  }`}
                >
                  <span className="text-[#245BFF] mr-1">{sec.code}</span>
                  {sec.label}
                  {isActive && (
                    <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[#C6FF00] shadow-[0_0_10px_#C6FF00]" />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Right Controls: 3D Inspect Toggle, Replay / Skip Intro */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setInspect3DMode(!inspect3DMode)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-[10px] font-mono tracking-wider border transition-all ${
                inspect3DMode
                  ? 'border-[#C6FF00] bg-[#C6FF00] text-[#05070D] font-bold shadow-[0_0_18px_rgba(198,255,0,0.5)]'
                  : 'border-[#245BFF]/50 bg-[#0A1020]/80 text-[#F4F7FF] hover:border-[#C6FF00]'
              }`}
              title="Toggle manual 3D Earth rotation without blocking page scroll when off"
            >
              <Globe className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">
                {inspect3DMode ? 'EXIT 3D DRAG' : '3D DRAG MODE'}
              </span>
            </button>

            {!introComplete ? (
              <button
                onClick={handleSkipIntro}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded text-[10px] font-mono tracking-wider border border-[#C6FF00]/60 bg-[#C6FF00]/15 text-[#C6FF00] hover:bg-[#C6FF00]/25"
              >
                <FastForward className="w-3.5 h-3.5" />
                <span>SKIP INTRO</span>
              </button>
            ) : (
              <button
                onClick={handleReplayIntro}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded text-[10px] font-mono tracking-wider border border-[#245BFF]/50 bg-[#0A1020]/80 text-[#F4F7FF] hover:border-[#C6FF00] hover:text-[#C6FF00]"
                title="Replay 5-Step Opening Sequence"
              >
                <RotateCcw className="w-3.5 h-3.5 text-[#245BFF]" />
                <span className="hidden sm:inline">REPLAY INTRO</span>
              </button>
            )}

            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="xl:hidden p-2 rounded border border-[#245BFF]/40 bg-[#0A1020] text-[#F4F7FF]"
              aria-label="Toggle Menu"
            >
              {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Mobile Drawer */}
        {mobileMenuOpen && (
          <div className="xl:hidden bg-[#0A1020]/95 backdrop-blur-xl border-b border-[#245BFF]/40 px-6 py-4 grid grid-cols-2 gap-2 font-mono text-xs">
            {MISSION_SECTIONS.map((sec) => (
              <button
                key={sec.id}
                onClick={() => scrollToSection(sec.id)}
                className="text-left py-2 px-3 rounded border border-white/5 hover:border-[#C6FF00] text-[#F4F7FF] hover:text-[#C6FF00]"
              >
                <span className="text-[#245BFF] mr-1.5">{sec.code}</span>
                {sec.label}
              </button>
            ))}
          </div>
        )}
      </header>

      {/* ====================================================================
          RIGHT-EDGE VERTICAL MISSION PROGRESS RAIL (DESKTOP)
      ==================================================================== */}
      <aside className="hidden lg:flex fixed right-5 top-1/2 -translate-y-1/2 z-30 flex-col gap-2.5 p-2 rounded-full border border-[#245BFF]/30 bg-[#0A1020]/70 backdrop-blur-md">
        {MISSION_SECTIONS.map((sec) => {
          const active = activeSectionIdx === sec.index;
          return (
            <button
              key={sec.id}
              onClick={() => scrollToSection(sec.id)}
              className="group relative flex items-center justify-center w-6 h-6"
              title={`${sec.code} — ${sec.title}`}
            >
              <span
                className={`block rounded-full transition-all duration-300 ${
                  active
                    ? 'w-3 h-3 bg-[#C6FF00] shadow-[0_0_12px_#C6FF00]'
                    : 'w-1.5 h-1.5 bg-[#245BFF]/70 group-hover:bg-[#F4F7FF]'
                }`}
              />
              <span className="pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity absolute right-8 px-2 py-0.5 rounded bg-[#0A1020] border border-[#245BFF]/40 text-[10px] font-mono whitespace-nowrap text-[#F4F7FF]">
                {sec.code} {sec.label}
              </span>
            </button>
          );
        })}
      </aside>

      {/* ====================================================================
          FLOATING TELEMETRY BAR (BOTTOM HUD)
      ==================================================================== */}
      <div className="fixed bottom-3 left-3 right-3 sm:left-6 sm:right-6 z-30 pointer-events-none flex flex-wrap items-center justify-between gap-2">
        {/* Bottom-Left Mission Status Pill */}
        <div className="pointer-events-auto flex items-center gap-3 px-3.5 py-2 rounded-xl border border-[#245BFF]/45 bg-[#0A1020]/85 backdrop-blur-md shadow-[0_0_25px_rgba(5,7,13,0.9)]">
          <span className="flex items-center gap-1.5 text-[11px] font-mono font-bold text-[#C6FF00]">
            <span className="w-2 h-2 rounded-full bg-[#C6FF00] animate-ping" />
            ORBIT LINK: ACTIVE
          </span>
          <span className="text-[#8B95A7] text-xs">|</span>
          <span className="text-[11px] font-mono text-[#F4F7FF]">
            STAGE:{' '}
            <strong className="text-[#245BFF]">
              {MISSION_SECTIONS[activeSectionIdx]?.code}{' '}
              {MISSION_SECTIONS[activeSectionIdx]?.label}
            </strong>
          </span>
          {!introComplete && (
            <>
              <span className="text-[#8B95A7] text-xs hidden sm:inline">|</span>
              <span className="hidden sm:inline text-[10px] font-mono text-[#C6FF00]">
                {currentIntroStep.code}: {currentIntroStep.name} (
                {Math.round(introProgress * 100)}%)
              </span>
            </>
          )}
        </div>

        {/* Bottom-Right Live Coordinates Pill */}
        <div className="pointer-events-auto flex items-center gap-3 px-3.5 py-2 rounded-xl border border-[#245BFF]/45 bg-[#0A1020]/85 backdrop-blur-md shadow-[0_0_25px_rgba(5,7,13,0.9)] font-mono text-[11px]">
          <span>
            <span className="text-[#8B95A7]">LAT:</span>{' '}
            <strong className="text-[#F4F7FF]">{telemetry.lat}° N</strong>
          </span>
          <span>
            <span className="text-[#8B95A7]">LON:</span>{' '}
            <strong className="text-[#F4F7FF]">{telemetry.lon}° E</strong>
          </span>
          <span className="hidden sm:inline">
            <span className="text-[#8B95A7]">ALT:</span>{' '}
            <strong className="text-[#C6FF00]">{telemetry.alt} KM</strong>
          </span>
        </div>
      </div>

      {/* ====================================================================
          SCROLLABLE 8-SECTION CONTINUOUS MISSION DOCUMENT
      ==================================================================== */}
      <div className="relative z-10">
        {/* ==================================================================
            SECTION 1 — HERO (EARTH RIGHT, SATELLITE LEFT, WHITE LASER LINK)
        ================================================================== */}
        <section
          id="sec-hero"
          className="min-h-screen flex items-center pt-24 pb-20 px-5 sm:px-12 lg:px-20"
        >
          <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            <div className="lg:col-span-7 space-y-5">
              {/* Status Badge */}
              <div className="inline-flex flex-wrap items-center gap-2.5 px-3.5 py-1.5 rounded-md border border-[#245BFF]/60 bg-[#0A1020]/85 backdrop-blur-md shadow-[0_0_20px_rgba(36,91,255,0.3)]">
                <span className="flex items-center gap-1.5 text-[11px] font-mono font-bold tracking-wider text-[#C6FF00]">
                  <span className="w-2 h-2 rounded-full bg-[#C6FF00] shadow-[0_0_8px_#C6FF00] animate-ping" />
                  ● SATELLITE ONLINE
                </span>
                <span className="text-[#8B95A7]">•</span>
                <span className="text-[11px] font-mono tracking-[0.2em] text-[#F4F7FF]">
                  WHITE OPTICAL LASER LOCK: INDIA
                </span>
              </div>

              {/* Hero Heading */}
              <h1
                className="text-4xl sm:text-6xl xl:text-7xl font-black tracking-tight leading-[1.02]"
                style={{ textShadow: '0 8px 40px rgba(5, 7, 13, 0.95)' }}
              >
                <span className="block text-[#F4F7FF]">BEYOND EARTH.</span>
                <span className="block bg-clip-text text-transparent bg-gradient-to-r from-[#245BFF] via-[#F4F7FF] to-[#C6FF00]">
                  BEYOND LIMITS.
                </span>
              </h1>

              {/* Mission Tagline */}
              <div className="inline-block px-3.5 py-1.5 rounded border-l-4 border-[#C6FF00] bg-[#0A1020]/90 font-mono text-xs sm:text-sm tracking-[0.16em] text-[#C6FF00] font-bold">
                DETECT THE FLOOD. PRIORITIZE THE DATA. TRANSMIT WHAT MATTERS.
              </div>

              <p className="text-sm sm:text-base lg:text-lg text-[#8B95A7] max-w-xl leading-relaxed bg-[#05070D]/65 backdrop-blur-md p-4 rounded-xl border border-[#245BFF]/25">
                Autonomous onboard flood intelligence powered by dual Sentinel
                SAR/Optical fusion, 512×512 tile prioritization, conformal risk
                calibration, and high-reliability white optical laser downlink.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-wrap items-center gap-3.5 pt-1">
                <button
                  onClick={() => scrollToSection('sec-mission')}
                  className="group flex items-center gap-2.5 px-6 py-3.5 rounded-lg font-mono text-xs sm:text-sm font-bold tracking-[0.16em] text-[#F4F7FF] bg-[#245BFF] hover:bg-[#1947D6] shadow-[0_0_28px_rgba(36,91,255,0.65)] transition-all"
                >
                  <span>EXPLORE THE MISSION</span>
                  <ChevronRight className="w-4 h-4 text-[#C6FF00] group-hover:translate-x-1 transition-transform" />
                </button>

                <button
                  onClick={() => scrollToSection('sec-flood')}
                  className="flex items-center gap-2 px-5 py-3.5 rounded-lg font-mono text-xs sm:text-sm font-bold tracking-[0.15em] border border-[#C6FF00]/70 bg-[#C6FF00]/10 text-[#C6FF00] hover:bg-[#C6FF00]/20 backdrop-blur-md transition-all"
                >
                  <Zap className="w-4 h-4" />
                  <span>LAUNCH 4×4 FLOOD SIM</span>
                </button>
              </div>
            </div>

            {/* Right Column Spacer so 3D Earth & White Laser Beam shine unobscured */}
            <div className="lg:col-span-5 flex flex-col items-end justify-end">
              <div className="w-full max-w-xs p-4 rounded-xl border border-[#245BFF]/40 bg-[#0A1020]/75 backdrop-blur-md space-y-2.5 font-mono text-xs">
                <div className="flex items-center justify-between border-b border-[#245BFF]/25 pb-2">
                  <span className="text-[#C6FF00] font-bold">OPTICAL BEAM SPEC</span>
                  <span className="px-1.5 py-0.5 rounded bg-white/10 text-[#FFFFFF] text-[10px]">
                    COHERENT WHITE
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#8B95A7]">BEAM APERTURE:</span>
                  <span className="text-[#F4F7FF]">0.0045 rad ULTRA-THIN</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#8B95A7]">DOWNLINK RATE:</span>
                  <span className="text-[#C6FF00] font-bold">{telemetry.downlinkGbps} Gbps</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#8B95A7]">SCROLL TRAVEL:</span>
                  <span className="text-[#245BFF] font-bold">ACTIVE SPLINE</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================
            SECTION 2 — MISSION INTRODUCTION (EARTH SHIFTS TO CENTRE & SCALES UP)
        ================================================================== */}
        <section
          id="sec-mission"
          className="min-h-screen flex items-center py-24 px-5 sm:px-12 lg:px-20"
        >
          <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Left Editorial Mission Architecture */}
            <div className="lg:col-span-5 p-6 sm:p-8 rounded-2xl border border-[#245BFF]/45 bg-[#0A1020]/82 backdrop-blur-xl shadow-[0_0_50px_rgba(5,7,13,0.9)] space-y-4">
              <div className="flex items-center gap-2 text-xs font-mono text-[#C6FF00] tracking-[0.2em]">
                <Sparkles className="w-4 h-4" />
                <span>02 // ORBIT-PRISM MISSION</span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-[#F4F7FF]">
                Edge-First Orbital Triage Before Downlink
              </h2>
              <p className="text-sm sm:text-base text-[#8B95A7] leading-relaxed">
                Conventional Earth observation satellites waste up to 82% of
                downlink bandwidth transmitting cloud-obscured or non-critical
                ocean frames. <strong className="text-[#F4F7FF]">ORBIT-PRISM</strong>{' '}
                processes raw multi-spectral and radar swaths directly in orbit.
              </p>
              <div className="pt-2 space-y-2.5 font-mono text-xs">
                <div className="p-3 rounded-lg bg-[#05070D]/80 border border-[#245BFF]/30 flex items-center justify-between">
                  <span className="text-[#8B95A7]">01. ONBOARD SEGMENTATION</span>
                  <span className="text-[#C6FF00] font-bold">&lt; 38 ms / chip</span>
                </div>
                <div className="p-3 rounded-lg bg-[#05070D]/80 border border-[#245BFF]/30 flex items-center justify-between">
                  <span className="text-[#8B95A7]">02. PRIORITY QUEUEING</span>
                  <span className="text-[#245BFF] font-bold">L2 → L1 → L0</span>
                </div>
                <div className="p-3 rounded-lg bg-[#05070D]/80 border border-[#245BFF]/30 flex items-center justify-between">
                  <span className="text-[#8B95A7]">03. RISK CALIBRATION</span>
                  <span className="text-[#C6FF00] font-bold">CONFORMAL GUARANTEED</span>
                </div>
              </div>
            </div>

            {/* Center Viewport Clear for 3D Earth in Centre */}
            <div className="hidden lg:block lg:col-span-2" />

            {/* Right Telemetry Architecture Metrics */}
            <div className="lg:col-span-5 space-y-4">
              <div className="p-6 rounded-2xl border border-[#C6FF00]/45 bg-[#0A1020]/82 backdrop-blur-xl">
                <div className="text-xs font-mono text-[#C6FF00] tracking-widest mb-1">
                  CRITICAL IMPACT REDUCTION
                </div>
                <div className="text-4xl sm:text-5xl font-black text-[#F4F7FF] mb-2">
                  4.8× <span className="text-xl font-mono text-[#C6FF00]">FASTER ALERT</span>
                </div>
                <p className="text-xs sm:text-sm text-[#8B95A7]">
                  By discarding cloud-blocked L0 frames and laser-transmitting L2
                  flood tiles first, emergency response teams in Assam, Bihar,
                  and Odisha receive actionable flood perimeters in under 90
                  seconds.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-5 rounded-xl border border-[#245BFF]/40 bg-[#0A1020]/80 backdrop-blur-md">
                  <div className="text-[11px] font-mono text-[#8B95A7]">
                    SWATH FOOTPRINT
                  </div>
                  <div className="text-2xl font-bold text-[#245BFF] mt-1">
                    290 KM
                  </div>
                  <div className="text-[10px] font-mono text-[#F4F7FF]/70 mt-1">
                    Sun-Synchronous LEO
                  </div>
                </div>
                <div className="p-5 rounded-xl border border-[#245BFF]/40 bg-[#0A1020]/80 backdrop-blur-md">
                  <div className="text-[11px] font-mono text-[#8B95A7]">
                    GROUND RESOLUTION
                  </div>
                  <div className="text-2xl font-bold text-[#C6FF00] mt-1">
                    10.0 M / PX
                  </div>
                  <div className="text-[10px] font-mono text-[#F4F7FF]/70 mt-1">
                    512×512 Chip Tiling
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================
            SECTION 3 — EARTH OBSERVATION (EARTH MOVES LEFT, RIGHT PANEL CONTENT)
        ================================================================== */}
        <section
          id="sec-observation"
          className="min-h-screen flex items-center py-24 px-5 sm:px-12 lg:px-20"
        >
          <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Left Side Open so 3D Earth on the Left is Completely Visible */}
            <div className="lg:col-span-5 flex flex-col justify-end">
              <div className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg border border-[#245BFF]/45 bg-[#05070D]/80 backdrop-blur-md font-mono text-xs text-[#F4F7FF] w-fit">
                <Compass className="w-4 h-4 text-[#C6FF00]" />
                <span>3D EARTH POSITION: LEFT ORBITAL CORRIDOR</span>
              </div>
            </div>

            {/* Right Editorial Observation Panel */}
            <div className="lg:col-span-7 p-6 sm:p-9 rounded-2xl border border-[#245BFF]/50 bg-[#0A1020]/85 backdrop-blur-xl shadow-[0_0_50px_rgba(5,7,13,0.92)] space-y-5">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <span className="text-xs font-mono text-[#C6FF00] tracking-[0.2em]">
                  03 // PLANETARY OBSERVATION CORRIDOR
                </span>
                <span className="text-xs font-mono px-2.5 py-0.5 rounded bg-[#245BFF]/20 text-[#245BFF] border border-[#245BFF]/40">
                  AOI: EASTERN INDIA DELTA
                </span>
              </div>

              <h2 className="text-3xl sm:text-5xl font-black tracking-tight text-[#F4F7FF]">
                Continuous Subcontinent Horizon Scanning
              </h2>

              <p className="text-sm sm:text-base text-[#8B95A7] leading-relaxed">
                As the orbital camera sweeps around the planetary horizon,
                ORBIT-PRISM locks onto high-vulnerability river basins across
                the Indian subcontinent—monitoring the Brahmaputra, Ganga, and
                Mahanadi floodplains in real time.
              </p>

              {/* Live Basin Telemetry Table */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5 pt-2 font-mono text-xs">
                {[
                  {
                    basin: 'BRAHMAPUTRA VALLEY',
                    coords: '26.14°N, 91.73°E',
                    discharge: '+14,200 m³/s',
                    status: 'CRITICAL SURGE',
                    crit: true,
                  },
                  {
                    basin: 'MAHANADI DELTA',
                    coords: '20.29°N, 85.82°E',
                    discharge: '+8,940 m³/s',
                    status: 'HIGH WATCH',
                    crit: true,
                  },
                  {
                    basin: 'GODAVARI ESTUARY',
                    coords: '16.98°N, 81.78°E',
                    discharge: '+3,120 m³/s',
                    status: 'NOMINAL FLOW',
                    crit: false,
                  },
                ].map((b) => (
                  <div
                    key={b.basin}
                    className={`p-4 rounded-xl border ${
                      b.crit
                        ? 'border-[#C6FF00]/60 bg-[#05070D]/90'
                        : 'border-[#245BFF]/35 bg-[#05070D]/75'
                    }`}
                  >
                    <div className="text-[10px] text-[#8B95A7]">{b.coords}</div>
                    <div className="font-bold text-[#F4F7FF] mt-1">{b.basin}</div>
                    <div className="text-sm font-bold text-[#245BFF] mt-2">
                      {b.discharge}
                    </div>
                    <div
                      className={`text-[10px] font-bold mt-1 ${
                        b.crit ? 'text-[#C6FF00]' : 'text-[#8B95A7]'
                      }`}
                    >
                      ● {b.status}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================
            SECTION 4 — FLOOD DETECTION (512x512 IMAGE CHIP & 4x4 ACID LIME GRID)
        ================================================================== */}
        <section
          id="sec-flood"
          className="min-h-screen flex items-center py-24 px-5 sm:px-12 lg:px-20"
        >
          <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Left Side: 3D Earth Zoomed In with Synchronized Surface Grid */}
            <div className="lg:col-span-5 space-y-4">
              <div className="p-6 rounded-2xl border border-[#C6FF00]/50 bg-[#0A1020]/85 backdrop-blur-xl space-y-3">
                <div className="flex items-center gap-2 text-xs font-mono text-[#C6FF00] tracking-widest">
                  <Crosshair className="w-4 h-4" />
                  <span>04 // 512×512 ONBOARD CHIP SEGMENTATION</span>
                </div>
                <h2 className="text-2xl sm:text-4xl font-black text-[#F4F7FF]">
                  4×4 Tile Triage & Acid-Lime Flood Lock
                </h2>
                <p className="text-xs sm:text-sm text-[#8B95A7] leading-relaxed">
                  Each 512×512 satellite image chip is partitioned into sixteen
                  128×128 sub-tiles. Tiles exceeding the conformal flood
                  inundation threshold illuminate in{' '}
                  <strong className="text-[#C6FF00]">Acid Lime (#C6FF00)</strong>{' '}
                  and are promoted to the L2 laser downlink queue.
                </p>

                {/* Selected Tile Inspector Box */}
                <div className="p-4 rounded-xl bg-[#05070D] border border-[#245BFF]/40 font-mono text-xs space-y-1.5">
                  <div className="flex items-center justify-between border-b border-white/10 pb-1.5">
                    <span className="text-[#8B95A7]">INSPECTED TILE:</span>
                    <span className="font-bold text-[#C6FF00]">
                      {selectedTileObj.coord} ({selectedTileObj.pxBounds})
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8B95A7]">INUNDATION RATIO:</span>
                    <span className="text-[#F4F7FF] font-bold">
                      {selectedTileObj.waterPct}% FLOOD WATER
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8B95A7]">DOWNLINK TIER:</span>
                    <span
                      className={`font-bold ${
                        selectedTileObj.isCritical
                          ? 'text-[#C6FF00]'
                          : selectedTileObj.isModerate
                          ? 'text-[#245BFF]'
                          : 'text-[#8B95A7]'
                      }`}
                    >
                      {selectedTileObj.priority}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8B95A7]">POSTERIOR CONFIDENCE:</span>
                    <span className="text-[#F4F7FF]">{selectedTileObj.confidence}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Side: Interactive 512x512 Satellite Image Chip with 4x4 Grid */}
            <div className="lg:col-span-7 flex justify-center">
              <div className="w-full max-w-xl p-5 sm:p-6 rounded-2xl border border-[#245BFF]/55 bg-[#0A1020]/90 backdrop-blur-xl shadow-[0_0_60px_rgba(36,91,255,0.28)]">
                <div className="flex items-center justify-between mb-3 font-mono text-xs">
                  <span className="text-[#F4F7FF] font-bold">
                    CHIP ID: IND-BHM-512×512
                  </span>
                  <span className="text-[#C6FF00]">
                    ● 4 / 16 CRITICAL FLOOD TILES
                  </span>
                </div>

                {/* 512x512 Visual Satellite River Basin + 4x4 Interactive Overlay */}
                <div className="relative aspect-square w-full rounded-xl overflow-hidden border border-[#245BFF]/50 bg-[#05070D]">
                  {/* Synthetic SAR + Optical Flooded River Channel SVG Backdrop */}
                  <svg
                    viewBox="0 0 512 512"
                    className="absolute inset-0 w-full h-full object-cover opacity-80"
                  >
                    <defs>
                      <radialGradient id="basinGlow" cx="50%" cy="50%" r="55%">
                        <stop offset="0%" stopColor="#14326E" stopOpacity="0.9" />
                        <stop offset="100%" stopColor="#05070D" stopOpacity="1" />
                      </radialGradient>
                    </defs>
                    <rect width="512" height="512" fill="url(#basinGlow)" />
                    {/* Riverbed & Flooded Inundation Plume */}
                    <path
                      d="M 40,90 Q 190,140 245,245 T 470,410"
                      fill="none"
                      stroke="#245BFF"
                      strokeWidth="95"
                      strokeLinecap="round"
                      opacity="0.42"
                    />
                    <path
                      d="M 40,90 Q 190,140 245,245 T 470,410"
                      fill="none"
                      stroke="#C6FF00"
                      strokeWidth="38"
                      strokeLinecap="round"
                      opacity="0.75"
                    />
                    {/* Tributary branches */}
                    <path
                      d="M 340,70 Q 280,190 250,255"
                      fill="none"
                      stroke="#C6FF00"
                      strokeWidth="22"
                      opacity="0.65"
                    />
                    <path
                      d="M 140,390 Q 220,310 265,265"
                      fill="none"
                      stroke="#245BFF"
                      strokeWidth="28"
                      opacity="0.55"
                    />
                  </svg>

                  {/* 4x4 Interactive Tile Grid */}
                  <div className="relative z-10 grid grid-cols-4 grid-rows-4 w-full h-full">
                    {FLOOD_TILES.map((tile) => {
                      const isSelected = tile.id === selectedTileId;
                      return (
                        <button
                          key={tile.id}
                          onClick={() => setSelectedTileId(tile.id)}
                          className={`group relative flex flex-col justify-between p-2 text-left font-mono transition-all border ${
                            tile.isCritical
                              ? 'border-[#C6FF00] bg-[#C6FF00]/22 hover:bg-[#C6FF00]/35 shadow-[inset_0_0_20px_rgba(198,255,0,0.45)]'
                              : tile.isModerate
                              ? 'border-[#245BFF]/60 bg-[#245BFF]/15 hover:bg-[#245BFF]/25'
                              : 'border-white/10 bg-black/25 hover:bg-white/5'
                          } ${
                            isSelected
                              ? 'ring-2 ring-white z-20 scale-[0.98]'
                              : ''
                          }`}
                        >
                          <div className="flex items-center justify-between w-full">
                            <span
                              className={`text-[10px] font-bold ${
                                tile.isCritical
                                  ? 'text-[#C6FF00]'
                                  : 'text-[#F4F7FF]/80'
                              }`}
                            >
                              {tile.coord}
                            </span>
                            {tile.isCritical && (
                              <span className="w-2 h-2 rounded-full bg-[#C6FF00] animate-ping" />
                            )}
                          </div>

                          <div>
                            <div
                              className={`text-xs sm:text-sm font-extrabold ${
                                tile.isCritical
                                  ? 'text-[#C6FF00]'
                                  : tile.isModerate
                                  ? 'text-[#245BFF]'
                                  : 'text-[#8B95A7]'
                              }`}
                            >
                              {tile.waterPct}%
                            </div>
                            <div className="text-[9px] text-[#F4F7FF]/75 hidden sm:block">
                              {tile.priority}
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Legend */}
                <div className="flex flex-wrap items-center justify-between gap-2 mt-3 text-[11px] font-mono text-[#8B95A7]">
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-sm bg-[#C6FF00] inline-block" />
                    <strong className="text-[#C6FF00]">L2 CRITICAL FLOOD</strong>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-sm bg-[#245BFF] inline-block" />
                    <strong className="text-[#245BFF]">L1 BUFFER</strong>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-sm bg-white/20 inline-block" />
                    <span>L0 ARCHIVE (SKIPPED)</span>
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================
            SECTION 5 — OPTICAL & RADAR INTELLIGENCE (EARTH MOVES RIGHT)
        ================================================================== */}
        <section
          id="sec-sensors"
          className="min-h-screen flex items-center py-24 px-5 sm:px-12 lg:px-20"
        >
          <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Left Side: Interactive Sentinel-2 Optical vs Sentinel-1 SAR Controller */}
            <div className="lg:col-span-7 p-6 sm:p-8 rounded-2xl border border-[#245BFF]/50 bg-[#0A1020]/88 backdrop-blur-xl shadow-[0_0_55px_rgba(5,7,13,0.92)] space-y-5">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <span className="text-xs font-mono text-[#C6FF00] tracking-[0.2em]">
                  05 // OPTICAL & RADAR INTELLIGENCE
                </span>
                <span className="text-xs font-mono px-2.5 py-0.5 rounded bg-[#C6FF00]/15 text-[#C6FF00] border border-[#C6FF00]/40">
                  ACTIVE SENSOR: {effectiveSensor === 'SAR' ? 'SENTINEL-1 SAR C-BAND' : 'SENTINEL-2 MSI OPTICAL'}
                </span>
              </div>

              <h2 className="text-3xl sm:text-4xl font-black text-[#F4F7FF]">
                Autonomous Cloud-Cover Detection & SAR Handover
              </h2>

              <p className="text-sm sm:text-base text-[#8B95A7] leading-relaxed">
                Monsoon floods are almost always accompanied by dense cumulonimbus
                cloud cover. When onboard cloud-masking detects greater than{' '}
                <strong className="text-[#C6FF00]">30% opacity</strong>,
                ORBIT-PRISM automatically switches inference from Sentinel-2
                Optical to <strong className="text-[#245BFF]">Sentinel-1 C-Band SAR</strong>{' '}
                microwave backscatter, penetrating clouds without interruption.
              </p>

              {/* Interactive Cloud Cover Slider */}
              <div className="p-4 rounded-xl bg-[#05070D] border border-[#245BFF]/40 space-y-3">
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="flex items-center gap-1.5 text-[#F4F7FF]">
                    <CloudRain className="w-4 h-4 text-[#245BFF]" />
                    MONSOON CLOUD COVER SIMULATION:
                  </span>
                  <span
                    className={`font-bold text-sm ${
                      cloudCoverPct > 30 ? 'text-[#C6FF00]' : 'text-[#245BFF]'
                    }`}
                  >
                    {cloudCoverPct}% OPACITY{' '}
                    {cloudCoverPct > 30
                      ? '(SAR TRIGGERED)'
                      : '(OPTICAL CLEAR)'}
                  </span>
                </div>

                <input
                  type="range"
                  min="0"
                  max="100"
                  value={cloudCoverPct}
                  onChange={(e) => setCloudCoverPct(Number(e.target.value))}
                  className="w-full h-2 bg-[#0A1020] rounded-lg appearance-none cursor-pointer accent-[#C6FF00]"
                />

                <div className="flex justify-between text-[10px] font-mono text-[#8B95A7]">
                  <span>0% CLEAR SKY</span>
                  <span className="text-[#C6FF00]">30% AUTO-SAR THRESHOLD</span>
                  <span>100% CYCLONE OVERCAST</span>
                </div>
              </div>

              {/* Sensor Mode Selector Buttons */}
              <div className="grid grid-cols-3 gap-3 font-mono text-xs">
                {[
                  { id: 'AUTO', label: 'AUTO FUSION LOGIC' },
                  { id: 'OPTICAL', label: 'SENTINEL-2 OPTICAL' },
                  { id: 'SAR', label: 'SENTINEL-1 SAR' },
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setSensorMode(m.id)}
                    className={`py-3 px-3 rounded-lg border font-bold tracking-wider transition-all ${
                      sensorMode === m.id
                        ? 'border-[#C6FF00] bg-[#C6FF00]/15 text-[#C6FF00]'
                        : 'border-[#245BFF]/35 bg-[#05070D]/70 text-[#8B95A7] hover:text-[#F4F7FF]'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              {/* Sensor Comparison Readout */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 font-mono text-xs pt-1">
                <div
                  className={`p-4 rounded-xl border ${
                    effectiveSensor === 'OPTICAL'
                      ? 'border-[#C6FF00] bg-[#05070D]'
                      : 'border-white/10 bg-[#05070D]/50 opacity-75'
                  }`}
                >
                  <div className="text-[#245BFF] font-bold mb-1">
                    SENTINEL-2 MSI (OPTICAL)
                  </div>
                  <div className="text-[#8B95A7] text-[11px]">
                    Bands: B03 (Green), B08 (NIR), NDWI Water Index. High spectral
                    clarity in <strong className="text-[#F4F7FF]">&lt;30% cloud</strong> conditions.
                  </div>
                </div>

                <div
                  className={`p-4 rounded-xl border ${
                    effectiveSensor === 'SAR'
                      ? 'border-[#C6FF00] bg-[#05070D]'
                      : 'border-white/10 bg-[#05070D]/50 opacity-75'
                  }`}
                >
                  <div className="text-[#C6FF00] font-bold mb-1">
                    SENTINEL-1 SAR (5.405 GHz C-BAND)
                  </div>
                  <div className="text-[#8B95A7] text-[11px]">
                    Dual-polarization VV/VH specular water return.{' '}
                    <strong className="text-[#F4F7FF]">100% cloud & night penetration</strong>{' '}
                    across cyclone eyes.
                  </div>
                </div>
              </div>
            </div>

            {/* Right Side Open so 3D Earth on the Right is Clearly Visible */}
            <div className="lg:col-span-5" />
          </div>
        </section>

        {/* ==================================================================
            SECTION 6 — DATA TRANSMISSION (L0, L1, L2 PACKETS & LASER ARCS)
        ================================================================== */}
        <section
          id="sec-downlink"
          className="min-h-screen flex items-center py-24 px-5 sm:px-12 lg:px-20"
        >
          <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            <div className="lg:col-span-6 p-6 sm:p-8 rounded-2xl border border-[#245BFF]/50 bg-[#0A1020]/88 backdrop-blur-xl space-y-5">
              <div className="flex items-center gap-2 text-xs font-mono text-[#C6FF00] tracking-[0.2em]">
                <Wifi className="w-4 h-4" />
                <span>06 // PRIORITY OPTICAL DATA TRANSMISSION</span>
              </div>

              <h2 className="text-3xl sm:text-4xl font-black text-[#F4F7FF]">
                L2 Acid-Lime Priority Packets Over Coherent White Laser
              </h2>

              <p className="text-sm sm:text-base text-[#8B95A7] leading-relaxed">
                As the camera pulls into high orbital perspective, observe the
                live optical downlink between the satellite and the ISTRAC ground
                station in India. Data is strictly ordered into three telemetry
                tiers:
              </p>

              {/* Filterable Packet Tiers */}
              <div className="flex flex-wrap gap-2 font-mono text-xs">
                {['ALL', 'L2', 'L1', 'L0'].map((tier) => (
                  <button
                    key={tier}
                    onClick={() => setActivePacketTier(tier)}
                    className={`px-3.5 py-1.5 rounded-lg border font-bold transition-all ${
                      activePacketTier === tier
                        ? 'border-[#C6FF00] bg-[#C6FF00] text-[#05070D]'
                        : 'border-[#245BFF]/40 bg-[#05070D] text-[#F4F7FF]'
                    }`}
                  >
                    {tier === 'ALL' ? 'ALL TIERS' : `${tier} STREAM`}
                  </button>
                ))}
              </div>

              <div className="space-y-3 font-mono text-xs">
                {(activePacketTier === 'ALL' || activePacketTier === 'L2') && (
                  <div className="p-4 rounded-xl border-2 border-[#C6FF00] bg-[#05070D]/95 shadow-[0_0_25px_rgba(198,255,0,0.2)]">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-[#C6FF00] text-sm">
                        ● L2 CRITICAL FLOOD MASK + VECTOR POLYGONS
                      </span>
                      <span className="px-2 py-0.5 rounded bg-[#C6FF00] text-[#05070D] font-bold text-[10px]">
                        LATENCY: 0.42s
                      </span>
                    </div>
                    <p className="text-[#8B95A7] text-[11px]">
                      Transmitted immediately over the razor-thin white optical
                      laser link. Contains binary flood masks and geo-referenced
                      inundation boundaries for tiles T-11, T-12, T-21, T-22.
                    </p>
                  </div>
                )}

                {(activePacketTier === 'ALL' || activePacketTier === 'L1') && (
                  <div className="p-4 rounded-xl border border-[#245BFF] bg-[#05070D]/85">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-[#245BFF] text-sm">
                        ● L1 RADIOMETRICALLY CALIBRATED SUB-SWATH
                      </span>
                      <span className="px-2 py-0.5 rounded bg-[#245BFF]/20 text-[#245BFF] font-bold text-[10px]">
                        LATENCY: 8.4s
                      </span>
                    </div>
                    <p className="text-[#8B95A7] text-[11px]">
                      Secondary Electric Cobalt telemetry stream carrying
                      moderate-risk riverine buffer tiles for hydrological model
                      assimilation.
                    </p>
                  </div>
                )}

                {(activePacketTier === 'ALL' || activePacketTier === 'L0') && (
                  <div className="p-4 rounded-xl border border-white/15 bg-[#05070D]/60">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-[#8B95A7] text-sm">
                        ○ L0 RAW UNCOMPRESSED SENSOR ARCHIVE
                      </span>
                      <span className="px-2 py-0.5 rounded bg-white/10 text-[#8B95A7] text-[10px]">
                        DEFERRED / COMPRESSED
                      </span>
                    </div>
                    <p className="text-[#8B95A7] text-[11px]">
                      Non-flooded background and cloud-blocked tiles retained in
                      onboard NVMe storage for off-peak Ka-band dump.
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="lg:col-span-6" />
          </div>
        </section>

        {/* ==================================================================
            SECTION 7 — CALIBRATION & INTELLIGENCE (1%, 5%, 10% MISS-RATE TARGETS)
        ================================================================== */}
        <section
          id="sec-calibration"
          className="min-h-screen flex items-center py-24 px-5 sm:px-12 lg:px-20"
        >
          <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            <div className="lg:col-span-5" />

            <div className="lg:col-span-7 p-6 sm:p-9 rounded-2xl border border-[#245BFF]/50 bg-[#0A1020]/90 backdrop-blur-xl shadow-[0_0_60px_rgba(5,7,13,0.95)] space-y-5">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <span className="text-xs font-mono text-[#C6FF00] tracking-[0.2em]">
                  07 // CONFORMAL RISK CONTROL & CALIBRATION
                </span>
                <span className="text-xs font-mono px-2.5 py-0.5 rounded bg-[#C6FF00]/15 text-[#C6FF00] border border-[#C6FF00]/40">
                  VERIFIED EVALUATION DATASET: SEN12FLOODS
                </span>
              </div>

              <h2 className="text-3xl sm:text-4xl font-black text-[#F4F7FF]">
                Mathematically Bounded Flood Miss-Rate Targets
              </h2>

              <p className="text-sm sm:text-base text-[#8B95A7] leading-relaxed">
                Un-calibrated neural networks can silently miss inundated
                villages under domain shift. ORBIT-PRISM applies{' '}
                <strong className="text-[#F4F7FF]">Conformal Risk Control</strong>{' '}
                to calibrate the decision threshold <code className="text-[#C6FF00]">τ</code>{' '}
                against strict user-selected miss-rate bounds (<strong className="text-[#C6FF00]">1%, 5%, and 10%</strong>).
              </p>

              {/* Miss-Rate Target Selector (1%, 5%, 10%) */}
              <div className="grid grid-cols-3 gap-3 font-mono">
                {(['1%', '5%', '10%']).map((target) => {
                  const active = calibTarget === target;
                  return (
                    <button
                      key={target}
                      onClick={() => setCalibTarget(target)}
                      className={`p-3.5 rounded-xl border text-left transition-all ${
                        active
                          ? 'border-[#C6FF00] bg-[#C6FF00]/15 shadow-[0_0_20px_rgba(198,255,0,0.25)]'
                          : 'border-[#245BFF]/35 bg-[#05070D]/80 hover:border-[#245BFF]'
                      }`}
                    >
                      <div className="text-[10px] text-[#8B95A7]">
                        TARGET MISS RATE
                      </div>
                      <div
                        className={`text-2xl font-black mt-0.5 ${
                          active ? 'text-[#C6FF00]' : 'text-[#F4F7FF]'
                        }`}
                      >
                        α = {target}
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Empirical Calibration Results Card */}
              <div className="p-5 rounded-xl bg-[#05070D] border border-[#245BFF]/45 space-y-4">
                <p className="text-xs sm:text-sm text-[#F4F7FF]/90 leading-relaxed">
                  {activeCalibData.desc}
                </p>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
                  <div className="p-3 rounded-lg bg-[#0A1020] border border-white/10">
                    <div className="text-[10px] text-[#8B95A7]">TARGET α</div>
                    <div className="text-base font-bold text-[#F4F7FF] mt-1">
                      {activeCalibData.alpha}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-[#0A1020] border border-[#C6FF00]/40">
                    <div className="text-[10px] text-[#8B95A7]">EMPIRICAL MISS</div>
                    <div className="text-base font-bold text-[#C6FF00] mt-1">
                      {activeCalibData.empiricalMiss}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-[#0A1020] border border-white/10">
                    <div className="text-[10px] text-[#8B95A7]">THRESHOLD τ</div>
                    <div className="text-base font-bold text-[#245BFF] mt-1">
                      {activeCalibData.thresholdTau}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-[#0A1020] border border-white/10">
                    <div className="text-[10px] text-[#8B95A7]">BANDWIDTH SAVED</div>
                    <div className="text-base font-bold text-[#C6FF00] mt-1">
                      {activeCalibData.bandwidthSaved}
                    </div>
                  </div>
                </div>

                {/* Recall Stability Bar Visualization */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex justify-between text-[10px] font-mono text-[#8B95A7]">
                    <span>EMPIRICAL FLOOD RECALL ACROSS 6 VALIDATION PASSES</span>
                    <span className="text-[#C6FF00] font-bold">
                      GUARANTEED ≥ {100 - parseInt(calibTarget, 10)}%
                    </span>
                  </div>
                  <div className="grid grid-cols-6 gap-2 pt-1">
                    {activeCalibData.curvePoints.map((val, i) => (
                      <div key={i} className="bg-[#0A1020] p-2 rounded border border-white/5 text-center font-mono">
                        <div className="text-[10px] text-[#8B95A7]">PASS 0{i + 1}</div>
                        <div className="text-xs font-bold text-[#C6FF00] mt-0.5">
                          {val}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================
            SECTION 8 — FINAL MISSION SUMMARY & FOOTER (EARTH CONTINUOUS IN ORBIT)
        ================================================================== */}
        <footer
          id="sec-footer"
          className="min-h-[85vh] flex flex-col justify-between pt-24 pb-20 px-5 sm:px-12 lg:px-20 border-t border-[#245BFF]/30 bg-gradient-to-t from-[#05070D] via-[#05070D]/80 to-transparent"
        >
          <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center my-auto">
            <div className="lg:col-span-7 space-y-5">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded border border-[#C6FF00]/50 bg-[#C6FF00]/10 font-mono text-xs text-[#C6FF00]">
                <span>08 // CONTINUOUS ORBITAL HORIZON</span>
              </div>

              <h2 className="text-4xl sm:text-6xl font-black tracking-tight text-[#F4F7FF]">
                ORBIT-PRISM IS LIVE IN LOW-EARTH ORBIT.
              </h2>

              <p className="text-base sm:text-lg text-[#8B95A7] max-w-xl leading-relaxed">
                From the opening satellite laser handshake to real-time 512×512
                flood tile triage and conformal calibration, Earth remains in
                continuous orbital motion—ready for the next pass over the
                Indian subcontinent.
              </p>

              <div className="flex flex-wrap items-center gap-4 pt-2">
                <button
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="flex items-center gap-2 px-6 py-3.5 rounded-lg bg-[#245BFF] hover:bg-[#1947D6] text-[#F4F7FF] font-mono text-xs font-bold tracking-[0.16em] shadow-[0_0_25px_rgba(36,91,255,0.6)] transition-all"
                >
                  <ArrowUp className="w-4 h-4 text-[#C6FF00]" />
                  <span>BACK TO TOP ORBIT</span>
                </button>

                <button
                  onClick={handleReplayIntro}
                  className="flex items-center gap-2 px-6 py-3.5 rounded-lg border border-[#C6FF00] bg-[#0A1020]/90 text-[#C6FF00] hover:bg-[#C6FF00]/15 font-mono text-xs font-bold tracking-[0.16em] transition-all"
                >
                  <RotateCcw className="w-4 h-4" />
                  <span>REPLAY SATELLITE LASER INTRO</span>
                </button>
              </div>
            </div>

            <div className="lg:col-span-5" />
          </div>

          {/* Bottom Footer Credits */}
          <div className="max-w-7xl mx-auto w-full pt-12 mt-12 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-4 font-mono text-xs text-[#8B95A7]">
            <div>
              <strong className="text-[#F4F7FF]">ORBIT-PRISM</strong> — DETECT
              THE FLOOD. PRIORITIZE THE DATA. TRANSMIT WHAT MATTERS.
            </div>
            <div>
              PERSISTENT SINGLE-CANVAS THREE.JS • INDIA NADIR LOCK 20.5937°N,
              78.9629°E
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
