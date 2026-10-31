/**
 * Three.js + GSAP ScrollTrigger 3D Orbital Intelligence Scene
 * Continuous Cinematic Earth Journey across the entire mission document
 *
 * Choreography:
 * - Section 1 (Hero): Earth on RIGHT, large & dominant. Satellite on left with white optical laser link.
 * - Section 2 (The Problem): Earth shifts to CENTRE, apparent size increases, camera approaches atmosphere.
 * - Section 3 (How It Works): Earth glides to LEFT, orbital perspective reveals the 10-step streaming pipeline.
 * - Section 4 (Try It): Camera zooms close to delta, 3D 4x4 flood grid illuminates with Acid Lime critical markers.
 * - Section 5 (Link Simulator): Earth shifts to RIGHT, satellite communication beams stream L0/L1/L2 packets.
 * - Section 6 (Results): Camera pulls away, Earth becomes smaller, global telemetry and benchmark tables float cleanly.
 * - Section 7 (Judges Q&A): High-angle orbital view reveals polar calibration ring; defense accordion.
 * - Section 8 (Limits & Footer): Camera pulls into deep space, Earth remains visible in background down to the footer.
 */

(function () {
  'use strict';

  var COLORS = {
    midnightBlack: '#05070D',
    deepNavy: '#0A1020',
    cobaltBlue: '#245BFF',
    cobaltGlow: 'rgba(36, 91, 255, 0.65)',
    acidLime: '#C6FF00',
    white: '#F4F7FF',
    pureWhite: '#FFFFFF',
    mutedGrey: '#8B95A7'
  };

  var INDIA_COORDS = { lat: 20.5937, lon: 78.9629, alt: 408 };

  var INTRO_STEPS = [
    { id: 1, code: 'INTRO 01', name: 'SATELLITE REVEAL', range: [0.0, 0.20] },
    { id: 2, code: 'INTRO 02', name: 'ANTENNA GIMBAL LOCK', range: [0.20, 0.40] },
    { id: 3, code: 'INTRO 03', name: 'WHITE OPTICAL LASER LINK', range: [0.40, 0.62] },
    { id: 4, code: 'INTRO 04', name: 'SUBCONTINENT SCAN', range: [0.62, 0.82] },
    { id: 5, code: 'INTRO 05', name: 'ORBITAL HANDOFF', range: [0.82, 1.0] }
  ];

  var MISSION_SECTIONS = [
    { id: 'top', code: '01', label: 'HERO' },
    { id: 'problem', code: '02', label: 'THE PROBLEM' },
    { id: 'how', code: '03', label: 'HOW IT WORKS' },
    { id: 'try', code: '04', label: 'TRY IT' },
    { id: 'link', code: '05', label: 'LINK SIM' },
    { id: 'results', code: '06', label: 'RESULTS' },
    { id: 'questions', code: '07', label: 'DEFENSE' },
    { id: 'glossary', code: '08', label: 'LIMITS' }
  ];

  var FLOOD_TILES = Array.from({ length: 16 }, function (_, idx) {
    var isCritical = [5, 6, 9, 10].indexOf(idx) !== -1;
    var isModerate = [1, 2, 8, 11, 14].indexOf(idx) !== -1;
    return {
      id: idx,
      isCritical: isCritical,
      isModerate: isModerate
    };
  });

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  function smoothstep(edge0, edge1, x) {
    var t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
  }

  // 1D Catmull-Rom cubic spline interpolation across keyframes
  function sampleSpline(values, t) {
    var n = values.length - 1;
    var scaled = clamp(t, 0, 1) * n;
    var idx = Math.min(Math.floor(scaled), n - 1);
    var frac = scaled - idx;

    var p0 = values[Math.max(0, idx - 1)];
    var p1 = values[idx];
    var p2 = values[Math.min(n, idx + 1)];
    var p3 = values[Math.min(n, idx + 2)];

    var f2 = frac * frac;
    var f3 = f2 * frac;
    return (
      0.5 *
      (2 * p1 +
        (-p0 + p2) * frac +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2 +
        (-p0 + 3 * p1 - 3 * p2 + p3) * f3)
    );
  }

  function latLonToVector3(lat, lon, radius) {
    var phi = (90 - lat) * (Math.PI / 180);
    var theta = (lon + 180) * (Math.PI / 180);
    var x = -(radius * Math.sin(phi) * Math.cos(theta));
    var z = radius * Math.sin(phi) * Math.sin(theta);
    var y = radius * Math.cos(phi);
    return new THREE.Vector3(x, y, z);
  }

  function createKaptonTexture() {
    var canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    var ctx = canvas.getContext('2d');

    var grad = ctx.createLinearGradient(0, 0, 512, 512);
    grad.addColorStop(0, '#D99B26');
    grad.addColorStop(0.5, '#F5BD41');
    grad.addColorStop(1, '#B37412');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 512, 512);

    for (var i = 0; i < 700; i++) {
      var x = Math.random() * 512;
      var y = Math.random() * 512;
      var size = 8 + Math.random() * 24;
      ctx.fillStyle = Math.random() > 0.5 ? 'rgba(255, 235, 160, 0.2)' : 'rgba(90, 48, 4, 0.22)';
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + size * (Math.random() - 0.3), y + size * (Math.random() - 0.3));
      ctx.closePath();
      ctx.fill();
    }

    ctx.strokeStyle = 'rgba(70, 38, 5, 0.45)';
    ctx.lineWidth = 2;
    for (var j = 0; j <= 512; j += 64) {
      ctx.beginPath();
      ctx.moveTo(j, 0);
      ctx.lineTo(j, 512);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, j);
      ctx.lineTo(512, j);
      ctx.stroke();
    }

    var map = new THREE.CanvasTexture(canvas);
    map.wrapS = THREE.RepeatWrapping;
    map.wrapT = THREE.RepeatWrapping;
    map.repeat.set(2, 2);
    return map;
  }

  function createSolarTexture() {
    var canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 1024;
    var ctx = canvas.getContext('2d');

    ctx.fillStyle = '#05112E';
    ctx.fillRect(0, 0, 512, 1024);

    var cols = 8;
    var rows = 24;
    var pad = 4;
    var cellW = (512 - pad * (cols + 1)) / cols;
    var cellH = (1024 - pad * (rows + 1)) / rows;

    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var x = pad + c * (cellW + pad);
        var y = pad + r * (cellH + pad);
        var g = ctx.createLinearGradient(x, y, x + cellW, y + cellH);
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

    var tex = new THREE.CanvasTexture(canvas);
    tex.anisotropy = 4;
    return tex;
  }

  function createEarthTextures() {
    var width = 2048;
    var height = 1024;

    function toXY(lat, lon) {
      var x = ((lon + 180) / 360) * width;
      var y = ((90 - lat) / 180) * height;
      return [x, y];
    }

    var indiaPolygon = [
      [35.5, 74.0], [34.5, 77.8], [32.0, 78.5], [30.0, 81.0],
      [28.2, 88.2], [27.8, 92.0], [28.0, 97.0], [24.0, 94.5],
      [22.0, 92.5], [22.2, 89.0], [21.5, 87.0], [19.8, 85.8],
      [16.5, 82.2], [13.1, 80.3], [10.3, 79.8], [8.08, 77.55],
      [9.9, 76.2], [12.9, 74.8], [15.4, 73.8], [19.0, 72.8],
      [20.7, 70.9], [22.3, 69.0], [23.8, 68.4], [24.5, 71.0],
      [27.5, 70.5], [30.5, 73.8], [33.5, 73.5]
    ];

    var eurasiaPoly = [
      [36.0, -9.0], [43.0, -9.0], [48.0, -4.5], [54.0, 8.5],
      [60.0, 5.0], [70.0, 25.0], [70.0, 60.0], [73.0, 100.0],
      [70.0, 140.0], [65.0, 170.0], [55.0, 160.0], [50.0, 140.0],
      [40.0, 125.0], [35.0, 120.0], [23.0, 114.0], [20.0, 106.0],
      [10.0, 105.0], [1.3, 103.8], [7.0, 99.0], [16.0, 97.5],
      [22.0, 92.5], [28.0, 97.0], [35.5, 74.0], [25.2, 62.0],
      [26.5, 56.5], [30.0, 48.5], [32.0, 35.0], [36.5, 36.0]
    ];

    var africaPoly = [
      [37.0, 10.0], [32.0, 32.0], [12.0, 44.0], [11.8, 51.3],
      [2.0, 45.0], [-10.0, 40.5], [-26.0, 33.0], [-34.8, 20.0],
      [-30.0, 17.0], [-17.0, 11.8], [-5.0, 12.0], [4.0, 9.5],
      [5.0, -5.0], [15.0, -17.5], [25.0, -15.0], [35.8, -5.8]
    ];

    var dayCanvas = document.createElement('canvas');
    dayCanvas.width = width;
    dayCanvas.height = height;
    var dCtx = dayCanvas.getContext('2d');

    // Rich deep-space ocean gradient
    var oceanGrad = dCtx.createLinearGradient(0, 0, 0, height);
    oceanGrad.addColorStop(0, '#020612');
    oceanGrad.addColorStop(0.3, '#051332');
    oceanGrad.addColorStop(0.5, '#082154');
    oceanGrad.addColorStop(0.7, '#051332');
    oceanGrad.addColorStop(1, '#020612');
    dCtx.fillStyle = oceanGrad;
    dCtx.fillRect(0, 0, width, height);

    function drawPoly(ctx, poly, fill, stroke, lw, shadowColor, shadowBlur) {
      ctx.save();
      ctx.beginPath();
      for (var i = 0; i < poly.length; i++) {
        var pt = toXY(poly[i][0], poly[i][1]);
        if (i === 0) ctx.moveTo(pt[0], pt[1]);
        else ctx.lineTo(pt[0], pt[1]);
      }
      ctx.closePath();
      if (shadowColor && shadowBlur) {
        ctx.shadowColor = shadowColor;
        ctx.shadowBlur = shadowBlur;
      }
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 2; ctx.stroke(); }
      ctx.restore();
    }

    // Continents with Electric Cobalt coastal glow
    drawPoly(dCtx, eurasiaPoly, '#103468', '#245BFF', 6, '#245BFF', 16);
    drawPoly(dCtx, africaPoly, '#103468', '#245BFF', 6, '#245BFF', 16);
    drawPoly(dCtx, eurasiaPoly, '#12262B', '#1E4844', 2);
    drawPoly(dCtx, africaPoly, '#12262B', '#1E4844', 2);

    // India Subcontinent Highlight with intense Electric Cobalt & Acid Lime coastal border
    drawPoly(dCtx, indiaPolygon, '#16363B', '#245BFF', 5, '#245BFF', 24);

    // Riverine corridor (Ganga & Brahmaputra delta)
    dCtx.save();
    dCtx.strokeStyle = 'rgba(198, 255, 0, 0.75)';
    dCtx.lineWidth = 3;
    dCtx.beginPath();
    [[30.0, 78.5], [27.5, 81.0], [25.5, 85.0], [24.0, 88.5], [22.2, 89.2]].forEach(function (pt, idx) {
      var xy = toXY(pt[0], pt[1]);
      if (idx === 0) dCtx.moveTo(xy[0], xy[1]);
      else dCtx.lineTo(xy[0], xy[1]);
    });
    dCtx.stroke();
    dCtx.restore();

    // Graticule grid lines
    dCtx.strokeStyle = 'rgba(36, 91, 255, 0.12)';
    dCtx.lineWidth = 1;
    for (var lat = -60; lat <= 60; lat += 20) {
      var gy = toXY(lat, 0)[1];
      dCtx.beginPath();
      dCtx.moveTo(0, gy);
      dCtx.lineTo(width, gy);
      dCtx.stroke();
    }
    for (var lon = -160; lon <= 180; lon += 20) {
      var gx = toXY(0, lon)[0];
      dCtx.beginPath();
      dCtx.moveTo(gx, 0);
      dCtx.lineTo(gx, height);
      dCtx.stroke();
    }

    // Emissive night-lights canvas
    var emissiveCanvas = document.createElement('canvas');
    emissiveCanvas.width = width;
    emissiveCanvas.height = height;
    var eCtx = emissiveCanvas.getContext('2d');
    eCtx.fillStyle = '#000000';
    eCtx.fillRect(0, 0, width, height);

    drawPoly(eCtx, indiaPolygon, 'rgba(36, 91, 255, 0.35)', '#245BFF', 4, '#245BFF', 20);

    var cities = [
      [28.6139, 77.209, 8],
      [19.076, 72.8777, 8],
      [12.9716, 77.5946, 7],
      [13.0827, 80.2707, 7],
      [22.5726, 88.3639, 9],
      [26.1445, 91.7362, 8], // Guwahati / Brahmaputra
      [25.5941, 85.1376, 8], // Patna / Ganga
      [20.5937, 78.9629, 9]  // Central AOI
    ];
    for (var k = 0; k < cities.length; k++) {
      var cPt = toXY(cities[k][0], cities[k][1]);
      var rad = cities[k][2];
      var g = eCtx.createRadialGradient(cPt[0], cPt[1], 0, cPt[0], cPt[1], rad * 3.2);
      g.addColorStop(0, '#FFFFFF');
      g.addColorStop(0.3, '#C6FF00');
      g.addColorStop(0.7, 'rgba(36, 91, 255, 0.85)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      eCtx.fillStyle = g;
      eCtx.beginPath();
      eCtx.arc(cPt[0], cPt[1], rad * 3.2, 0, Math.PI * 2);
      eCtx.fill();
    }

    // Cloud layer canvas
    var cloudCanvas = document.createElement('canvas');
    cloudCanvas.width = 1024;
    cloudCanvas.height = 512;
    var cCtx = cloudCanvas.getContext('2d');
    cCtx.clearRect(0, 0, 1024, 512);

    for (var m = 0; m < 340; m++) {
      var cx = Math.random() * 1024;
      var cy = 35 + Math.random() * 440;
      var rx = 22 + Math.random() * 65;
      var ry = 6 + Math.random() * 18;
      var cg = cCtx.createRadialGradient(cx, cy, 0, cx, cy, rx);
      cg.addColorStop(0, 'rgba(244, 247, 255, 0.32)');
      cg.addColorStop(0.6, 'rgba(180, 215, 255, 0.14)');
      cg.addColorStop(1, 'rgba(255, 255, 255, 0)');
      cCtx.fillStyle = cg;
      cCtx.save();
      cCtx.translate(cx, cy);
      cCtx.scale(1, ry / rx);
      cCtx.beginPath();
      cCtx.arc(0, 0, rx, 0, Math.PI * 2);
      cCtx.fill();
      cCtx.restore();
    }

    var dayTex = new THREE.CanvasTexture(dayCanvas);
    var emissiveTex = new THREE.CanvasTexture(emissiveCanvas);
    var cloudTex = new THREE.CanvasTexture(cloudCanvas);
    dayTex.anisotropy = 4;
    emissiveTex.anisotropy = 4;

    return { dayTex: dayTex, emissiveTex: emissiveTex, cloudTex: cloudTex };
  }

  function buildSatelliteAssembly(kaptonTex, solarTex) {
    var root = new THREE.Group();

    var kaptonMat = new THREE.MeshStandardMaterial({
      map: kaptonTex,
      roughnessMap: kaptonTex,
      color: new THREE.Color('#FFBE3D'),
      metalness: 0.88,
      roughness: 0.28
    });

    var titaniumMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#C4D0E2'),
      metalness: 0.92,
      roughness: 0.18
    });

    var darkAlloyMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#141C2B'),
      metalness: 0.86,
      roughness: 0.32
    });

    var solarMat = new THREE.MeshStandardMaterial({
      map: solarTex,
      color: new THREE.Color('#4A88FF'),
      metalness: 0.82,
      roughness: 0.15,
      emissive: new THREE.Color('#0B2569'),
      emissiveIntensity: 0.6
    });

    // Octagonal Gold Kapton Bus
    var busMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.82, 0.82, 2.35, 8), kaptonMat);
    busMesh.rotation.z = Math.PI / 2;
    root.add(busMesh);

    // Bulkhead rings
    [-1.18, -0.45, 0.45, 1.18].forEach(function (xPos) {
      var ring = new THREE.Mesh(new THREE.CylinderGeometry(0.87, 0.87, 0.08, 16), titaniumMat);
      ring.rotation.z = Math.PI / 2;
      ring.position.x = xPos;
      root.add(ring);
    });

    // Nose Module
    var noseModule = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.78, 0.7, 12), darkAlloyMat);
    noseModule.rotation.z = -Math.PI / 2;
    noseModule.position.x = 1.52;
    root.add(noseModule);

    // Thruster housing
    var thrusterHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.72, 0.5, 12), darkAlloyMat);
    thrusterHousing.rotation.z = Math.PI / 2;
    thrusterHousing.position.x = -1.42;
    root.add(thrusterHousing);

    // Dual Solar Array Wings
    var wingsGroup = new THREE.Group();
    [-1, 1].forEach(function (dir) {
      var wingPivot = new THREE.Group();
      wingPivot.position.set(0, 0, dir * 0.9);

      var boomLen = 4.4;
      var boom = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, boomLen), titaniumMat);
      boom.position.z = dir * (boomLen / 2 + 0.15);
      wingPivot.add(boom);

      for (var s = 0; s < 3; s++) {
        var segZ = dir * (1.05 + s * 1.38);
        var frame = new THREE.Mesh(new THREE.BoxGeometry(1.42, 0.04, 1.32), darkAlloyMat);
        frame.position.z = segZ;
        var cells = new THREE.Mesh(new THREE.BoxGeometry(1.34, 0.052, 1.24), solarMat);
        cells.position.z = segZ;
        wingPivot.add(frame);
        wingPivot.add(cells);
      }
      wingsGroup.add(wingPivot);
    });
    root.add(wingsGroup);

    // Articulated High-Gain Dish
    var dishPivot = new THREE.Group();
    dishPivot.position.set(1.15, -0.35, 0.55);

    var dishBowl = new THREE.Mesh(
      new THREE.SphereGeometry(0.65, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.42),
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#E2ECFF'),
        metalness: 0.88,
        roughness: 0.18,
        side: THREE.DoubleSide
      })
    );
    dishBowl.rotation.x = Math.PI / 2;

    var dishHead = new THREE.Group();
    dishHead.add(dishBowl);

    var emitterCoreMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#FFFFFF') });
    var emitterCore = new THREE.Mesh(new THREE.SphereGeometry(0.065, 16, 16), emitterCoreMat);
    emitterCore.position.z = 0.55;
    dishHead.add(emitterCore);

    var emitterHalo = new THREE.Mesh(
      new THREE.RingGeometry(0.09, 0.18, 32),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#FFFFFF'),
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.0
      })
    );
    emitterHalo.position.z = 0.56;
    dishHead.add(emitterHalo);

    dishPivot.add(dishHead);
    root.add(dishPivot);

    var beaconLime = new THREE.Mesh(
      new THREE.SphereGeometry(0.048, 12, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.acidLime) })
    );
    beaconLime.position.set(0.55, 1.02, 0.15);
    root.add(beaconLime);

    return {
      root: root,
      wingsGroup: wingsGroup,
      dishPivot: dishPivot,
      emitterCore: emitterCore,
      emitterHalo: emitterHalo,
      beaconLime: beaconLime
    };
  }

  // 8-Section Master Cinematic Keyframe Trajectory (0.0 to 1.0)
  var SCROLL_KEYFRAMES = {
    // 01 Hero, 02 Problem, 03 How it works, 04 Try it, 05 Link sim, 06 Results, 07 Judges, 08 Limits/Footer
    earthX:           [ 3.60,   0.25,   -3.20,   -2.20,    3.00,    2.40,   -2.60,    1.20 ],
    earthY:           [-0.30,  -0.15,   -0.25,    0.45,   -0.20,   -0.45,    0.18,   -0.30 ],
    earthZ:           [-1.50,   0.40,    1.20,    3.40,    1.50,   -2.80,   -0.35,   -4.50 ],
    earthScale:       [ 1.15,   1.38,    1.46,    1.75,    1.36,    0.92,    1.18,    0.78 ],
    earthYawOffset:   [ 0.00,   0.25,    0.52,    0.18,   -0.30,    0.12,    0.65,    1.18 ],
    earthPitchOffset: [ 0.00,   0.08,    0.18,    0.28,    0.12,   -0.08,    0.32,    0.06 ],

    satX:             [-3.95,  -6.20,   -8.80,   -9.50,   -8.50,   -3.15,   -7.20,   -2.80 ],
    satY:             [ 0.28,   1.15,    2.10,    2.60,    1.95,    0.65,    1.85,    0.95 ],
    satZ:             [ 8.85,   6.40,    3.50,    2.00,    3.20,    7.80,    4.20,    6.10 ],
    satScale:         [ 1.05,   0.82,    0.60,    0.48,    0.60,    0.94,    0.65,    0.78 ],

    camX:             [-0.40,   0.00,   -0.55,   -0.75,    0.60,   -0.20,   -0.40,    0.15 ],
    camY:             [ 0.35,   0.25,    0.45,    0.58,    0.30,    0.40,    0.85,    0.55 ],
    camZ:             [13.50,  12.20,   11.40,   10.20,   11.50,   13.80,   12.30,   14.60 ],

    lookX:            [ 0.65,   0.18,   -1.45,   -1.35,    1.45,    0.15,   -1.15,    0.35 ],
    lookY:            [-0.08,  -0.05,   -0.10,    0.22,   -0.08,   -0.05,    0.08,   -0.10 ],
    lookZ:            [ 1.80,   0.80,    0.90,    2.50,    1.10,    0.50,    0.20,   -1.20 ],

    laserOpacity:     [ 1.00,   0.75,    0.45,    0.30,    0.95,    0.85,    0.50,    0.25 ],
    chipGridOpacity:  [ 0.00,   0.05,    0.35,    1.00,    0.55,    0.15,    0.20,    0.05 ],
    orbitRingOpacity: [ 0.70,   0.90,    0.80,    0.50,    0.75,    0.95,    0.90,    0.80 ]
  };

  window.initOrbital3D = function (container) {
    if (!container || !window.THREE) return;

    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: 'high-performance',
        alpha: false
      });
    } catch (e) {
      console.warn('WebGL initialization failed, falling back:', e);
      return;
    }

    var isMobile = window.innerWidth < 768;
    var dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.midnightBlack);
    scene.fog = new THREE.FogExp2(COLORS.midnightBlack, 0.0035);

    var camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 400);

    var ambientLight = new THREE.AmbientLight('#182B52', 0.75);
    scene.add(ambientLight);

    var sunLight = new THREE.DirectionalLight('#FFF9EE', 2.9);
    sunLight.position.set(18, 12, 18);
    scene.add(sunLight);

    var cobaltRimLight = new THREE.PointLight(COLORS.cobaltBlue, 3.5, 36);
    cobaltRimLight.position.set(-7, 4, 13);
    scene.add(cobaltRimLight);

    // Deep space starfield
    var starCount = isMobile ? 1400 : 3200;
    var starGeo = new THREE.BufferGeometry();
    var starPositions = new Float32Array(starCount * 3);
    var starColors = new Float32Array(starCount * 3);

    for (var i = 0; i < starCount; i++) {
      var r = 65 + Math.random() * 145;
      var theta = Math.random() * Math.PI * 2;
      var phi = Math.acos(2 * Math.random() - 1);
      starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      starPositions[i * 3 + 2] = r * Math.cos(phi);

      var c = Math.random() > 0.8
        ? new THREE.Color('#78A4FF')
        : (Math.random() > 0.94 ? new THREE.Color('#C6FF00') : new THREE.Color('#F4F7FF'));
      starColors[i * 3] = c.r;
      starColors[i * 3 + 1] = c.g;
      starColors[i * 3 + 2] = c.b;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
    var starMat = new THREE.PointsMaterial({
      size: isMobile ? 0.24 : 0.19,
      vertexColors: true,
      transparent: true,
      opacity: 0.92
    });
    var starField = new THREE.Points(starGeo, starMat);
    scene.add(starField);

    // 3D Satellite Model
    var kaptonTex = createKaptonTexture();
    var solarTex = createSolarTexture();
    var satellite = buildSatelliteAssembly(kaptonTex, solarTex);
    satellite.root.position.set(-3.95, 0.28, 8.85);
    satellite.root.rotation.set(0.22, 0.55, -0.18);
    scene.add(satellite.root);

    // Persistent 3D Earth Group
    var EARTH_RADIUS = 4.65;
    var earthGroup = new THREE.Group();
    earthGroup.position.set(3.6, -0.3, -1.5);
    scene.add(earthGroup);

    var textures = createEarthTextures();
    var earthMat = new THREE.MeshStandardMaterial({
      map: textures.dayTex,
      emissiveMap: textures.emissiveTex,
      emissive: new THREE.Color('#FFFFFF'),
      emissiveIntensity: 1.1,
      roughness: 0.5,
      metalness: 0.22
    });
    var earthMesh = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS, 64, 64), earthMat);
    earthGroup.add(earthMesh);

    var baseEarthRotY = -((INDIA_COORDS.lon + 98) * Math.PI) / 180;
    var baseEarthRotX = ((INDIA_COORDS.lat - 8) * Math.PI) / 180;
    earthGroup.rotation.set(baseEarthRotX, baseEarthRotY, 0.06);

    // Cloud Layer
    var cloudMat = new THREE.MeshStandardMaterial({
      map: textures.cloudTex,
      transparent: true,
      opacity: 0.48,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });
    var cloudMesh = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS * 1.013, 48, 48), cloudMat);
    earthGroup.add(cloudMesh);

    // Atmospheric Glow (Fresnel Shader)
    var atmoMat = new THREE.ShaderMaterial({
      uniforms: { uIntensity: { value: 1.25 } },
      vertexShader: [
        'varying vec3 vNormal;',
        'varying vec3 vPositionNormal;',
        'void main() {',
        '  vNormal = normalize(normalMatrix * normal);',
        '  vPositionNormal = normalize((modelViewMatrix * vec4(position, 1.0)).xyz);',
        '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
        '}'
      ].join('\n'),
      fragmentShader: [
        'varying vec3 vNormal;',
        'varying vec3 vPositionNormal;',
        'uniform float uIntensity;',
        'void main() {',
        '  float fresnel = pow(0.68 - dot(vNormal, vPositionNormal), 2.7);',
        '  vec3 col = mix(vec3(0.141, 0.357, 1.0), vec3(0.65, 0.88, 1.0), fresnel);',
        '  gl_FragColor = vec4(col, clamp(fresnel * uIntensity, 0.0, 0.95));',
        '}'
      ].join('\n'),
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false
    });
    var atmoMesh = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS * 1.068, 64, 64), atmoMat);
    earthGroup.add(atmoMesh);

    // India Ground Station Target Pin
    var indiaSurfacePos = latLonToVector3(INDIA_COORDS.lat, INDIA_COORDS.lon, EARTH_RADIUS * 1.016);
    var indiaTargetGroup = new THREE.Group();
    indiaTargetGroup.position.copy(indiaSurfacePos);
    indiaTargetGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), indiaSurfacePos.clone().normalize());
    earthGroup.add(indiaTargetGroup);

    var indiaCorePin = new THREE.Mesh(
      new THREE.CircleGeometry(0.075, 24),
      new THREE.MeshBasicMaterial({ color: new THREE.Color('#FFFFFF'), side: THREE.DoubleSide })
    );
    indiaTargetGroup.add(indiaCorePin);

    // 3D 4x4 Flood Grid on Earth Surface (Illuminates in Section 4)
    var chip3DGroup = new THREE.Group();
    chip3DGroup.position.z = 0.035;
    indiaTargetGroup.add(chip3DGroup);

    var tile3DMeshes = [];
    var gridSpan = 1.32;
    var cellSpan = gridSpan / 4;
    FLOOD_TILES.forEach(function (t, idx) {
      var r = Math.floor(idx / 4);
      var c = idx % 4;
      var tx = (c - 1.5) * cellSpan;
      var ty = (1.5 - r) * cellSpan;

      var tileMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(t.isCritical ? COLORS.acidLime : (t.isModerate ? COLORS.cobaltBlue : '#1E3A6E')),
        transparent: true,
        opacity: 0.45,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      var tileMesh = new THREE.Mesh(new THREE.PlaneGeometry(cellSpan * 0.86, cellSpan * 0.86), tileMat);
      tileMesh.position.set(tx, ty, 0);
      chip3DGroup.add(tileMesh);
      tile3DMeshes.push({ mesh: tileMesh, mat: tileMat, isCritical: t.isCritical, id: idx });
    });

    // Expanding Sonar Ripples on Ground Station
    var rippleMeshes = [];
    for (var rIdx = 0; rIdx < 3; rIdx++) {
      var rColor = rIdx === 0 ? '#FFFFFF' : (rIdx === 1 ? COLORS.acidLime : COLORS.cobaltBlue);
      var rippleMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(rColor),
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      var ripple = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.24, 48), rippleMat);
      indiaTargetGroup.add(ripple);
      rippleMeshes.push(ripple);
    }

    // ------------------------------------------------------------------------
    // LUMINOUS ORBITAL RINGS & MINIATURE ORBITING SATELLITE
    // ------------------------------------------------------------------------
    var orbitRadius = EARTH_RADIUS * 1.22;

    // 1. Primary Inclined Orbit (Electric Cobalt + Pure White Core Filament)
    var primaryOrbitPoints = [];
    for (var oi = 0; oi < 128; oi++) {
      var oa = (oi / 128) * Math.PI * 2;
      primaryOrbitPoints.push(
        new THREE.Vector3(
          Math.cos(oa) * orbitRadius,
          Math.sin(oa) * orbitRadius * 0.44,
          Math.sin(oa) * orbitRadius * 0.90
        )
      );
    }
    var primaryCurve = new THREE.CatmullRomCurve3(primaryOrbitPoints, true);
    var primaryTubeGeo = new THREE.TubeGeometry(primaryCurve, 128, 0.018, 6, true);
    var orbitTubeMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(COLORS.cobaltBlue),
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending
    });
    var primaryOrbitTube = new THREE.Mesh(primaryTubeGeo, orbitTubeMat);
    earthGroup.add(primaryOrbitTube);

    var primaryCoreGeo = new THREE.TubeGeometry(primaryCurve, 128, 0.006, 4, true);
    var primaryCoreMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color('#FFFFFF'),
      transparent: true,
      opacity: 0.70,
      blending: THREE.AdditiveBlending
    });
    var primaryCoreTube = new THREE.Mesh(primaryCoreGeo, primaryCoreMat);
    earthGroup.add(primaryCoreTube);

    // Glowing Orbital Waypoint Nodes along the Primary Orbit
    var orbitNodeCoords = [];
    for (var onIdx = 0; onIdx < 20; onIdx++) {
      var pt = primaryCurve.getPoint(onIdx / 20);
      orbitNodeCoords.push(pt.x, pt.y, pt.z);
    }
    var orbitNodeGeo = new THREE.BufferGeometry();
    orbitNodeGeo.setAttribute('position', new THREE.Float32BufferAttribute(orbitNodeCoords, 3));
    var orbitNodeMat = new THREE.PointsMaterial({
      color: new THREE.Color('#B6D5FF'),
      size: 0.16,
      transparent: true,
      opacity: 0.90,
      blending: THREE.AdditiveBlending
    });
    var orbitNodes = new THREE.Points(orbitNodeGeo, orbitNodeMat);
    earthGroup.add(orbitNodes);

    // 2. Second Polar Calibration Orbit Ring (Acid Lime)
    var polarOrbitPoints = [];
    for (var pi = 0; pi < 128; pi++) {
      var pa = (pi / 128) * Math.PI * 2;
      polarOrbitPoints.push(
        new THREE.Vector3(
          Math.cos(pa) * orbitRadius * 0.35,
          Math.sin(pa) * orbitRadius * 1.05,
          Math.cos(pa) * orbitRadius * 0.96
        )
      );
    }
    var polarCurve = new THREE.CatmullRomCurve3(polarOrbitPoints, true);
    var polarTubeGeo = new THREE.TubeGeometry(polarCurve, 128, 0.014, 6, true);
    var polarTubeMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(COLORS.acidLime),
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending
    });
    var polarOrbitTube = new THREE.Mesh(polarTubeGeo, polarTubeMat);
    earthGroup.add(polarOrbitTube);

    // 3. Third Equatorial Navigation Ring (Subtle Cyan-Cobalt)
    var eqOrbitPoints = [];
    var eqRadius = EARTH_RADIUS * 1.32;
    for (var ei = 0; ei < 128; ei++) {
      var ea = (ei / 128) * Math.PI * 2;
      eqOrbitPoints.push(
        new THREE.Vector3(
          Math.cos(ea) * eqRadius,
          Math.sin(ea) * eqRadius * 0.14,
          Math.sin(ea) * eqRadius * 0.98
        )
      );
    }
    var eqCurve = new THREE.CatmullRomCurve3(eqOrbitPoints, true);
    var eqTubeGeo = new THREE.TubeGeometry(eqCurve, 128, 0.010, 4, true);
    var eqTubeMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color('#4E8AFF'),
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending
    });
    var eqOrbitTube = new THREE.Mesh(eqTubeGeo, eqTubeMat);
    earthGroup.add(eqOrbitTube);

    // 4. Miniature Orbiting Satellite revolving along the Primary Orbit
    var miniSatGroup = new THREE.Group();
    var miniBody = new THREE.Mesh(
      new THREE.BoxGeometry(0.14, 0.14, 0.24),
      new THREE.MeshBasicMaterial({ color: new THREE.Color('#FFFFFF') })
    );
    var miniWings = new THREE.Mesh(
      new THREE.BoxGeometry(0.72, 0.02, 0.18),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.cobaltBlue) })
    );
    var miniBeacon = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 12, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.acidLime) })
    );
    miniBeacon.position.set(0, 0.09, 0.09);
    miniSatGroup.add(miniBody);
    miniSatGroup.add(miniWings);
    miniSatGroup.add(miniBeacon);
    earthGroup.add(miniSatGroup);

    // Laser Core & Sheath
    var laserGroup = new THREE.Group();
    scene.add(laserGroup);

    var coreGeo = new THREE.CylinderGeometry(0.005, 0.005, 1, 12, 1, true);
    coreGeo.translate(0, 0.5, 0);
    coreGeo.rotateX(Math.PI / 2);
    var coreLaserMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color('#FFFFFF'),
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending
    });
    var coreLaserMesh = new THREE.Mesh(coreGeo, coreLaserMat);
    laserGroup.add(coreLaserMesh);

    var sheathGeo = new THREE.CylinderGeometry(0.02, 0.028, 1, 16, 1, true);
    sheathGeo.translate(0, 0.5, 0);
    sheathGeo.rotateX(Math.PI / 2);
    var sheathLaserMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(COLORS.cobaltBlue),
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending
    });
    var sheathLaserMesh = new THREE.Mesh(sheathGeo, sheathLaserMat);
    laserGroup.add(sheathLaserMesh);

    // Photonic Data Packets
    var packetCount = 36;
    var packetGeo = new THREE.BufferGeometry();
    var packetPos = new Float32Array(packetCount * 3);
    var packetColors = new Float32Array(packetCount * 3);
    for (var p = 0; p < packetCount; p++) {
      var pCol = p % 4 === 0 ? new THREE.Color(COLORS.acidLime) : new THREE.Color('#FFFFFF');
      packetColors[p * 3] = pCol.r;
      packetColors[p * 3 + 1] = pCol.g;
      packetColors[p * 3 + 2] = pCol.b;
    }
    packetGeo.setAttribute('position', new THREE.BufferAttribute(packetPos, 3));
    packetGeo.setAttribute('color', new THREE.BufferAttribute(packetColors, 3));
    var packetMat = new THREE.PointsMaterial({
      size: 0.15,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending
    });
    var laserPackets = new THREE.Points(packetGeo, packetMat);
    scene.add(laserPackets);

    // Interactive 3D Orbiting Highlight Function
    window.highlight3DTile = function (tileId, isCritical) {
      tile3DMeshes.forEach(function (item) {
        if (item.id === tileId) {
          item.mat.color.set(isCritical ? COLORS.acidLime : '#FFFFFF');
          item.mat.opacity = 1.0;
          item.mesh.scale.set(1.4, 1.4, 1.4);
        } else {
          item.mesh.scale.set(1.0, 1.0, 1.0);
        }
      });
    };

    // Intro & Scroll State Management
    var introProgress = 0.0;
    var introComplete = false;
    var targetScroll = 0.0;
    var smoothScroll = 0.0;
    var inspect3D = false;
    var dragData = { isDragging: false, prevX: 0, prevY: 0, rotY: 0, rotX: 0 };

    window.skipIntro = function () {
      introProgress = 1.0;
      introComplete = true;
      var introBanner = document.getElementById('introBanner');
      if (introBanner) introBanner.style.display = 'none';
    };

    window.replayIntro = function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      introProgress = 0.0;
      introComplete = false;
      var introBanner = document.getElementById('introBanner');
      if (introBanner) introBanner.style.display = 'inline-flex';
    };

    window.toggle3DDragMode = function () {
      inspect3D = !inspect3D;
      container.style.pointerEvents = inspect3D ? 'auto' : 'none';
      container.style.cursor = inspect3D ? 'grab' : 'default';
      var btn = document.getElementById('drag3dBtn');
      if (btn) {
        btn.classList.toggle('active', inspect3D);
        btn.innerHTML = inspect3D ? '&#9679; EXIT 3D DRAG' : '3D DRAG MODE';
      }
      return inspect3D;
    };

    // GSAP ScrollTrigger Integration for silky-smooth continuous document scrolling
    if (window.gsap && window.ScrollTrigger) {
      window.gsap.registerPlugin(window.ScrollTrigger);

      window.ScrollTrigger.create({
        trigger: 'body',
        start: 'top top',
        end: 'bottom bottom',
        scrub: 1.2,
        onUpdate: function (self) {
          targetScroll = self.progress;

          if (self.progress > 0.02 && !introComplete) {
            window.skipIntro();
          }

          var ribbon = document.getElementById('scrollRibbon');
          if (ribbon) {
            ribbon.style.width = Math.max(2, self.progress * 100) + '%';
          }

          var railDots = document.querySelectorAll('.rail-dot');
          var secIdx = Math.min(railDots.length - 1, Math.floor(self.progress * railDots.length + 0.05));
          railDots.forEach(function (dot, i) {
            dot.classList.toggle('active', i === secIdx);
          });

          var stageLabel = document.getElementById('hudStage');
          if (stageLabel && MISSION_SECTIONS[secIdx]) {
            stageLabel.innerText = MISSION_SECTIONS[secIdx].code + ' ' + MISSION_SECTIONS[secIdx].label;
          }
        }
      });
    } else {
      // Native scroll fallback
      function onScroll() {
        var docEl = document.documentElement;
        var max = Math.max(1, docEl.scrollHeight - window.innerHeight);
        var cur = window.scrollY || docEl.scrollTop || 0;
        targetScroll = clamp(cur / max, 0, 1);

        if (cur > 40 && !introComplete) {
          window.skipIntro();
        }

        var ribbon = document.getElementById('scrollRibbon');
        if (ribbon) {
          ribbon.style.width = Math.max(2, targetScroll * 100) + '%';
        }

        var railDots = document.querySelectorAll('.rail-dot');
        var secIdx = Math.min(railDots.length - 1, Math.floor(targetScroll * railDots.length + 0.05));
        railDots.forEach(function (dot, i) {
          dot.classList.toggle('active', i === secIdx);
        });

        var stageLabel = document.getElementById('hudStage');
        if (stageLabel && MISSION_SECTIONS[secIdx]) {
          stageLabel.innerText = MISSION_SECTIONS[secIdx].code + ' ' + MISSION_SECTIONS[secIdx].label;
        }
      }
      window.addEventListener('scroll', onScroll, { passive: true });
      onScroll();
    }

    // Interactive Drag Listeners for 3D Inspect Mode
    function onPointerDown(e) {
      if (!inspect3D) return;
      var cx = e.touches ? e.touches[0].clientX : e.clientX;
      var cy = e.touches ? e.touches[0].clientY : e.clientY;
      dragData.isDragging = true;
      dragData.prevX = cx;
      dragData.prevY = cy;
    }
    function onPointerMove(e) {
      if (!inspect3D || !dragData.isDragging) return;
      var cx = e.touches ? e.touches[0].clientX : e.clientX;
      var cy = e.touches ? e.touches[0].clientY : e.clientY;
      var dx = cx - dragData.prevX;
      var dy = cy - dragData.prevY;
      dragData.prevX = cx;
      dragData.prevY = cy;
      dragData.rotY += dx * 0.0055;
      dragData.rotX = clamp(dragData.rotX + dy * 0.004, -0.65, 0.65);
    }
    function onPointerUp() {
      dragData.isDragging = false;
    }

    renderer.domElement.addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);
    renderer.domElement.addEventListener('touchstart', onPointerDown, { passive: true });
    window.addEventListener('touchmove', onPointerMove, { passive: true });
    window.addEventListener('touchend', onPointerUp);

    function onResize() {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }
    window.addEventListener('resize', onResize);

    var clock = new THREE.Clock();
    var emitterWorldPos = new THREE.Vector3();
    var indiaWorldPos = new THREE.Vector3();
    var currentCamLook = new THREE.Vector3(0.65, -0.08, 1.8);

    function animate() {
      requestAnimationFrame(animate);
      var dt = Math.min(clock.getDelta(), 0.05);
      var elapsed = clock.getElapsedTime();

      // 1. Advance Intro Sequence if active
      if (!introComplete) {
        introProgress = clamp(introProgress + dt * 0.12, 0.0, 1.0);
        if (introProgress >= 1.0) {
          introComplete = true;
          var introBanner = document.getElementById('introBanner');
          if (introBanner) introBanner.style.display = 'none';
        } else {
          var step = INTRO_STEPS[INTRO_STEPS.length - 1];
          for (var sIdx = 0; sIdx < INTRO_STEPS.length; sIdx++) {
            if (introProgress >= INTRO_STEPS[sIdx].range[0] && introProgress <= INTRO_STEPS[sIdx].range[1]) {
              step = INTRO_STEPS[sIdx];
              break;
            }
          }
          var stepText = document.getElementById('introStepText');
          if (stepText) {
            stepText.innerText = step.code + ': ' + step.name + ' (' + Math.round(introProgress * 100) + '%)';
          }
        }
      }

      var ip = introProgress;

      // 2. Continuous Scroll Interpolation with Exponential Damping
      smoothScroll = THREE.MathUtils.damp(smoothScroll, targetScroll, 5.2, dt);
      var s = smoothScroll;

      var mobileScaleMod = isMobile ? 0.78 : 1.0;
      var mobileXMod = isMobile ? 0.45 : 1.0;

      // Evaluate Continuous Spline Target Poses
      var scrollEarthX = sampleSpline(SCROLL_KEYFRAMES.earthX, s) * mobileXMod;
      var scrollEarthY = sampleSpline(SCROLL_KEYFRAMES.earthY, s);
      var scrollEarthZ = sampleSpline(SCROLL_KEYFRAMES.earthZ, s);
      var scrollEarthScale = sampleSpline(SCROLL_KEYFRAMES.earthScale, s) * mobileScaleMod;
      var scrollEarthYaw = sampleSpline(SCROLL_KEYFRAMES.earthYawOffset, s);
      var scrollEarthPitch = sampleSpline(SCROLL_KEYFRAMES.earthPitchOffset, s);

      var scrollSatX = sampleSpline(SCROLL_KEYFRAMES.satX, s) * mobileXMod;
      var scrollSatY = sampleSpline(SCROLL_KEYFRAMES.satY, s);
      var scrollSatZ = sampleSpline(SCROLL_KEYFRAMES.satZ, s);
      var scrollSatScale = sampleSpline(SCROLL_KEYFRAMES.satScale, s) * mobileScaleMod;

      var scrollCamX = sampleSpline(SCROLL_KEYFRAMES.camX, s) * mobileXMod;
      var scrollCamY = sampleSpline(SCROLL_KEYFRAMES.camY, s);
      var scrollCamZ = sampleSpline(SCROLL_KEYFRAMES.camZ, s);

      var scrollLookX = sampleSpline(SCROLL_KEYFRAMES.lookX, s) * mobileXMod;
      var scrollLookY = sampleSpline(SCROLL_KEYFRAMES.lookY, s);
      var scrollLookZ = sampleSpline(SCROLL_KEYFRAMES.lookZ, s);

      var laserOp = sampleSpline(SCROLL_KEYFRAMES.laserOpacity, s);
      var chipGridOp = sampleSpline(SCROLL_KEYFRAMES.chipGridOpacity, s);

      // Smooth Intro Camera Handoff into Scroll Pose (Zero Jump!)
      var introBlend = smoothstep(0.0, 1.0, ip);
      var targetCamX = THREE.MathUtils.lerp(-2.35 * mobileXMod, scrollCamX, introBlend);
      var targetCamY = THREE.MathUtils.lerp(0.42, scrollCamY, introBlend);
      var targetCamZ = THREE.MathUtils.lerp(13.95, scrollCamZ, introBlend);

      var targetLookX = THREE.MathUtils.lerp(-0.95 * mobileXMod, scrollLookX, introBlend);
      var targetLookY = THREE.MathUtils.lerp(0.08, scrollLookY, introBlend);
      var targetLookZ = THREE.MathUtils.lerp(4.2, scrollLookZ, introBlend);

      // Apply Transformations to Earth
      earthGroup.position.x = THREE.MathUtils.damp(earthGroup.position.x, scrollEarthX, 6, dt);
      earthGroup.position.y = THREE.MathUtils.damp(earthGroup.position.y, scrollEarthY, 6, dt);
      earthGroup.position.z = THREE.MathUtils.damp(earthGroup.position.z, scrollEarthZ, 6, dt);

      var curScale = THREE.MathUtils.damp(earthGroup.scale.x, scrollEarthScale, 6, dt);
      earthGroup.scale.setScalar(curScale);

      // Geographic viewing focus + subtle orbital drift + manual drag offset
      var desRotY = baseEarthRotY + scrollEarthYaw + dragData.rotY + Math.sin(elapsed * 0.12) * 0.035;
      var desRotX = baseEarthRotX + scrollEarthPitch + dragData.rotX;
      earthGroup.rotation.y = THREE.MathUtils.damp(earthGroup.rotation.y, desRotY, 6, dt);
      earthGroup.rotation.x = THREE.MathUtils.damp(earthGroup.rotation.x, desRotX, 6, dt);

      // Continuous cloud movement
      cloudMesh.rotation.y = elapsed * 0.022;

      // Apply Transformations to Satellite
      satellite.root.position.x = THREE.MathUtils.damp(satellite.root.position.x, scrollSatX, 6, dt);
      satellite.root.position.y = THREE.MathUtils.damp(satellite.root.position.y, scrollSatY + Math.sin(elapsed * 0.9) * 0.08, 6, dt);
      satellite.root.position.z = THREE.MathUtils.damp(satellite.root.position.z, scrollSatZ, 6, dt);
      satellite.root.scale.setScalar(THREE.MathUtils.damp(satellite.root.scale.x, scrollSatScale, 6, dt));
      satellite.beaconLime.scale.setScalar(0.85 + Math.sin(elapsed * 7.0) * 0.35);

      // Apply Transformations to Camera
      camera.position.x = THREE.MathUtils.damp(camera.position.x, targetCamX, 6, dt);
      camera.position.y = THREE.MathUtils.damp(camera.position.y, targetCamY, 6, dt);
      camera.position.z = THREE.MathUtils.damp(camera.position.z, targetCamZ, 6, dt);

      currentCamLook.x = THREE.MathUtils.damp(currentCamLook.x, targetLookX, 6, dt);
      currentCamLook.y = THREE.MathUtils.damp(currentCamLook.y, targetLookY, 6, dt);
      currentCamLook.z = THREE.MathUtils.damp(currentCamLook.z, targetLookZ, 6, dt);
      camera.lookAt(currentCamLook);

      // Track laser dish towards India Target
      satellite.emitterCore.getWorldPosition(emitterWorldPos);
      indiaCorePin.getWorldPosition(indiaWorldPos);
      satellite.dishPivot.lookAt(indiaWorldPos);

      var antennaLock = smoothstep(0.22, 0.40, ip);
      var laserExtend = smoothstep(0.40, 0.65, ip);
      var effectiveLaserOp = antennaLock * laserOp;

      var beamDist = emitterWorldPos.distanceTo(indiaWorldPos);
      coreLaserMesh.position.copy(emitterWorldPos);
      coreLaserMesh.lookAt(indiaWorldPos);
      coreLaserMesh.scale.set(1, 1, beamDist * laserExtend);
      coreLaserMat.opacity = effectiveLaserOp * 0.95;

      sheathLaserMesh.position.copy(emitterWorldPos);
      sheathLaserMesh.lookAt(indiaWorldPos);
      sheathLaserMesh.scale.set(1, 1, beamDist * laserExtend);
      sheathLaserMat.opacity = effectiveLaserOp * 0.75;

      satellite.emitterHalo.material.opacity = effectiveLaserOp * 0.9;
      satellite.emitterHalo.scale.setScalar(1.0 + Math.sin(elapsed * 10.0) * 0.18);

      // Animate photonic telemetry packets streaming down laser beam
      packetMat.opacity = effectiveLaserOp * 0.95;
      var pAttr = packetGeo.attributes.position;
      for (var k = 0; k < packetCount; k++) {
        var t = ((k / packetCount + elapsed * 0.82) % 1.0) * laserExtend;
        pAttr.setXYZ(
          k,
          THREE.MathUtils.lerp(emitterWorldPos.x, indiaWorldPos.x, t),
          THREE.MathUtils.lerp(emitterWorldPos.y, indiaWorldPos.y, t),
          THREE.MathUtils.lerp(emitterWorldPos.z, indiaWorldPos.z, t)
        );
      }
      pAttr.needsUpdate = true;

      // Pulse 3D flood tiles on Earth surface
      tile3DMeshes.forEach(function (item, idx) {
        var pulse = item.isCritical ? 0.65 + 0.35 * Math.sin(elapsed * 5.0 + idx) : 0.35;
        item.mat.opacity = chipGridOp * pulse;
      });

      // Target point flash & expanding sonar ripples over India
      var impactActive = smoothstep(0.55, 0.70, ip);
      rippleMeshes.forEach(function (rip, idx) {
        var ph = (elapsed * 0.72 + idx * 0.33) % 1.0;
        rip.scale.setScalar(0.35 + ph * 3.4);
        rip.material.opacity = (1.0 - ph) * impactActive * 0.85;
      });

      // Update Orbital Rings & Miniature Orbiting Satellite
      var orbitOp = sampleSpline(SCROLL_KEYFRAMES.orbitRingOpacity, s);
      orbitTubeMat.opacity = orbitOp * 0.85;
      primaryCoreMat.opacity = orbitOp * 0.70;
      polarTubeMat.opacity = orbitOp * 0.75;
      eqTubeMat.opacity = orbitOp * 0.45;
      orbitNodeMat.opacity = orbitOp * 0.90;

      // Animate Mini Satellite along Primary Orbit Curve
      var miniU = (elapsed * 0.07) % 1.0;
      var miniPos = primaryCurve.getPoint(miniU);
      var miniNext = primaryCurve.getPoint((miniU + 0.01) % 1.0);
      miniSatGroup.position.copy(miniPos);
      miniSatGroup.lookAt(miniNext);
      miniBeacon.scale.setScalar(0.9 + Math.sin(elapsed * 12.0) * 0.4);

      renderer.render(scene, camera);
    }

    animate();
  };
})();
