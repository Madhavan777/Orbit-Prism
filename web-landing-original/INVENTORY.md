# AETHER-IX Orbital Intelligence // Landing Page Inventory

## 1. Stack & Architecture
- **Framework**: React 18+ (TSX), TypeScript
- **Styling**: Tailwind CSS utility classes + custom CSS variables
- **3D Graphics Engine**: Three.js WebGL Renderer (ACES Filmic Tone Mapping, 60 FPS requestAnimationFrame loop)
- **Iconography**: Lucide React (Satellite, Radio, Globe, Compass, Activity, Crosshair, Wifi, Terminal, etc.)
- **Execution Target**: Single-canvas persistent WebGL background overlay with zero scroll-jacking.

## 2. Color System & Design Tokens
- Midnight Black: `#05070D` (Page background)
- Deep Navy: `#0A1020` (Surfaces, panels, HUD cards)
- Cobalt Blue: `#245BFF` (Secondary telemetry, radar channels, link arcs)
- Acid Lime: `#C6FF00` (Critical flood inundation alert, high priority, L2 packets)
- Pale Ice / White: `#F4F7FF` (Primary typography)
- Pure Laser White: `#FFFFFF` (Coherent optical laser core)
- Muted Grey: `#8B95A7` (Metadata, units, coordinates)

## 3. Sections
1. **Hero (`#sec-hero`)**: Satellite reveal, coherent white laser lock onto India (20.5937°N, 78.9629°E), live coordinates and status telemetry pill.
2. **Mission Architecture (`#sec-mission`)**: Edge-first orbital triage, onboard segmentation (<38 ms), priority queuing (L2→L1→L0), swath metrics (290 km swath, 10 m/px).
3. **Observation Corridor (`#sec-observation`)**: Subcontinent horizon scan, AOI Eastern India delta, river basin telemetry (Brahmaputra, Mahanadi, Godavari).
4. **Flood Chip Segmentation (`#sec-flood`)**: 512×512 satellite chip split into sixteen 128×128 tiles with interactive 4×4 Acid Lime flood lock and per-tile inspection.
5. **Optical & Radar Intelligence (`#sec-sensors`)**: Autonomous cloud-cover simulation slider, automatic switch to Sentinel-1 C-Band SAR microwave penetration at cloud cover > 30%.
6. **Data Downlink (`#sec-downlink`)**: L0/L1/L2 tiered telemetry stream filtering, laser packet transmission latency.
7. **Conformal Risk Calibration (`#sec-calibration`)**: Miss-rate target selectors (1%, 5%, 10%), empirical recall curves, threshold τ, bandwidth conservation metrics.
8. **Orbital Horizon & Footer (`#sec-footer`)**: Continuous orbital rotation, back to top orbit, team credits.

## 4. Fonts, Images & Textures
- **Fonts**: System sans-serif (`system-ui, -apple-system, sans-serif`) and monospace (`ui-monospace, "Cascadia Mono", monospace`).
- **Textures**: Procedural HTML5 2D Canvas textures:
  - Procedural Kapton gold foil texture (specular wrinkles, gridlines)
  - Procedural solar cell panel array texture (deep blue silicon cells, silver busbars)
  - Procedural high-contrast Earth Day map, Night city lights emissive map, and procedural cloud layer map.
- **Images**: Procedural SVG vector flood basin plume and riverbed geometry.

## 5. Animations & Keyframes
- 5-step satellite laser introduction sequence (intro progress 0 to 1).
- 8-section Catmull-Rom continuous cubic spline interpolation mapping scroll s ∈ [0, 1] to camera positions, Earth positions/orientations, satellite poses, and atmospheric Fresnel glow.
- Razor-thin coherent laser beam with traveling photonic data packets.
- Interactive 3D orbit drag inspection toggle with dampening.
