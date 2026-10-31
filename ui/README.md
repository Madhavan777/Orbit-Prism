# ORBIT-PRISM // AETHER-IX Orbital Intelligence UI

Interactive 3D Mission Intelligence & Visual Triage Dashboard for ORBIT-PRISM.

## Architecture

- **Component:** `aether_ix_orbital_intelligence.tsx` (also mirrored at `web/` and root)
- **Engine:** React 18+ with single-canvas persistent Three.js (60 FPS ACES Filmic Tone Mapping)
- **Visuals:**
  - Procedural Kapton gold foil satellite bus with articulated parabolic dish and dual solar wings
  - Continuous 8-section spline choreography with smooth camera & Earth nadir lock (India: 20.5937°N, 78.9629°E)
  - Razor-thin coherent white optical laser link with photonic telemetry packets
  - 512×512 satellite flood chip with interactive 4×4 Acid Lime (`#C6FF00`) priority grid
  - Sentinel-1 SAR vs Sentinel-2 Optical cloud-penetration toggle
  - Conformal risk calibration dial (1%, 5%, 10% miss-rate targets)

## Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "three": "^0.160.0",
    "lucide-react": "^0.300.0"
  }
}
```
