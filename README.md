# Thessaloniki in 3D: the White Tower

A 3 km square of Thessaloniki around the White Tower, built from map data with Python and Blender (headless).
One script per stage; each reads and writes files under `data/`.

| Stage | Script | What it does |
|---|---|---|
| 1 | `stages/01_fetch.py` | OpenStreetMap data for the box in `config.json` |
| 2 | `stages/02_process.py` | Buildings, roads, sea, parks in local metres; building heights |
| 3 | `stages/03_dem.py` | Ground elevation (SRTM via OpenTopoData) |
| 4 | `stages/04_preview.py` | Check map |
| 5 | `stages/05_imagery.py`, `05b_assets.py` | Satellite basemap; sky HDRI and PBR textures (Poly Haven) |
| 6 | `stages/06_geometry.py` | Ground, buildings, trees, plaza, cars as plain arrays |
| 7 | `stages/07_tower.py` | The White Tower, built in code |
| 8 | `stages/08_scene.py` | Runs in Blender: materials, light, cameras, render |
| 9 | `stages/09_post.py` | Haze, glow, grain |

## Run

Needs Python 3.11 with `shapely numpy matplotlib pillow mapbox-earcut`, and Blender 5.3 unpacked in `tools/`.

```bash
make all
```

## Limits

- Most building heights are estimates by building type. For visualization, not planning.
- Tower sizes: drum 23 m across and 27 m high, turret 12 m across (published figures). Window, arch and merlon counts are read off photos.

## Data and licences

- Map data: OpenStreetMap contributors, ODbL.
- Elevation: NASA SRTM via OpenTopoData.
- Satellite imagery: Esri World Imagery. For showing only, not for sale.
- Sky and textures: Poly Haven, CC0.
