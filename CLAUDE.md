# Thessaloniki 3D: notes for picking the work up

The user writes in Greek; answer in Greek. They send reference photos and want the model to match them closely.

## What this is
A 3 km square of Thessaloniki built from OpenStreetMap data with Python, rendered in Blender (headless, Cycles on CPU),
plus a Three.js viewer in `docs/` served by GitHub Pages at https://elioar.github.io/thessaloniki-3d/ (branch `main`, folder `/docs`).
Two areas are detailed: the White Tower and its park, and Aristotelous Square about 930 m to the north-west.

## Rebuilding from a fresh clone
`data/`, `tools/` and the `.blend` are not in the repo (too large). They are rebuilt:
1. `pip install shapely numpy matplotlib pillow mapbox-earcut`
2. Unpack Blender 5.3 into `tools/` (https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz).
3. `make all`: has never been run end to end on a clean machine. Stages were run one at a time.
   The Aristotelous imagery layer was first fetched by hand and only later added to `stages/05_imagery.py`; that line is untested.
Network notes: Geofabrik and overpass-api.de are unreachable from the cloud container; `01_fetch.py` uses the OSM API in small cells instead.

## Stages
`01_fetch` OSM · `02_process` layers + heights · `03_dem` elevation · `04_preview` check map · `05_imagery` satellite ·
`05b_assets` Poly Haven sky/textures · `05c_paving` paving textures made in code · `06_geometry` all geometry as .npz arrays
(buildings, trees, tower park, Aristotelous Square) · `07_tower` the White Tower · `08_scene` Blender scene, cameras, render ·
`09_post` haze/glow · `10_web` export to `docs/`.

Render a view: `tools/blender-5.3.0-linux-x64/blender --background --python-exit-code 1 --python stages/08_scene.py -- --cams aris_north,drone --resx 1600 --samples 48 --exposure -0.6 --sun-az 225`
then `python3 stages/09_post.py aris_north drone`. Camera names are in `CAMDEF` in `08_scene.py`. A frame takes roughly 5 minutes on 4 CPU cores.

## State and known gaps
- White Tower: shape, crown and stone done from photos. Level difference between tower and road (steps, retaining wall) is not modelled.
- Tower park: paving zones, stone slabs, stelae line, planters, pines, people, boats done. Shapes of some planters come straight from the map and are off.
- Aristotelous Square: 26 buildings with arcades, balconies, loggias, pergolas or tiled roofs. Facades are too white (should be cream, with ochre
  on the flank buildings); the large red paving panel near Mitropoleos did not appear in renders; the discs in the half-round plaza came out grey, not black;
  flank buildings use a random mix of features, not their real facades; the Aristotle statue's position is a guess and was never checked in a render.
- Most building heights across the city are estimates by building type.
- Web viewer: 85 MB, uses a lighter version of the square's buildings. Recent changes there were never checked in a browser.
- Satellite imagery is Esri World Imagery: for showing only, not for sale.
