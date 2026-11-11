BLENDER=tools/blender-5.3.0-linux-x64/blender
CAMS=earth,plaza,drone,nikis
all: fetch process dem preview imagery assets tower geometry scene post web
fetch: ; python3 stages/01_fetch.py
process: ; python3 stages/02_process.py
dem: ; python3 stages/03_dem.py
preview: ; python3 stages/04_preview.py
imagery: ; python3 stages/05_imagery.py
assets: ; python3 stages/05b_assets.py && python3 stages/05c_paving.py
geometry: ; python3 stages/06_geometry.py
tower: ; python3 stages/07_tower.py
scene: ; $(BLENDER) --background --python-exit-code 1 --python stages/08_scene.py -- --cams $(CAMS) --resx 1600 --samples 64 --exposure -0.35 --sun-az 225 --save
post: ; python3 stages/09_post.py $(subst $(comma), ,$(CAMS))
web: ; python3 stages/10_web.py
comma := ,
