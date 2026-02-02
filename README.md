# attribution_tutorial



To export pix env to environment.yml for binder, use: 
pixi project export conda-environment environment.yml

then you need to add:
- python>=3.14
to the first of the list of dependencies, otherwise you get an error when building as they default to python 3.10