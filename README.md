# GEO.INFORMED_Biotope
To use the script to rasterize the ground truth data:

1.Download the zip file of the shapefile of the ground truth data through: https://mega.nz/file/VAJWWRiR#30npVW85UAWEp_wCYWu-E4mltfmu_uCfxytSdXeV9aE

2.Run the script to rasterize polygons-defined surface labels. Or to directly use prepared one: https://mega.nz/file/pYYhUbpK#24xV9VHyIwsEB6v3-0Uj1pEXzcCl4RUfu2zcooNpFZU

3.The script also comes with an overlay function which examines the validity of raster patches against original polygons by overlaying and checking sharpness of corresponding boundaries.

4.To check desired locations in Flanders region, please provide the script with sentinel-2 satellite and rasterized ground truth tiff images. To check the outcome of overlay, use "loc_raster.tiff" and "to_use.tiff": 
https://mega.nz/file/cAAiQTSZ#DoMplYmpdpnT4_pQVmekyOc0p95bv55afB_O5Op_0K0
https://mega.nz/file/BEAkhJIY#LRKuBH9cKULr8QI1utqyV4R25Foz7MIiRPp65erY5gM
