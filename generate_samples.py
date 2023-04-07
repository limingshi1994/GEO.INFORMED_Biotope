import matplotlib
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import openeo
import argparse
from tqdm import tqdm

import os
import re
import random
import shapefile
import shapely
import pyproj

import shapely.ops as sp_ops


DEFAULT_FIGSIZE = (5, 4)


def pil_cm(im, cm="viridis"):
    cm_hot = matplotlib.colormaps.get_cmap(cm)
    im = np.array(im)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    return im


def trans_paste(fg_img, bg_img, alpha=1.0, box=(0, 0)):
    fg_img_trans = Image.new("RGBA", fg_img.size)
    fg_img_trans = Image.blend(fg_img_trans, fg_img, alpha)
    bg_img.paste(fg_img_trans, box, fg_img_trans)
    return bg_img


def show_tiff(
    filename: str,
    figsize=DEFAULT_FIGSIZE,
    vmin=None,
    vmax=None,
    rescale_percentile=97,
    add_colorbar=False,
):
    """Small helper to load a geotiff and visualize it"""
    with rasterio.open(filename) as ds:
        data = ds.read()
    ret_data = data
    fig, ax = plt.subplots(figsize=figsize)

    if len(data.shape) == 3:
        if data.max() > 500:
            p = np.percentile(data, rescale_percentile, axis=[1, 2])
            data = data / p[:, None, None]
            data = np.clip(data, 0, 1)
        data = np.moveaxis(data, 0, 2)

    im = ax.imshow(data, vmin=vmin, vmax=vmax)
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.05)
    return


def load_tiff(filename: str):
    """Small helper to load a geotiff"""
    with rasterio.open(filename) as ds:
        data = ds.read()
    return data


def show_array(
    data,
    figsize=DEFAULT_FIGSIZE,
    vmin=None,
    vmax=None,
    rescale_percentile=97,
    add_colorbar=False,
):
    fig, ax = plt.subplots(figsize=figsize)
    if len(data.shape) == 3:
        if data.max() > 500:
            p = np.percentile(data, rescale_percentile, axis=[1, 2])
            data = data / p[:, None, None]
            data = np.clip(data, 0, 1)
        data = np.moveaxis(data, 0, 2)

    im = ax.imshow(data, vmin=vmin, vmax=vmax)
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.05)


def random_points_in_polygon(polygon, k):
    "Return list of k points chosen uniformly at random inside the polygon."
    areas = []
    transforms = []
    for t in shapely.ops.triangulate(polygon):
        areas.append(t.area)
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        transforms.append([x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0])
    points = []
    for transform in random.choices(transforms, weights=areas, k=k):
        x, y = [random.random() for _ in range(2)]
        if x + y > 1:
            p = shapely.geometry.Point(1 - x, 1 - y)
        else:
            p = shapely.geometry.Point(x, y)
        points.append(shapely.affinity.affine_transform(p, transform))
    return points


def transform_geom_to_srid(geom, scrs, tcrs):
    """Transform a geometry to a new target CRS.
    Works with pyproj version >= 2.x.x
    - geom is a Shapely geometry instance
    - scrs is your input CRS EPSG integer (the one of your original data)
    - tcrs is your target CRS EPSG integer (the one you want to reproject
              your data in, probably 4326 in your case)
    """
    project = pyproj.Transformer.from_crs(
        "EPSG:" + str(scrs), "EPSG:" + str(tcrs), always_xy=True
    )
    return sp_ops.transform(project.transform, geom)


def random_points_in_country(shp_location, country_name, num_points=100, tcrs="4326"):
    shapes = shapefile.Reader(shp_location)  # reading shapefile with pyshp library
    shapeRecs = shapes.shapeRecords()
    countries = [
        s for s in shapes.records() if country_name in s
    ]  # getting feature(s) that match the country name
    country_ids = [
        int(re.findall(r"\d+", str(item))[0]) for item in countries
    ]  # getting feature(s)'s id of that match

    features = [
        shapeRecs[country_id].shape.__geo_interface__ for country_id in country_ids
    ]
    shp_geoms = [shapely.geometry.shape(feature) for feature in features]
    # shp_geoms = [transform_geom_to_srid(shape, scrs="4326", tcrs="31370")]
    if tcrs != "4326":
        # e.g. tcrs = "31370" for Lambert
        shp_geoms = [
            transform_geom_to_srid(shape, scrs="4326", tcrs=tcrs) for shape in shp_geoms
        ]
    shp_geom = shapely.ops.unary_union(shp_geoms)

    points = random_points_in_polygon(shp_geom, num_points)
    return [(p.y, p.x) for p in points]


def polygonsToRaster(to_raster, save_path):
    to_raster.to_crs(epsg="32631", inplace=True)
    to_raster_raster = make_geocube(
        vector_data=to_raster, measurements=["label"], resolution=(-10, 10), fill=0
    )
    to_raster_raster.rio.to_raster(save_path)


def main():
    # The arguments - later use argparse
    show_overlay = True
    save_dir = "generated_samples"
    resource_dir = "resources"
    num_samples = 50
    location = "Flemish Region"
    patch_width = 0.025
    patch_height = 0.0125
    bands = ["B04", "B03", "B02"]
    dates = ("2021-03-06", "2021-03-07")
    # The arguments - later use argparse

    shp_location = os.path.join(
        resource_dir,
        "ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp",
    )
    points = random_points_in_country(shp_location, location, num_points=num_samples)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eoconn = openeo.connect("openeo.vito.be").authenticate_oidc()

    cal_census = gpd.read_file(os.path.join(resource_dir, "BVM_labeled.zip"))
    cal_census = cal_census.rename(columns={"Label": "label"})
    cal_census.set_crs(epsg="31370", inplace=True)
    cal_census.to_crs(epsg="4326", inplace=True)

    for i, loc_center in tqdm(enumerate(points)):
        frame = {}
        frame["west"] = loc_center[1] - patch_width / 2
        frame["east"] = loc_center[1] + patch_width / 2
        frame["north"] = loc_center[0] + patch_height / 2
        frame["south"] = loc_center[0] - patch_height / 2
        loc_rectangle = shapely.geometry.Polygon(
            [
                (frame["west"], frame["south"]),
                (frame["west"], frame["north"]),
                (frame["east"], frame["north"]),
                (frame["east"], frame["south"]),
                (frame["west"], frame["south"]),
            ]
        )

        cube = eoconn.load_collection("TERRASCOPE_S2_TOC_V2", bands=bands)
        cube = cube.filter_bbox(bbox=loc_rectangle)
        cube = cube.filter_temporal(dates)

        cube.download(os.path.join(save_dir, f"sat_{i}.tiff"), format="GTIFF")

        loc_clipped = gpd.clip(cal_census, loc_rectangle)
        polygonsToRaster(loc_clipped, save_path=os.path.join(save_dir, f"gt_{i}.tiff"))

        if show_overlay:
            data_gt = load_tiff(os.path.join(save_dir, f"gt_{i}.tiff"))
            data_gt = np.uint8(data_gt[0] * 255 / data_gt.max())
            img_gt = Image.fromarray(data_gt, mode="L")
            img_gt = pil_cm(img_gt)

            data_sat = load_tiff(os.path.join(save_dir, f"sat_{i}.tiff"))
            if len(data_sat.shape) == 3:
                if data_sat.max() > 500:
                    p = np.percentile(data_sat, 97, axis=[1, 2])
                    data_sat = data_sat / p[:, None, None]
                    data_sat = np.clip(data_sat, 0, 1)
                data_sat = np.moveaxis(data_sat, 0, 2)
            data_sat = np.uint8(data_sat * 255 / data_sat.max())
            img_sat = Image.fromarray(data_sat, mode="RGB")

            img_ol = trans_paste(img_gt, img_sat, alpha=0.6)
            img_ol.save(os.path.join(save_dir, f"vis_{i}.png"))


if __name__ == "__main__":
    main()
