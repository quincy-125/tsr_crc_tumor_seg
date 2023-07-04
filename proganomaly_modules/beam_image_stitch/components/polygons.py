from apache_beam.io.gcp import gcsio
from google.cloud import bigquery
import geopandas as gpd
import geojson
import logging
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from PIL import ImageOps
import tensorflow as tf
from shapely.geometry import Point, Polygon
from skimage.measure import find_contours
from skimage.morphology import square, binary_dilation
import sys
from typing import Union

from typing import Tuple

from . import bigquery_utils as bq

nucleus_tumor_threshold_default = 7.0

######################################################
# Coordinate Reference System (CRS)-related functions:

# Helpful links for understanding CRS projections:
#   - https://sites.google.com/a/wisc.edu/wiscsims-micro-qgis/learn-to-use-qgis/map-creation-basics/coordinate-reference-systems
#   - https://eos.org/science-updates/making-maps-on-a-micrometer-scale
# We use an "Equal Earth" projection (https://proj.org/operations/projections/eqearth.html),
# which is an "equal area" projection (i.e., area calculations will not change
# when reprojecting geoms). It uses WGS84, which BigQuery requires
# (https://cloud.google.com/bigquery/docs/geospatial-data)
base_crs = "+proj=eqearth +datum=WGS84 +ellps=WGS84"


def wgs84_crs(
    microns_per_pixel: float, projecting_from_slide_scan_to_meter: bool = True
) -> str:
    """
    Create a Coordinate Reference System (CRS) projection string that will
    translate from pixels into meters, the standard unit of measure for WGS84
    geographic coordinates (which BigQuery assumes all GEOGRAPHY data points
    to be, as documented at
    https://cloud.google.com/bigquery/docs/geospatial-data).

    Args:
        microns_per_pixel (float): The number of microns a pixel (or, put
        differently, that one unit of current measurement) represents. Using
        OpenSlide, for example, one can retrieve this value from a slide image
        using `openslide.OpenSlide(slide_uri).mpp-x` or
        `openslide.OpenSlide(slide_uri).mpp-y`.

        projecting_from_slide_scan_to_meter (bool, optional): Whether one is
        projecting from pixels to meters (True), or from meters to pixels
        (False). Defaults to True.

    Returns:
        string: A CRS Projection ("PROJ") string, as documented at
        https://proj.org/.
    """
    pixels_per_micron = (
        (1 / microns_per_pixel)
        if projecting_from_slide_scan_to_meter is True
        else microns_per_pixel
    )
    # 1 meter = 1e+6 micron.
    return f"+proj=eqearth +lon_0=0 +datum=WGS84 +ellps=WGS84 +to_meter={pixels_per_micron * (1e6 if projecting_from_slide_scan_to_meter is True else 1e-6)} +no_defs"


def reproject_polygons_around_meter_scale(
    polygons: gpd.GeoSeries,
    microns_per_pixel: float,
    projecting_from_slide_scan_to_meter: bool = True,
):
    """
    Change a GeoPandas GeoSeries' Coordinate Reference System (CRS) such that
    geom units are changed from pixels to meters, or from meters to pixels.
    (Meters are the standard unit of measure for BigQuery, which uses WGS84 and
    considers all `GEOGRAPHY`-type data to be "a point set on the Earth's
    surface," as stated in the documentation at
    https://cloud.google.com/bigquery/docs/geospatial-data).

    Args:
        polygons (geopandas.GeoSeries): A GeoPandas GeoSeries comprising
        polygons to convert from one scale (pixels vs. meters) to the other.

        microns_per_pixel (float): The number of microns a pixel (or, put
        differently, that one unit of current measurement) represents. Using
        OpenSlide, for example, one can retrieve this value from a slide image
        using `openslide.OpenSlide(slide_uri).mpp-x` or
        `openslide.OpenSlide(slide_uri).mpp-y`.

        projecting_from_slide_scan_to_meter (bool, optional): Whether one is
        projecting from pixels to meters (True), or from meters to pixels
        (False). Defaults to True.

    Returns:
        geopandas.GeoSeries: The input polygons, transformed to the new scale.
    """
    return polygons.set_crs(base_crs).to_crs(
        crs=wgs84_crs(
            microns_per_pixel,
            projecting_from_slide_scan_to_meter=projecting_from_slide_scan_to_meter,
        )
    )


######################################################


def get_dimensions_of_polygon(
    polygon: Polygon, sort: bool = False
) -> Tuple[float, float]:
    """
    Find the length of the sides of the smallest rectangle that can be
    constructed to fit a polygon.

    Args:
        polygon (shapely.geometry.Polygon): A polygon (of any shape).

        sort (bool, optional): Whether to sort the sizes of sides in the output
        tuple. Defaults to False.

    Returns:
        Tuple[float, float]: A tuple of lengths (in the scale of polygon)
        of the sides of the smallest rectangle that can be constructed to fit
        the polygon.
    """
    # See, e.g., https://gis.stackexchange.com/a/359025 for an explanation.
    # Find the edge distances in the minimum rectangle needed to fit around
    # polygon:
    if polygon.is_empty:
        return (0, 0)

    rectangle = polygon.minimum_rotated_rectangle
    x, y = rectangle.exterior.coords.xy
    # Return a tuple of distances:
    distance1 = Point(x[0], y[0]).distance(Point(x[1], y[1]))
    distance2 = Point(x[1], y[1]).distance(Point(x[2], y[2]))
    if sort is True:
        return (min(distance1, distance2), max(distance1, distance2))

    return (distance1, distance2)


def plot_slide_and_polygons(
    anomalies: gpd.GeoSeries,
    thumbnail: np.array,
    output_uri: str,
    figsize_factor: int = 20,
    show_axes: bool = False,
) -> None:
    """
    Plot a collection of polygons on top of a thumbnail of a Whole Slide Image.

    Args:
        anomalies (geopandas.GeoSeries): A collection of polygons (for example,
        representing anomalous regions on the slide image).

        thumbnail (np.array): A thumbnail on top of which to plot anomalies.

        output_uri (str): A GCS path ('gs://...') to write to.

        figsize_factor (int, optional): Factor to magnify the entire plot
        by at the end of the plotting process (e.g., to make details easier
        to see in certain text editors). Defaults to 1.

        fill_in (bool, optional): Whether to plot polygons as filled-in shapes
        (True) or as outlines (False). Defaults to True.

        show_axes (bool, optional): Whether to display matplotlib axes.
        Defaults to True.

    Returns:
        None: (Nothing is returned, but a plot is created and displayed)
    """
    # Save the figure
    gcs = gcsio.GcsIO()

    with gcs.open(output_uri, mode="w") as f:
        thumbnail_dimensions = thumbnail.shape

        logging.info("Plotting...")
        thumbnail_dimension_ratio = min(thumbnail_dimensions) / max(
            thumbnail_dimensions
        )
        figsize = tuple([x * figsize_factor for x in (1, thumbnail_dimension_ratio)])

        _, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(figsize[0] * 3, figsize[1]))

        anomalies.exterior.plot(ax=ax2, cmap="gist_rainbow")
        anomalies.plot(ax=ax3, cmap="gist_rainbow")

        ax.imshow(
            thumbnail,
            aspect="auto",
            interpolation="bilinear",
            extent=(
                0,  # Left
                thumbnail_dimensions[0],  # Right
                -thumbnail_dimensions[1],  # Bottom
                0,  # Top
            ),
            # origin="upper",
            alpha=0.75,
            cmap="gray",
        )

        ax2.imshow(
            thumbnail,
            aspect="auto",
            interpolation="bilinear",
            extent=(
                0,  # Left
                thumbnail_dimensions[0],  # Right
                -thumbnail_dimensions[1],  # Bottom
                0,  # Top
            ),
            # origin="upper",
            alpha=0.75,
            cmap="gray",
        )

        ax3.imshow(
            thumbnail,
            aspect="auto",
            interpolation="bilinear",
            extent=(
                0,  # Left
                thumbnail_dimensions[0],  # Right
                -thumbnail_dimensions[1],  # Bottom
                0,  # Top
            ),
            # origin="upper",
            alpha=0.75,
            cmap="gray",
        )

        if show_axes is not True:
            ax.axis("off")
            ax2.axis("off")
            ax3.axis("off")

        plt.savefig(f, format="png")


def create_anomaly_polygons(
    my_tuple,
    output_filename: str,
    anomaly_resize_factor: float,
    slide_microns_per_pixel: float,
    min_size: Tuple[int, int] = (2000, 2000),
    polygons_initially_drop_below: float = 0.90,
    output_anomaly_polygon_geoson: bool = False,
    output_anomaly_polygon_visualizations: bool = False,
    output_anomaly_polygon_debugging_pickle: bool = False,
) -> gpd.GeoSeries:
    """Take anomaly flag pixel locations at the stitched slide level and create
    polygons from them at the Whole Slide level, optionally filtering for
    minimum acceptable size (e.g., 2mm x 2mm).

        Args:
            my_tuple: 2-tuple, grid index and dictionary of images & index
            stacks.

            anomaly_resize_factor (float), the ratio of the original Whole
            Slide Image to the slide image that is overlaid on the output from
            the anomaly inference pipeline step. For example, a value of 16.0
            means that anomalies from the anomaly inference pipeline step
            should be expanded by 16 times in order to align with the original
            Whole Slide Image.

            output_filename (str), the output filename in GCS.

            slide_microns_per_pixel (float): The number of microns per pixel
            for the Whole Slide Image.

            min_size: Tuple[int, int], Minimum width and height (or height and
            width) threshold for polygons, taking the width and height of the
            smallest possible rectangle that fully encompases the polygon.
            Expressed in number of microns. If either value is 0 or negative, both
            values will be ignored. Defaults to (2000, 2000).

            slide_level_dimensions (Tuple[Tuple[int, int],...]): A list of
            (width, height) tuples, one for each level of the slide.

            slide_level_downsamples (Tuple[float, ...]): A list of
            downsample factors, one for each level of the slide.

            polygons_initially_drop_below (float): In the initial process of
            creating polygons, drop all polygons below this percentile by area.
            This serves to quickly remove tiny "dots" of anomalous polygons,
            speeding up processing time for subsequent steps. Defaults to 0.90
            (i.e., keep the top 10% of polygons by area).

            output_anomaly_polygon_geoson (bool): Whether to write anomalies
            (filtered based on min_size argument) to a GeoJSON file at the output_filename argument.

            output_anomaly_polygon_visualizations (bool): Whether to write
            visualizations of anomalies (pre- and post-filtering by on min_size
            argument) to a PNG file (or two PNG files -- one pre-filtering, one
            post-filtering -- if min_size is set) in the directory specified in
            the output_gcs_path argument.

            output_anomaly_polygon_debugging_pickle (bool): Whether to write a
            pickle containing intermediate output related to anomaly creation
            to a Pickle file in the directory specified in the output_gcs_path
            argument.

        Returns:
            anomalies (geopandas.GeoSeries): A geopandas.GeoSeries with each
            row comprising a polygon constructed from the image of pixels
            flagged as anomalous. This is scaled to the size of the original
            Whole Slide Image.
    """

    gcs = gcsio.GcsIO()

    _, branch_dict = my_tuple
    anomalous_pixels_img = ImageOps.invert(
        tf.keras.utils.array_to_img(branch_dict[0]["image"])
    )
    anomalous_pixels = np.array(anomalous_pixels_img, "uint8")

    # Dilate all pixels that are > 0, in order to connect them into larger,
    # consistent areas of "brightness":
    anomalous_pixels = binary_dilation(anomalous_pixels, square(4))

    # "Smooth out" the image with a uniform blur:
    anomalous_pixels = ndi.uniform_filter(anomalous_pixels, size=8)

    # Find the contours in the "blurred" flagged pixel image:
    contours = find_contours(anomalous_pixels, level=0.9, fully_connected="high")

    # Convert contours into a GeoPandas DataFrame (and simultaneously
    # rotate the polygons created from contours into their correct
    # orientation, to match the original input Whole Slide Image):
    contours_df = gpd.GeoDataFrame(
        {
            "geometry": gpd.GeoSeries(
                [Polygon(contour) for contour in contours]
            ).rotate(-90, origin=(0, 0))
        }
    )

    if len(contours_df) == 0:
        logging.warning("No initial anomaly contours found.")

        return gpd.GeoSeries()

    # Remove tiny polygons created from contours above:
    min_considerable_area = 200  # An aribtrary number, just to get beyond
    # the large amounts of tiny lit-up areas

    # Retain just the largest 10%ile of polygons created from contours:
    # Using round() here is just to avoid throwing out similarly-sized
    # polygons:
    upper_area_threshold = float(
        round(
            contours_df[contours_df.area > min_considerable_area].area.quantile(
                [polygons_initially_drop_below]
            )
        )
    )
    anomalies = contours_df[contours_df.area >= upper_area_threshold]

    if len(anomalies) == 0:
        logging.warning(
            "No anomaly polygons remaining after initial %d pixel area filter step.",
            min_considerable_area,
        )

        return gpd.GeoSeries()

    # For each of the polygons remaining, subtract the unary union of all
    # other polygons from that polygon. This accounts for "holes" within
    # the polygon, which are, at this stage, represented by other polygons
    # that overlap with a given polygon (since these polygons were all
    # created from contours in the anomaly pixel image):
    # We use GeoPandas' sindex feature to speed up the overlap queries:
    overlaps = anomalies["geometry"].sindex.query_bulk(
        anomalies["geometry"], predicate="intersects"
    )

    overlaps_df = pd.DataFrame(
        {"input_index": overlaps[0], "touching_index": overlaps[1]}
    )
    # Remove self-overlaps:
    overlaps_df = overlaps_df[
        overlaps_df["input_index"] != overlaps_df["touching_index"]
    ]

    grouped_overlaps = (
        overlaps_df.groupby("input_index")
        .agg({"touching_index": lambda x: list(x) if len(x) > 0 else Polygon()})
        .loc[:, "touching_index"]
    )

    # Reset the index of anomalies, to be able to be used with
    # the output of overlaps above, which comprises 0-indexed input indexes
    # and the 0-indexed indexes of all polygons that overlap the polygon
    # at that index:
    anomalies.reset_index(inplace=True, drop=True)

    anomalies = anomalies.assign(touching_indexes=[[] for _ in range(len(anomalies))])
    anomalies.loc[grouped_overlaps.index, "touching_indexes"] = grouped_overlaps

    # After all the setup above, we go ahead and replace the polygon from
    # each row with the polygon minus any overlapping polygons:

    # This is written in a for loop rather than in an apply function to make
    # it easier to read:
    touching_indexes_unions = []
    for _, row in anomalies.iterrows():
        if len(row.get("touching_indexes", [])) > 0:
            touching_indexes_unions.append(
                anomalies.loc[row.get("touching_indexes", []), "geometry"].unary_union
            )
        else:
            touching_indexes_unions.append(Polygon())

    anomalies = gpd.GeoDataFrame.from_features(
        anomalies.difference(gpd.GeoSeries(touching_indexes_unions))
    )

    # Drop empty rows:
    anomalies = anomalies[anomalies["geometry"] != None].reset_index(
        drop=True, inplace=False
    )

    # Filter anomalies that are not a minumum coarse size:
    anomaly_dimensions = anomalies["geometry"].apply(
        get_dimensions_of_polygon, sort=True
    )

    anomalies["min_dimension"], anomalies["max_dimension"] = zip(*anomaly_dimensions)

    # Smooth the boundaries of the poilygons by dilating them slightly, and
    # then immediately un-dilating by the same amount:
    anomalies["geometry"] = anomalies.buffer(10).buffer(-10)

    anomalies.set_crs(
        base_crs,
        allow_override=True,
        inplace=True,
    )

    anomalies_filtered = anomalies

    # Note that the polygons are currently at the scale of the stitched
    # image (e.g., 8192x8192), *not* at the scale of the original Whole
    # Slide Image. Thus, below, we use the ratio of the stitched
    # image's size to the original Whole Slide Image (i.e., the
    # anomaly_resize_factor argument) in order to filter
    # polygons at the current scale. (We filter at the current scale rather
    # than after scaling up in order to be able to create a visualization
    # below at the smaller scale, which is faster than using the scaled-up
    # version that we will create further below):

    if min_size[0] > 0 or min_size[1] > 0:
        logging.info("Filtering polygons...")

        min_size_pixels = (
            min_size[0] / slide_microns_per_pixel / anomaly_resize_factor,
            min_size[1] / slide_microns_per_pixel / anomaly_resize_factor,
        )
        anomalies_filtered = anomalies[
            ((anomalies["min_dimension"]) >= min(min_size_pixels))
            & ((anomalies["max_dimension"]) >= max(min_size_pixels))
        ]

    if output_anomaly_polygon_debugging_pickle:
        pickle_output_filename = output_filename + "_img_array_and_polygons_df.pickle"
        logging.info('Saving output to "%s"...', pickle_output_filename)
        with gcs.open(pickle_output_filename, mode="w") as f:
            pickle.dump(
                {
                    "anomalous_pixels_img": anomalous_pixels_img,
                    "anomalies": anomalies,
                    "anomalies_filtered": anomalies_filtered,
                    "anomaly_resize_factor": anomaly_resize_factor,
                },
                f,
            )

    if output_anomaly_polygon_visualizations:
        # Save a visualization of the polygons, to aid in debugging:
        visualization_filename = output_filename + "_pre-filtering.png"
        logging.info(
            'Creating visualization of all polygons (pre-filtering) and saving it to "%s"...',
            visualization_filename,
        )
        plot_slide_and_polygons(
            anomalies=anomalies,
            thumbnail=np.array(anomalous_pixels_img) > 0,
            output_uri=visualization_filename,
        )
        filtered_visualization_filename = output_filename + "_filtered.png"

        if min_size[0] > 0 or min_size[1] > 0:
            logging.info(
                'Creating visualization of filtered polygons and saving it to "%s"...',
                filtered_visualization_filename,
            )
            plot_slide_and_polygons(
                anomalies=anomalies_filtered,
                thumbnail=np.array(anomalous_pixels_img) > 0,
                output_uri=filtered_visualization_filename,
            )

    if len(anomalies_filtered) == 0:
        logging.warning(
            "No anomalies remaining after filtering (min size: %d).", min_size
        )

        return gpd.GeoSeries()

    logging.info("Scaling polygons by a factor of %d...", anomaly_resize_factor)
    anomalies_full_scale = anomalies_filtered.scale(
        anomaly_resize_factor, anomaly_resize_factor, origin=(0, 0)
    )

    anomalies_unfiltered_full_scale = anomalies.scale(
        anomaly_resize_factor, anomaly_resize_factor, origin=(0, 0)
    )

    if output_anomaly_polygon_geoson:
        # Save the polygons as GeoJSON, to aid in debugging:
        geojson_output_filename = output_filename + ".geojson"
        logging.info('Saving output to "%s"...', geojson_output_filename)
        with gcs.open(geojson_output_filename, mode="w") as f:
            anomalies_full_scale.to_file(f, driver="GeoJSON")

        geojson_unfiltered_output_filename = output_filename + "_unfiltered.geojson"
        logging.info('Saving output to "%s"...', geojson_unfiltered_output_filename)
        with gcs.open(geojson_unfiltered_output_filename, mode="w") as f:
            anomalies_unfiltered_full_scale.to_file(f, driver="GeoJSON")

    return anomalies_full_scale


def upload_anomaly_polygons(
    anomaly_polygons: gpd.GeoSeries,
    slide_uuid: str,
    project_id: str,
    slide_microns_per_pixel: float,
    table_name: str = "anomaly_polygons",
    dataset_name: str = "phi_main_us_p",
    replace_existing_slide_polygons: bool = True,
):
    """
    Take anomaly polygons at the scale of the original Whole Slide Image and
    upload them, and related metadata, to BigQuery if necessary.

    Args:
        anomaly_polygons (geopandas.GeoSeries): A collection of polygons
        representing anomalous regions on
        the slide image). These are sized relative to the output of
        the inference process (e.g., (8192x8192), rather than to the
        original Whole Slide Image.

        slide_uuid (str): A unique, consistent identifier for a slide, used to
        look up that slide's values in BigQuery.

        project_id (str): The name of the BigQuery project to query.

        slide_microns_per_pixel (float): The number of microns per pixel
        for the Whole Slide Image.

        table_name (str, optional): The name of the BigQuery table within
        the dataset_name dataset that contains nuclei data.
        Defaults to "anomaly_polygons".

        dataset_name (str, optional): The name of the BigQuery dataset to query.
        Defaults to "phi_main_us_p".

        replace_existing_slide_polygons (bool, optional): Whether to delete all
        anomaly data associated with slide_uuid before uploading data. Defaults
        to True.

    Returns:
        None: (Nothing is returned, but rows are uploaded to BigQuery)

    """
    if len(anomaly_polygons) == 0:
        logging.info(
            "There are no anomaly polygons to upload. We have still uploaded the slide to BigQuery if it wasn't there already, however."
        )

        return

    logging.info("Uploading %d polygons...", len(anomaly_polygons))

    anomalies = reproject_polygons_around_meter_scale(
        anomaly_polygons,
        microns_per_pixel=slide_microns_per_pixel,
    )

    # See https://cloud.google.com/bigquery/docs/samples/bigquery-insert-geojson
    # for a relevant example.
    dataframe = pd.DataFrame(
        {
            "slide_uuid": slide_uuid,
            # We use geojson here because BigQuery was seeing WKT
            # representations of polygons with holes as invalid (but seems
            # able to parse the JSON representations.)
            "segment": pd.Series(anomalies.apply(geojson.dumps)),
            "partition_number": bq.string_to_partition(slide_uuid),
        }
    )

    bigquery_client = bigquery.Client(project=project_id)

    table = bigquery_client.get_table(f"{project_id}.{dataset_name}.{table_name}")

    # See https://googleapis.dev/python/bigquery/latest/generated/google.cloud.bigquery.client.Client.html#google.cloud.bigquery.client.Client.insert_rows_from_dataframe
    # for documentation:
    errors = bigquery_client.insert_rows_from_dataframe(
        table=table, dataframe=dataframe
    )

    if errors and not all([x == [] for x in errors]):
        raise RuntimeError(f'Error when writing rows to table "{table_name}": {errors}')

    logging.info(
        f"Wrote {len(anomalies)} {'rows' if len(anomalies) > 1 else 'row'} to table \"{table_name}\"."
    )


def upload_nuclei_polygons(
    inference_output,
    slide_uuid: str,
    project_id: str,
    microns_per_pixel: float,
    table_name: str = "nuclear_polygons",
    dataset_name: str = "phi_main_us_p",
):
    """
    Take nuclei segmentation polygons at the scale of the original
    Whole Slide Image and upload them, and related metadata, to BigQuery
    if necessary.

    Args:
        inference_output (dict): A dict containing key "segmentation_cell_coords", a list comprising a collection of polygons, representing nuclei segmentation borders
         on the slide image). These are sized relative to the original
         Whole Slide Image.

        slide_uuid (str): A unique, consistent identifier for a slide, used to
        look up that slide's values in BigQuery.

        project_id (str): The name of the BigQuery project to query.

        microns_per_pixel (float): The number of microns per pixel
        for the Whole Slide Image.

        table_name (str, optional): The name of the BigQuery table within
        the dataset_name dataset that contains nuclei data. Defaults to "anomaly_polygons".

        dataset_name (str, optional): The name of the BigQuery dataset to query.
        Defaults to "phi_main_us_p".

    Returns:
        None: (Nothing is returned, but rows are )

    """

    import geopandas as gpd
    from math import sqrt
    import pandas as pd
    from shapely.validation import make_valid

    geom_list = inference_output["segmentation_cell_coords"]
    logging.info(
        "Yielding cell coordinates from list containing %d geoms)...",
        len(geom_list),
    )

    if len(geom_list) == 0:
        return

    # Re-project to meter scale:
    # (As explained in the docstring above, we do this to aid in the
    # calculations below. BigQuery requires this reprojection, so
    # we do it now, while still operating in parallel, since performing
    # these calculations, while fast for a collection of few nuclei,
    # adds up in time when processing huge numbers of nuclei at once.)
    logging.info("Reprojecting polygons to meter scale...")
    cells_df = gpd.GeoDataFrame(
        geometry=reproject_polygons_around_meter_scale(
            gpd.GeoSeries(geom_list),
            microns_per_pixel=microns_per_pixel,
            projecting_from_slide_scan_to_meter=True,
        )
    )
    logging.info("Finished reprojecting polygons to meter scale...")

    min_area_meters = nucleus_tumor_threshold_default * 1e-6

    logging.info("Calculating nuclei centroids and areas...")
    cells_df["centroid"] = cells_df.centroid.apply(geojson.dumps)
    cells_df["diameter"] = cells_df["geometry"].apply(
        # Get the minimum rectangle we can draw around the geom, and
        # take the largest of the dimensions of that rectangle. We will call
        # that the "diameter" of the geom (whatever the geom's shape):
        lambda geom: get_dimensions_of_polygon(geom, sort=True)[1]
    )
    cells_df["is_tumor"] = cells_df["diameter"] > min_area_meters

    # Confirm that all polygons are valid:
    logging.info("Validating polygons...")
    cells_df["geometry"] = cells_df["geometry"].apply(make_valid)
    logging.info("Finished validating polygons...")

    # See https://cloud.google.com/bigquery/docs/samples/bigquery-insert-geojson
    # for a relevant example.
    dataframe = pd.DataFrame(
        {
            "slide_uuid": slide_uuid,
            "partition_number": bq.string_to_partition(slide_uuid),
            "centroid": cells_df["centroid"],
            "diameter": cells_df["diameter"],
            "is_tumor": cells_df["is_tumor"],
        }
    )

    bigquery_client = bigquery.Client(project=project_id)

    table = bigquery_client.get_table(f"{project_id}.{dataset_name}.{table_name}")

    # See https://googleapis.dev/python/bigquery/latest/generated/google.cloud.bigquery.client.Client.html#google.cloud.bigquery.client.Client.insert_rows_from_dataframe
    # for documentation:
    errors = bigquery_client.insert_rows_from_dataframe(
        table=table, dataframe=dataframe
    )

    if errors and not all([x == [] for x in errors]):
        raise RuntimeError(f'Error when writing rows to table "{table_name}": {errors}')

    logging.info(
        f"Wrote {len(cells_df)} {'rows' if len(cells_df) > 1 else 'row'} to table \"{table_name}\"."
    )
