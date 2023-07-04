# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import apache_beam as beam
import argparse
import datetime
import logging
from math import ceil, log
import os
from openslide import OpenSlide
from pathlib import Path
import subprocess
import sys

from typing import Tuple

# from components import confusion_matrix
from components import bigquery_utils as bq
from components import images
from components import inference
from components import patch_coordinates
from components import polygons
from components import pre_inference_png
from components import pre_inference_wsi
from components import segmentation

# Disable GPUs (for our DLVMs):
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def nary_depth(
    slide_dimensions: Tuple[int, int], patch_height: int, patch_width: int
) -> int:
    """Calculate the "n-ary tree depth" of a slide for GAN image "stitching" --
    i.e., given the dimensions of a slide and the dimensions of the patches we
    will create from that  image, the number of times we will need to combine
    four patches, or groups of patches, or groups of groups of patches, etc.
    to reconstitute the original slide.

    Example: If we have a slide that is 86000 pixels x 112000 pixels, and are
    creating 1024 pixel x 1024 pixel patches from it, we will create
    83.984 ~ 84 patches the x dimension and 109.375 ~ 110 patches in the y
    dimension.

    Currently, the process of "stitching" patches back together into the
    original slide image comprises taking four adjacent patches, and
    making a new image from them:

    Patch 1 | Patch 2
    --------|--------
    Patch 3 | Patch 4

    Then, we take four of those adjacent patch groups, and create a new image
    from them:

    Patch 1  | Patch 2  || Patch 13 | Patch 14
    ---------|--------- || ---------|---------
    Patch 3  | Patch 4  || Patch 15 | Patch 16
    =========|==========||==========|=========
    Patch 9  | Patch 10 || Patch 17 | Patch 18
    ---------|--------- || ---------|---------
    Patch 11 | Patch 12 || Patch 19 | Patch 20

    Etc.

    For the example dimensions given above, we would need 7 iterations / a
    "depth" of 7 to capture all of the patches created from the slide:

    - Depth: 0, Size: 1 set of patches x 1 set of patches
    - Depth: 1, Size: 2 sets of patches x 2 sets of patches
    - Depth: 2, Size: 4 sets of patches x 4 sets of patches
    - Depth: 3, Size: 8 sets of patches x 8 sets of patches
    - Depth: 4, Size: 16x16
    - Depth: 5, Size: 32x32
    - Depth: 6, Size: 64x64
    - Depth: 7, Size: 128x128

    Put differently, this process will take log(128, 2) = 7 iterations, or a
    depth of 7 of a 4-ary tree.

    Args:
        slide_dimensions (Tuple[int, int]): The height and width of the
        original slide image.

        patch_height (int): The desired height, in pixels, of each patch
        created from the slide.

        patch_width (int): The desired width, in pixels, of each patch
        created from the slide.

    Returns:
        (int): The number of iterations, or depth of tree, needed to stitch
        all slides together back into the original slide image.
    """

    n_patches_width = ceil(slide_dimensions[0] / patch_width)
    n_patches_height = ceil(slide_dimensions[1] / patch_height)
    logging.info(f"n_patches_width: {n_patches_width}")
    logging.info(f"n_patches_height: {n_patches_height}")
    return ceil(max((log(n_patches_width, 2), log(n_patches_height, 2))))


def calculate_anomaly_resize_factor(
    wsi_dimensions: Tuple[int, int],
    stitched_output_image_size: Tuple[int, int],
    patches_width: int,
    patches_height: int,
    nary_depth: int,
):
    """
    Calculate the ratio of (1) the size of the slide placed on the (e.g.,
    8192x8192 px) "flagged/anomalous pixels" image created by the anomaly
    inference pipeline step, to (2) the size of the original slide.

    This ratio is needed to resize anomaly polygons (which are created in
    create_anomaly_polygons() from the inference pipeline step output) such that
    they align with, and are the appropriate size for, the original
    Whole Slide Image.

    Args:
        wsi_dimensions (Tuple[int, int]): The width and height, in pixels, of
        the Whole Slide Image.

        stitched_output_image_size (Tuple[int, int]): The width and height, in
        pixels, of the image output from the anomaly inference pipeline step.

        patches_width (int): The width, in pixels, of patches created by the
        anomaly inference pipeline step.

        patches_height (int): The height, in pixels, of patches created by the
        anomaly inference pipeline step.

        nary_depth (int):
            The "n-ary tree depth" of a slide for GAN image "stitching" --
            i.e., given the dimensions of a slide and the dimensions of the patches we
            will create from that  image, the number of times we will need to combine
            four patches, or groups of patches, or groups of groups of patches, etc.
            to reconstitute the original slide.

            Example: If we have a slide that is 86000 pixels x 112000 pixels, and are
            creating 1024 pixel x 1024 pixel patches from it, we will create
            83.984 ~ 84 patches the x dimension and 109.375 ~ 110 patches in the y
            dimension.

            Currently, the process of "stitching" patches back together into the
            original slide image comprises taking four adjacent patches, and
            making a new image from them:

            Patch 1 | Patch 2
            --------|--------
            Patch 3 | Patch 4

            Then, we take four of those adjacent patch groups, and create a new image
            from them:

            Patch 1  | Patch 2  || Patch 13 | Patch 14
            ---------|--------- || ---------|---------
            Patch 3  | Patch 4  || Patch 15 | Patch 16
            =========|==========||==========|=========
            Patch 9  | Patch 10 || Patch 17 | Patch 18
            ---------|--------- || ---------|---------
            Patch 11 | Patch 12 || Patch 19 | Patch 20

            Etc.

            For the example dimensions given above, we would need 7 iterations / a
            "depth" of 7 to capture all of the patches created from the slide:

            - Depth: 0, Size: 1 set of patches x 1 set of patches
            - Depth: 1, Size: 2 sets of patches x 2 sets of patches
            - Depth: 2, Size: 4 sets of patches x 4 sets of patches
            - Depth: 3, Size: 8 sets of patches x 8 sets of patches
            - Depth: 4, Size: 16x16
            - Depth: 5, Size: 32x32
            - Depth: 6, Size: 64x64
            - Depth: 7, Size: 128x128

            Put differently, this process will take log(128, 2) = 7 iterations, or a
            depth of 7 of a 4-ary tree.

            This can be calculated using nary_depth().

    Returns:
        float: The ratio of the size of the Whole Slide Image to the slide
        on the output from the anomaly inference pipeline step. For example, a
        value of 16.0 means that anomaly polygons from the inference pipeline
        step should be expanded by 16 times to align with the original Whole
        Slide Image.
    """
    num_patches_width = ceil(wsi_dimensions[0] / patches_width)
    num_patches_height = ceil(wsi_dimensions[1] / patches_height)

    max_patch_number = max(num_patches_width, num_patches_height)
    max_patches_dim = "width" if num_patches_width > num_patches_height else "height"

    # The max image dimension, plus whatever's left from patches being the
    # size they are:
    effective_image_dim = max_patch_number * (
        patches_width if max_patches_dim == "width" else patches_height
    )

    # The percent of the output stitched image that the overlaid slide will
    # comprise:
    percent_of_stitched_image = max_patch_number / (2 ** nary_depth)

    stitched_image_max_pixels = (
        stitched_output_image_size[0 if max_patches_dim == "width" else 1]
        * percent_of_stitched_image
    )

    effective_resize_factor = effective_image_dim / stitched_image_max_pixels

    return effective_resize_factor


def resolve_gcs_uri_to_local_uri(gcs_file_uri: str) -> str:
    """
    Use an existing gcsfuse mountpoint, or use gcsfuse to mount a bucket
    locally if not mountpoint exists, to get a local filepath for a file stored
    in a GCS bucket.

    Args:
        gcs_file_uri (str): A "gs:/..." URI for a file stored in a GCS bucket.

    Returns:
        str: A string indicating a local filepath where gcs_file_uri can be
        found.
    """
    if not gcs_file_uri.startswith("gs:/"):
        raise Exception('Please use a "gs:/" GCS path.')

    # Pathlib isn't wholly apposite to use for URLs (specifically, it changes
    # 'gs://...' to 'gs:/...'), but it's useful in this case:
    gcs_path = Path(gcs_file_uri)
    bucket_name = gcs_path.parts[1]
    file_remaining_path = Path(*gcs_path.parts[2:])
    logging.info("Checking for existing mountpoint for %s...", bucket_name)
    existing_mountpoint_check = subprocess.run(
        f'findmnt --noheadings --output TARGET --source "{bucket_name}"',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    existing_mountpoint_check_stdout = existing_mountpoint_check.stdout.decode().strip()

    if existing_mountpoint_check.returncode == 127:
        raise Exception(
            f'Error when running findmnt: "{existing_mountpoint_check_stdout}"'
        )

    if (
        existing_mountpoint_check.returncode == 0
        and existing_mountpoint_check_stdout != ""
    ):
        # An existing mountpoint for the bucket exists; we will reuse it here,
        # if it contains the file we're looking for (it's possible that the
        # bucket is mounted, but to a different subdirectory within the bucket
        # than we need).
        for mountpoint in existing_mountpoint_check_stdout.split("\n"):
            logging.info(
                'Checking for file "%s" at "%s"...', file_remaining_path, mountpoint
            )
            if Path(mountpoint, file_remaining_path).exists():
                found_file_path = os.path.join(mountpoint, file_remaining_path)
                logging.info('Found file at "%s"', found_file_path)
                return found_file_path

    # No existing mountpoint exists, so we will mount the bucket with gcsfuse:
    import tempfile

    tmp_directory = tempfile.mkdtemp()
    logging.info('Creating new mountpoint at "%s"...', tmp_directory)
    mount_command = subprocess.run(
        # Unexpectedly, using `f"gcsfuse --implicit-dirs --only-dir {specific_bucket_directory} {bucket_name} {tmp_directory}"`
        # resulted in a mountpoint that was *unusably slow* when trying to,
        # e.g., using `.get_thumbnail()`. Thus, here, we are not using
        # `--only-dir.`
        f"gcsfuse --implicit-dirs {bucket_name} {tmp_directory}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    mount_command_stdout = mount_command.stdout.decode().strip()
    if "File system has been successfully mounted" not in mount_command_stdout:
        raise Exception(
            f'Error when running gcsfuse: "{mount_command_stdout}". Exit code was {existing_mountpoint_check.returncode}.'
        )

    return os.path.join(tmp_directory, file_remaining_path)


def run(argv=None, save_main_session=True):
    """Runs the patch inference stitching pipeline."""

    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def _str_to_2_int_tuple_list(s):
        if not s:
            return []
        tuple_list = [tuple([int(z) for z in x.split(";")]) for x in s.split(",")]
        for tup in tuple_list:
            assert len(tup) == 2, "Tuples should have two values: height & width."
        return tuple_list

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stitch_query_images",
        type=_str2bool,
        default=False,
        help="Whether to stitch query images.",
    )
    parser.add_argument(
        "--stitch_query_gen_encoded_images",
        type=_str2bool,
        default=False,
        help="Whether to stitch query generator encoded images.",
    )
    parser.add_argument(
        "--stitch_query_anomaly_images_linear_rgb",
        type=_str2bool,
        default=False,
        help="Whether to stitch query anomaly images linear RGB.",
    )
    parser.add_argument(
        "--stitch_query_anomaly_images_linear_gs",
        type=_str2bool,
        default=False,
        help="Whether to stitch query anomaly images linear GS.",
    )
    parser.add_argument(
        "--stitch_query_mahalanobis_distance_images_linear",
        type=_str2bool,
        default=False,
        help="Whether to stitch query Mahalanobis distance images linear.",
    )
    parser.add_argument(
        "--stitch_query_pixel_anomaly_flag_images",
        type=_str2bool,
        default=False,
        help="Whether to stitch query pixel anomaly flag images.",
    )
    parser.add_argument(
        "--stitch_annotations",
        type=_str2bool,
        default=False,
        help="Whether to stitch annotation images.",
    )
    parser.add_argument(
        "--output_anomaly_polygons_bigquery",
        type=_str2bool,
        default=False,
        help="Whether to write anomaly polygons to BigQuery.",
    )
    parser.add_argument(
        "--output_nuclei_segmentation_cell_coords_bigquery",
        type=_str2bool,
        default=False,
        help="Whether to output segmentation cell coordinates (and write them to BigQuery).",
    )
    parser.add_argument(
        "--output_nuclei_segmentation_cell_coords_json",
        type=_str2bool,
        default=False,
        help="Whether to output segmentation cell coordinates in a JSONL file.",
    )
    parser.add_argument(
        "--output_patch_coordinates",
        type=_str2bool,
        default=False,
        help="Whether to output patch coordinates.",
    )
    parser.add_argument(
        "--output_anomaly_polygon_geoson",
        type=_str2bool,
        default=False,
        help="Whether to write anomalies (filtered based on min_size argument) to a GeoJSON file in the directory specified in the output_gcs_path argument.",
    )
    parser.add_argument(
        "--output_anomaly_polygon_visualizations",
        type=_str2bool,
        default=False,
        help="Whether to write visualizations of anomalies (pre- and post-filtering by on min_size argument) to a PNG file (or two PNG files -- one pre-filtering, one post-filtering -- if min_size is set) in the directory specified in the output_gcs_path argument.",
    )
    parser.add_argument(
        "--output_anomaly_polygon_debugging_pickle",
        type=_str2bool,
        default=False,
        help="Whether to write a pickle containing intermediate output related to anomaly creation to a Pickle file in the directory specified in the output_gcs_path argument.",
    )
    parser.add_argument(
        "--slide_name", type=str, default="", help="Name of slide to stitch."
    )
    parser.add_argument(
        "--png_patch_stitch_gcs_glob_pattern",
        type=str,
        default="",
        help="GCS path of PNG patch images.",
    )
    parser.add_argument(
        "--wsi_stitch_gcs_path", type=str, default="", help="GCS path of WSI."
    )
    parser.add_argument(
        "--wsi_slide_type",
        type=str,
        default="",
        help="Type of WSI. 'skin', 'colon', or 'lung'",
    )
    parser.add_argument(
        "--target_image_width",
        type=int,
        default=500,
        help="The target thumbnail image width.",
    )
    parser.add_argument(
        "--patch_height",
        type=int,
        default=1024,
        help="Number of pixels for patch's height.",
    )
    parser.add_argument(
        "--patch_width",
        type=int,
        default=1024,
        help="Number of pixels for patch's width.",
    )
    parser.add_argument(
        "--patch_depth",
        type=int,
        default=3,
        help="Number of channels for patch's depth.",
    )
    parser.add_argument(
        "--thumbnail_method",
        type=str,
        default="otsu",
        help="Method to apply to thumbnail.",
    )
    parser.add_argument(
        "--polygons_initially_drop_below",
        type=float,
        default=0.90,
        help='In the initial process of creating polygons, drop all polygons below this percentile by area. This serves to quickly remove tiny "dots" of anomalous polygons, speeding up processing time for subsequent steps.',
    )
    parser.add_argument(
        "--rgb2hed_threshold",
        type=float,
        default=-0.41,
        help="Threshold to use for RGB2HED thumbnail method.",
    )
    parser.add_argument(
        "--include_patch_threshold",
        type=float,
        default=0.0,
        help="Threshold using thumbnail when to include a patch.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images to inference the model at once.",
    )
    parser.add_argument(
        "--gan_export_dir",
        type=str,
        default="",
        help="Directory containing exported GAN models.",
    )
    parser.add_argument(
        "--gan_export_name",
        type=str,
        default="",
        help="Name of directory of exported GAN model.",
    )
    parser.add_argument(
        "--generator_architecture",
        type=str,
        default="GANomaly",
        help="Generator architecture type: 'berg' or 'GANomaly'.",
    )
    parser.add_argument(
        "--berg_use_Z_inputs",
        type=_str2bool,
        default=False,
        help="For berg architecture, whether to use Z inputs. Query image inputs are always used.",
    )
    parser.add_argument(
        "--berg_latent_size",
        type=int,
        default=512,
        help="For berg architecture, the latent size of the noise vector.",
    )
    parser.add_argument(
        "--berg_latent_mean",
        type=float,
        default=0.0,
        help="For berg architecture, the latent vector's random normal mean.",
    )
    parser.add_argument(
        "--berg_latent_stddev",
        type=float,
        default=1.0,
        help="For berg architecture, the latent vector's random normal standard deviation.",
    )
    parser.add_argument(
        "--annotation_patch_gcs_filepath",
        type=str,
        default="",
        help="Input file pattern of images.",
    )
    parser.add_argument(
        "--custom_mahalanobis_distance_threshold",
        type=float,
        default=-1.0,
        help="Custom Mahalanobis distance threshold.",
    )
    parser.add_argument(
        "--output_image_sizes",
        type=_str_to_2_int_tuple_list,
        default=[(1024, 1024)],
        help="List of 2-tuples of output image height and width for each n-ary level, starting from leaves.",
    )
    parser.add_argument(
        "--segmentation_export_dir",
        type=str,
        default="",
        help="Directory containing exported segmentation models.",
    )
    parser.add_argument(
        "--segmentation_model_name",
        type=str,
        default="",
        help="Name of segmentation model.",
    )
    parser.add_argument(
        "--segmentation_patch_size",
        type=int,
        default=128,
        help="Size of each patch of image for segmentation model.",
    )
    parser.add_argument(
        "--segmentation_stride",
        type=int,
        default=16,
        help="Number of pixels to skip for each patch of image for segmentation model.",
    )
    parser.add_argument(
        "--segmentation_median_blur_image",
        type=_str2bool,
        default=False,
        help="Whether to median blur images before segmentation.",
    )
    parser.add_argument(
        "--segmentation_median_blur_kernel_size",
        type=int,
        default=9,
        help="The kernel size of median blur for segmentation.",
    )
    parser.add_argument(
        "--segmentation_group_size",
        type=int,
        default=10,
        help="Number of patches to include in a group for segmentation.",
    )
    parser.add_argument(
        "--output_gcs_path",
        type=str,
        required=True,
        help="GCS file path to write outputs to.",
    )
    parser.add_argument(
        "--project", type=str, required=True, help="The GCP project to use for the job."
    )
    parser.add_argument(
        "--bucket", type=str, required=True, help="The GCS bucket to use for staging."
    )
    parser.add_argument(
        "--region", type=str, required=True, help="The GCP region to use for the job."
    )
    parser.add_argument(
        "--autoscaling_algorithm",
        type=str,
        choices=["THROUGHPUT_BASED", "NONE"],
        default="THROUGHPUT_BASED",
        help="Input file pattern of images.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of Dataflow workers."
    )
    parser.add_argument(
        "--machine_type",
        type=str,
        default="n1-standard-1",
        help="The machine type of each Dataflow worker.",
    )
    parser.add_argument(
        "--disk_size_gb",
        type=int,
        default=0,
        help="The disk size, in gigabytes, to use on each worker instance.",
    )
    parser.add_argument(
        "--service_account_email",
        type=str,
        required=True,
        help="User-managed controller service account, using the format my-service-account-name@<project-id>.iam.gserviceaccount.com.",
    )
    parser.add_argument(
        "--use_public_ips",
        type=_str2bool,
        default=False,
        help="Specifies that Dataflow workers must use public IP addresses. If the value is set to false, Dataflow workers use private IP addresses for all communication.",
    )
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        help="The Compute Engine network for launching Compute Engine instances to run your pipeline.",
    )
    parser.add_argument(
        "--subnetwork",
        type=str,
        required=True,
        help="The Compute Engine subnetwork for launching Compute Engine instances to run your pipeline.",
    )
    parser.add_argument("--runner", default="DirectRunner", help="Type of runner.")
    known_args, pipeline_args = parser.parse_known_args(argv)
    logging.info("known_args = {}".format(known_args))
    logging.info("pipeline_args = {}".format(pipeline_args))

    use_png = True if known_args.png_patch_stitch_gcs_glob_pattern else False
    use_wsi = True if known_args.wsi_stitch_gcs_path else False
    if use_wsi == True:
        # As noted elsewhere in this file, Pathlib isn't wholly apposite to use
        # for URLs (specifically, it changes 'gs://...' to 'gs:/...'), but
        # it's still useful in this case:
        uuid = Path(known_args.wsi_stitch_gcs_path).stem
        slide_types = [x for x in [e.name for e in bq.Slide_Type]]
        assert known_args.wsi_slide_type.upper() in slide_types

        wsi_slide_type = bq.Slide_Type[known_args.wsi_slide_type.upper()]

    assert use_png != use_wsi and (use_png or use_wsi)

    job_name = (
        "stitch-"
        + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        + (("-" + uuid) if use_wsi == True else "")
    )

    # We use the save_main_session option because one or more DoFn"s in this
    # workflow rely on global context (e.g., a module imported at module level).
    options = {
        "job_name": job_name,
        "experiments": [
            "shuffle_mode=service",
            # In order to use a custom Docker container (with the
            # "sdk_container_image" option below), DataFlow Runner v2 is
            # required, as documented at
            # https://cloud.google.com/dataflow/docs/guides/using-custom-containers#usage
            "use_runner_v2",
        ],
        "project": known_args.project,
        "staging_location": os.path.join(known_args.bucket, "tmp", "staging"),
        "temp_location": os.path.join(known_args.bucket, "tmp"),
        "region": known_args.region,
        "autoscaling_algorithm": known_args.autoscaling_algorithm,
        "num_workers": known_args.num_workers,
        "machine_type": known_args.machine_type,
        "disk_size_gb": known_args.disk_size_gb,
        "service_account_email": known_args.service_account_email,
        "use_public_ips": known_args.use_public_ips,
        "network": known_args.network,
        "subnetwork": known_args.subnetwork,
        # See the Dockerfile in this directory and the documentation at
        # https://cloud.google.com/dataflow/docs/guides/using-custom-containers#python
        # for more information on this custom container image:
        "sdk_container_image": "us-central1-docker.pkg.dev/ml-mps-aif-atsranom01-p-c347/phi-main-us-central1-p/ganomaly-beam-image-stitch:0.0.1",
    }
    pipeline_options = beam.options.pipeline_options.PipelineOptions(
        flags=pipeline_args, **options
    )
    pipeline_options.view_as(
        beam.options.pipeline_options.SetupOptions
    ).save_main_session = save_main_session

    image_stitch_types_set = set()
    if known_args.stitch_query_images:
        image_stitch_types_set.add("query_images")
    if known_args.stitch_query_gen_encoded_images:
        image_stitch_types_set.add("query_gen_encoded_images")
    if known_args.stitch_query_anomaly_images_linear_rgb:
        image_stitch_types_set.add("query_anomaly_images_linear_rgb")
    if known_args.stitch_query_anomaly_images_linear_gs:
        image_stitch_types_set.add("query_anomaly_images_linear_gs")
    if known_args.stitch_query_mahalanobis_distance_images_linear:
        image_stitch_types_set.add("query_mahalanobis_distance_images_linear")
    if known_args.stitch_query_pixel_anomaly_flag_images:
        image_stitch_types_set.add("query_pixel_anomaly_flag_images")
    if known_args.stitch_annotations:
        image_stitch_types_set.add("annotations")

    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(known_args.runner, options=pipeline_options) as p:
        if use_wsi:
            wsi_local_path = resolve_gcs_uri_to_local_uri(
                known_args.wsi_stitch_gcs_path
            )
            wsi = OpenSlide(wsi_local_path)

            wsi_microns_per_pixel_x = float(wsi.properties.get("openslide.mpp-x"))
            wsi_microns_per_pixel_y = float(wsi.properties.get("openslide.mpp-y"))

            if wsi_microns_per_pixel_x != wsi_microns_per_pixel_y:
                logging.warning(
                    "The Whole Slide Image has a different value for Microns Per Pixel in the x-dimension (%s) than in the y-dimension (%s). The pipeline is set up just to use a single value, and so will be using just the x-dimension value.",
                    wsi_microns_per_pixel_x,
                    wsi_microns_per_pixel_y,
                )

            logging.info("Inserting slide if it does not already exist...")
            bq.slide_to_bigquery(
                slide_uuid=uuid,
                slide_type=wsi_slide_type,
                project_id=known_args.project,
                slide_microns_per_pixel=wsi_microns_per_pixel_x,
                slide_dimensions=wsi.dimensions,
            )

            # TODO: Get this to work with non-WSI pipeline
            nary_tree_depth = nary_depth(
                wsi.dimensions, known_args.patch_height, known_args.patch_width
            )

            assert len(known_args.output_image_sizes) >= nary_tree_depth

            patches_metadata = pre_inference_wsi.wsi_pre_inference(
                wsi=wsi,
                target_image_width=known_args.target_image_width,
                patch_height=known_args.patch_height,
                patch_width=known_args.patch_width,
                thumbnail_method=known_args.thumbnail_method,
                rgb2hed_threshold=known_args.rgb2hed_threshold,
                include_patch_threshold=(known_args.include_patch_threshold),
                batch_size=known_args.batch_size,
            )

            pre_inf = p | "{} WSI Pre-inference".format(
                known_args.slide_name
            ) >> beam.Create(patches_metadata)
        # TODO: Work this out with resolve_gcs_uri_to_local_uri()
        # TODO: Incorporate resolve_gcs_uri_to_local_uri() elsewhere in the codebase.
        else:
            logging.warning(
                "The PNG Patch pre-inference approach is deprecated, and has not been updated to parity with the Whole-Slide Image (WSI) approach."
            )

            pre_inf = p | "{} PNG Patch Pre-inference".format(
                known_args.slide_name
            ) >> beam.Create(
                pre_inference_png.png_patch_pre_inference(
                    png_patch_stitch_gcs_glob_pattern=(
                        known_args.png_patch_stitch_gcs_glob_pattern
                    ),
                    patch_height=known_args.patch_height,
                    patch_width=known_args.patch_width,
                    batch_size=known_args.batch_size,
                )
            )

        if image_stitch_types_set or (
            known_args.output_nuclei_segmentation_cell_coords_bigquery
            or known_args.output_nuclei_segmentation_cell_coords_json
        ):
            batch = pre_inf | "Group Batch Index" >> beam.GroupByKey()

            inference_do = batch | "Inference" >> beam.ParDo(
                inference.InferenceDoFn(
                    wsi_stitch_gcs_path=known_args.wsi_stitch_gcs_path,
                    patch_height=known_args.patch_height,
                    patch_width=known_args.patch_width,
                    patch_depth=known_args.patch_depth,
                    gan_export_dir=known_args.gan_export_dir,
                    gan_export_name=known_args.gan_export_name,
                    generator_architecture=known_args.generator_architecture,
                    berg_use_Z_inputs=known_args.berg_use_Z_inputs,
                    berg_latent_size=known_args.berg_latent_size,
                    berg_latent_mean=known_args.berg_latent_mean,
                    berg_latent_stddev=known_args.berg_latent_stddev,
                    image_stitch_types_set=image_stitch_types_set,
                    annotation_patch_gcs_filepath=known_args.annotation_patch_gcs_filepath,
                    custom_mahalanobis_distance_threshold=(
                        known_args.custom_mahalanobis_distance_threshold
                    ),
                    output_anomaly_polygons_bigquery=known_args.output_anomaly_polygons_bigquery,
                    output_nuclei_segmentation_cell_coords=known_args.output_nuclei_segmentation_cell_coords_bigquery
                    or known_args.output_nuclei_segmentation_cell_coords_json,
                    segmentation_export_dir=(known_args.segmentation_export_dir),
                    segmentation_model_name=(known_args.segmentation_model_name),
                    segmentation_patch_size=(known_args.segmentation_patch_size),
                    segmentation_stride=known_args.segmentation_stride,
                    segmentation_median_blur_image=(
                        known_args.segmentation_median_blur_image
                    ),
                    segmentation_median_blur_kernel_size=(
                        known_args.segmentation_median_blur_kernel_size
                    ),
                    segmentation_group_size=known_args.segmentation_group_size,
                    microns_per_pixel=wsi_microns_per_pixel_x,
                )
            )

        # Images.
        for stitch_type in image_stitch_types_set:
            leaf_combine = inference_do | "Leaf Combine_image_{}".format(
                stitch_type
            ) >> beam.ParDo(images.LeafCombineDoFn(stitch_type=stitch_type))
            leaf_group = (
                leaf_combine | "Leaf Group_{}".format(stitch_type) >> beam.GroupByKey()
            )
            branch_combine = leaf_group | "Branch-leaf Combine_image_{}".format(
                stitch_type
            ) >> beam.ParDo(
                images.BranchCombineDoFn(
                    patch_height=known_args.output_image_sizes[0][0],
                    patch_width=known_args.output_image_sizes[0][1],
                )
            )
            for i in range(nary_tree_depth - 1):
                branch_group = (
                    branch_combine
                    | "Branch-branch Group image_{}_{}".format(i, stitch_type)
                    >> beam.GroupByKey()
                )
                branch_combine = (
                    branch_group
                    | "Branch-branch Combine image_{}_{}".format(i, stitch_type)
                    >> beam.ParDo(
                        images.BranchCombineDoFn(
                            patch_height=(known_args.output_image_sizes[i + 1][0]),
                            patch_width=known_args.output_image_sizes[i + 1][1],
                        )
                    )
                )

            write_images = branch_combine | "Write images_image_{}".format(
                stitch_type
            ) >> beam.ParDo(
                images.WriteImageDoFn(
                    output_filename=os.path.join(
                        known_args.output_gcs_path, stitch_type + ".png"
                    )
                )
            )

            # Create polygons from the anomaly_flag_images stitched image:
            if stitch_type == "query_pixel_anomaly_flag_images" and use_wsi == True:
                # TODO: Work this out with when use_wsi != True

                logging.info(
                    "Removing any existing anomaly rows pertaining to the slide in BigQuery..."
                )
                bq.delete_slide_rows(
                    project_id=known_args.project,
                    table_name="anomaly_polygons",
                    slide_uuid=uuid,
                    dataset_name="phi_main_us_p",
                )

                anomaly_polygons = branch_combine | "Create anomaly polygons from stitched query_pixel_anomaly_flag_images" >> beam.Map(
                    polygons.create_anomaly_polygons,
                    anomaly_resize_factor=calculate_anomaly_resize_factor(
                        wsi_dimensions=wsi.dimensions,
                        stitched_output_image_size=known_args.output_image_sizes[
                            nary_tree_depth
                        ],
                        patches_width=known_args.patch_width,
                        patches_height=known_args.patch_height,
                        nary_depth=nary_tree_depth,
                    ),
                    output_filename=os.path.join(
                        known_args.output_gcs_path, "anomaly_polygons"
                    ),
                    min_size=(2000, 2000),
                    slide_microns_per_pixel=wsi_microns_per_pixel_x,
                    polygons_initially_drop_below=known_args.polygons_initially_drop_below,
                    output_anomaly_polygon_geoson=known_args.output_anomaly_polygon_geoson,
                    output_anomaly_polygon_visualizations=known_args.output_anomaly_polygon_visualizations,
                    output_anomaly_polygon_debugging_pickle=known_args.output_anomaly_polygon_debugging_pickle,
                )

                if known_args.output_anomaly_polygons_bigquery:
                    anomaly_polygons = (
                        anomaly_polygons
                        | "Process anomaly polygons and send them to BigQuery"
                        >> beam.Map(
                            polygons.upload_anomaly_polygons,
                            slide_uuid=uuid,
                            project_id=known_args.project,
                            slide_microns_per_pixel=wsi_microns_per_pixel_x,
                        )
                    )

        # Segmentation coordinates.

        if known_args.output_nuclei_segmentation_cell_coords_bigquery:
            logging.info(
                "Removing any existing nuclei rows pertaining to the slide in BigQuery..."
            )
            bq.delete_slide_rows(
                project_id=known_args.project,
                table_name="nuclear_polygons",
                slide_uuid=uuid,
                dataset_name="phi_main_us_p",
            )

            nuclei_polygons = (
                inference_do
                | "Process cell polygons and their centroids and send them to BigQuery"
                >> beam.Map(
                    polygons.upload_nuclei_polygons,
                    slide_uuid=uuid,
                    project_id=known_args.project,
                    microns_per_pixel=wsi_microns_per_pixel_x,
                )
            )

        if known_args.output_nuclei_segmentation_cell_coords_json:
            nuclei_json_file_path = os.path.join(
                known_args.output_gcs_path,
                "nuclei.jsonl",
            )
            logging.info(
                'Writing nuclei centroids to JSON file(s) in "%s"...',
                nuclei_json_file_path,
            )
            cell_segmentation_dict_to_json_map = (
                inference_do
                | "Nuclei centroids GeoDataFrames to JSON"
                >> beam.FlatMap(
                    lambda x: segmentation.segmentation_dict_to_json(element_dict=x)
                )
            )

            cell_write_segmentation_coords = (
                cell_segmentation_dict_to_json_map
                | "Write nuclei centroids to JSONL file"
                >> beam.io.Write(
                    beam.io.WriteToText(
                        file_path_prefix=nuclei_json_file_path,
                        num_shards=1,
                    )
                )
            )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
