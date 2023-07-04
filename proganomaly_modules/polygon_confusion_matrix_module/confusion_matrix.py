# Copyright 2020 Google lnc. All Rights Reserved.
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


import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from proganomaly_modules.polygon_confusion_matrix_module import polygon_confusion_matrices, custom_configs


def polygon_confusion_matrix_main():
    print("P-CEAD Calculating Inference Statistics Results")
    arguments = custom_configs.custom_config()

    args_required = arguments["required"]
    args_optional = arguments["optional"]
    args_output = arguments["output"]

    confusion_matrix = polygon_confusion_matrices.Polygon_Confusion_Matrices(
        wsi_gcs_path = args_required["wsi_gcs_path"],
        wsi_level = args_required["wsi_level"],
        anomaly_polygon_geojson_gcs_path = args_required["anomaly_polygon_geojson_gcs_path"],
        annotation_polygon_geojson_gcs_path = args_required["annotation_polygon_geojson_gcs_path"]
    )

    if args_output["output_polygon_thumbnail_plot_gcs"]:
        print("Start Creating and Uploading Polygon-Thumbnail Plot to GCS Bucket")
        confusion_matrix.polygons_thumbnail_plot_to_gcs(
            figsize_factor = args_optional["figsize_factor"],
            plot_gcs_path = args_optional["polygon_thumbnail_plot_gcs_path"]
        )
    
    if args_output["output_inference_stats_json_gcs"]:
        print("Start Generating and Uploading Anomaly Detection Inference Statistics Measurements Json File to GCS Bucket")
        confusion_matrix.inference_stats_to_gcs(
            stats_json_gcs_path = args_required["stats_json_gcs_path"]
        )