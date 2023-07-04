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

from proganomaly_modules.pre_train_wsi_preprocess_module import wsi_pca_preprocess, custom_configs


def wsi_preprocess_main():
    print("P-CEAD WSI Pre-Processing")
    arguments = custom_configs.custom_config()

    args_required = arguments["required"]
    args_optional = arguments["optional"]
    args_output = arguments["output"]

    wsi_preprocess = wsi_pca_preprocess.PCEAD_WSI_TRAIN_PREP(
        bucket_name = args_required["bucket_name"],
        wsi_gcs_path = args_required["wsi_gcs_path"],
        target_image_width = args_required["target_image_width"],
        threshold_area_percent = args_required["threshold_area_percent"],
        patch_level = args_required["patch_level"],
        image_patch_depth = args_required["image_patch_depth"],
        min_patch_size = args_required["min_patch_size"],
        max_patch_size = args_required["max_patch_size"],
        include_top = args_required["include_top"],
        weights = args_required["weights"],
        input_shape = args_required["input_shape"],
        is_trainable = args_required["is_trainable"],
        layer_name = args_required["layer_name"],
        n_components = args_required["n_components"],
        pca_filter_percent = args_required["pca_filter_percent"],
        pca_filter_base_component = args_required["pca_filter_base_component"],
        is_filtered_patches = args_required["is_filtered_patches"],
        tfrecord_gcs_path = args_required["tfrecord_gcs_path"]
    )

    if args_output["output_tfrecords_gcs"]:
        print("Start Creating TFrecords")
        wsi_preprocess.patches_to_tfrecord_gcs()
    
    if args_output["output_all_patches_gcs"]:
        print("Start Uploading all Extracted Image Patches to GCS Bucket")
        wsi_preprocess.patches_local_to_gcp(
            patch_gcs_path = args_optional["all_patch_gcs_path"]
        )
    
    if args_output["output_pca_selected_patches_gcs"]:
        print("Start Uploading PCA Filtered Image Patches to GCS Bucket")
        wsi_preprocess.patches_local_to_gcp(
            patch_gcs_path = args_optional["pca_patch_gcs_path"]
        )
    
    if args_output["output_selected_pca_results_plots_gcs"]:
        print("Start Uploading PCA Analysis Results Plots to GCS Bucket")
        wsi_preprocess.pca_results_plot_gcs(
            x_pca_index = args_optional["x_pca_index"],
            y_pca_index = args_optional["y_pca_index"],
            z_pca_index = args_optional["z_pca_index"],
            pca_plot_gcs_path = args_optional["pca_plot_gcs_path"]
        )