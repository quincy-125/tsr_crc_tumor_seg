# Pre-Training Whole Slide Image (WSI) Pre-Process 

## License

Copyright 2020 Google lnc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.
You may obtain a copy of the License at [Apache License Page](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

## Required Package Installation
- Required package installation commands
######
        pip3 install --upgrade pip
        pip3 install tensorflow
        pip3 install numpy
        pip3 install openslide
        pip3 install scikit-image
        pip3 install matplotlib
        pip3 install sklearn
        pip3 install shapely

## Pre-Training Whole Slide Image (WSI) Pre-Process Config Introduction

There are four main groups of configs: required, optional, and output.

The required config contains all parameters required to extract patches with tissue regions only from a WSI, applying the principal component analysis (PCA) to reduce data redundancy, and create TFRecords with selected patches out from the PCA-based filtering process.

The optional config contains all parameters regarding output the intermediate results.

The output config contains the all the boolean type parameters regarding whether or not return the results files.

## Pre-Training Whole Slide Image (WSI) Pre-Process Parameters

required config

- the ID of user's GCS bucket
######
        "bucket_name": "",
        
- GCS path of WSI file
######
        "wsi_gcs_path" : "gs://",
        
- the approximate width of resultant thumbnail image
######
        "target_image_width" : 500,
    
- threshold with the percentage of tissue area included within a single patch, which used to determine whether a patch will be extracted from WSI
######
        "threshold_area_percent" : 0.5,

- initial image level of each image patches, default be 0 indicates patch resolution size be (1024,1024)
######
        "patch_level" : 0,

- the depth of each image patches, represent the number of channels of each image patches, default be 3, which represents 3 RGB-channel image patches
######
        "image_patch_depth" : 3,

- the smallest patch size required for progressive GAN training purpose, default be 4
######
        "min_patch_size" : 4,

- original patch size extracted from WSI, default be 1024
######
        "max_patch_size" : 1024,

- whether to include the fully-connected layer at the top of the network, default be False
######
        "include_top" : False,

- weights of the restnet50 model pre-trained on one of the following three general options, which are None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded
######
        "weights" : "imagenet",

- optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value. In our case, default be (1024, 1024, 3)
######
        "input_shape" : (1024, 1024, 3),

- whether to re-train the restnet50 model, default be False
######
        "is_trainable" : False,

- user defined layer to customize the restnet50 model, which used to return the output feature vectors from the user defined layer. In our case, default be 'conv4_block1_0_conv', which used to create a customized resnet50 model based on original resnet50 model ended after the 3rd residual block
######
        "layer_name" : "conv4_block1_0_conv",

- number of components to keep. If n_components is not set all components are kept. Default value be None
######
        "n_components" : 10,

- the percentage of the total number of patches used to determine the number of patches will be selected from PCA for the anomaly detection model training purpose. Default value be 0.8
######
        "pca_filter_percent" : 1.0,

- the index of the principal component ndarray from PCA used to select image patches for the anomaly detection model training purpose. Default value be 0, which indicates the 1st principal component
######
        "pca_filter_base_component" : 0,

- whether to save the selected patches filtered by the principal component analysis (PCA), or all patches with tissue only regions extracted from WSI to a local file path. Default be True, which intends to save PCA filtered patches
######
        "is_filtered_patches" : False,

- path to where the tfrecord files will be stored in GCS bucket
######
        "tfrecord_gcs_path" : "gs://"

optional_config 

- all patches with tissue only regions extracted from WSI stored in a GCS bucket
######
        "all_patch_gcs_path" : "gs://",

- selected patches filtered by the principal component analysis (PCA) stored in a GCS bucket
######
        "pca_patch_gcs_path" : "gs://",

- index to determine which principla component related ndarray features will be used to plot on the x-axis
######
        "x_pca_index" : 0,

- index to determine which principla component related ndarray features will be used to plot on the y-axis
######
        "y_pca_index" : 1,

- index to determine which principla component related ndarray features will be used to plot on the z-axis
######
        "z_pca_index" : 2,
        
- path to where the plots (listed in the description of the function) will be stored in GCS bucket
######
        "pca_plot_gcs_path" : "gs://"

output_config

- whether or not save tfrecords in a GCS bucket
######
        "output_tfrecords_gcs" : True,

- whether or not save all patches with tissue only regions extracted from the WSI in a GCS bucket
######
        "output_all_patches_gcs" : False,

- whether or not save selected patches filtered by the principal component analysis (PCA) in a GCS bucket
######
        "output_pca_selected_patches_gcs" : False,

- whether or not save PCA analysis results plots in a GCS bucket
######
        "output_selected_pca_results_plots_gcs" : False

config 
- entire inference pipeline config
######
        "required": required_config,
        "optional": optional_config,
        "output": output_config
