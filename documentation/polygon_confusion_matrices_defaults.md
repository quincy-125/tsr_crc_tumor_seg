# Polygon Confusion Matrices 

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
        pip3 install geopandans
        pip3 install openslide
        pip3 install geojson
        pip3 install matplotlib
        pip3 install shapely

## Polygon Confusion Matrices  Config Introduction

There are four main groups of configs: required, optional, and output.

The required config contains all parameters required to extract patches with tissue regions only from a WSI, applying the principal component analysis (PCA) to reduce data redundancy, and create TFRecords with selected patches out from the PCA-based filtering process.

The optional config contains all parameters regarding output the intermediate results.

The output config contains the all the boolean type parameters regarding whether or not return the results files.

## Polygon Confusion Matrices  Parameters

required config
       
- GCS path of WSI file
######
        "wsi_gcs_path" : "gs://",
        
- level dimension index for WSI, default be 0
######
        "wsi_level": 0,
    
- path to the anomaly polygon geojson file stored in GCS bucket
######
        "anomaly_polygon_geojson_gcs_path":  "gs://",

- path to the annotated polygon geojson file stored in GCS bucket
######
        "annotation_polygon_geojson_gcs_path" : "gs://",

- path to the anomaly detection inference statistics measurement json file in a GCS bucket
######
        "stats_json_gcs_path": "gs://"

optional_config 

- factor to magnify the entire plot by at the end of the plotting process (e.g., to make details easier to see in certain text editors). Defaults to 20
######
        "figsize_factor" : 20,

- path to the plot of a collection of anomaly and annottaion polygons on top of a thumbnail of a WSI in GCS bucket
######
        "polygon_thumbnail_plot_gcs_path" : "gs://"

output_config

- whether or not upload the plot a collection of anomaly and annotation polygons on top of a thumbnail of a WSI to GCS bucket
######
        "output_polygon_thumbnail_plot_gcs" : True,

- whether or not upload the anomaly detection inference statistics measurement json file to the GCS bucket
######
        "output_inference_stats_json_gcs" : True

config 
- entire inference pipeline config
######
        "required": required_config,
        "optional": optional_config,
        "output": output_config
