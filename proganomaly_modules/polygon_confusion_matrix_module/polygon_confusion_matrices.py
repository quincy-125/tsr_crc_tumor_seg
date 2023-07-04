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


from apache_beam.io.gcp import gcsio
import tensorflow as tf
import geopandas as gpd
import logging
from matplotlib import pyplot as plt
from shapely.geometry import Point
import openslide
import geojson
import json
import os


class Polygon_Confusion_Matrices():
    """
    A class of functionalities provided to evaluate the trained anomaly detection model performance by comparing the predicted anomlous polygons with the ground truth annotated anomalous polygons provided by pathologists.

    Args:
        Check input args for wsi_patch_extraction class with details inside the init function.
    
    Functionalities:
        - Load Predicted and Annotated Anomalous Polygon GeoJson Files
        - Evaluate Model Performance
    
    Methods:
        ************************************************************************************************************************
        Note: 
            - If any method is optional (i.e., depent on user's preference to determine wheher to be executed), the notation
            "[optional]" will be added after the method name.
            - If no notations aded after the method name, which indicates that method is requried.
        ************************************************************************************************************************
        - General
            - __init__: used to initialize required args for wsi_patch_extraction class.
        
        - Load Predicted and Annotated Anomalous Polygon GeoJson Files
            - Functionality Specific Methods:
                - get_anomaly_polygons: get the predicted anomaly polygons from anomaly polygon geojson file.
                - get_annotation_polygons: get the annotation polygons from annotation polygon geojson file.
                - plot_slide_and_polygons [optional]: plot a collection of anomaly and annotation polygons on top of a thumbnail of a WSI.
        
        - Evaluate Model Performance
            - Functionality Specific Methods
                - anomaly_detection_inference_stat: measure the trained anomaly detection model performance by computing the intersection of union (IOU), confusion matrix (i.e., true/false positive/negative), sensitivity, specificity, precision, accuracy, and f1.
            - Utility Methods
                - calculate_iou: calculate the intersection over union (IOU) by comparing the predicted and annotated anomalous multi-polygons.
                - calculate_confusion_matrix: calculate the confusion matrix (i.e., true/false positive/negative) by comparing the predicted and annotated anomalous multi-polygons.
    
    Returns:
        Check returnning outputs inside each of methods' functions.
    """
    def __init__(
        self,
        wsi_gcs_path, 
        wsi_level,
        anomaly_polygon_geojson_gcs_path,
        annotation_polygon_geojson_gcs_path
    ):
        """
        Used to initialize required args for polygon_confusion_matrices class.

        Args:
            wsi_gcs_path: str, GCS path of WSI file.
            wsi_level: int, level dimension index for WSI, default be 0.
            anomaly_polygon_geojson_gcs_path: str, path to the anomaly polygon geojson file stored in GCS bucket.
            annotation_polygon_geojson_gcs_path: str, path to the annotated polygon geojson file stored in GCS bucket.

        Returns:
            None: Return N/A and only initialize the args of polygon_confusion_matrices class.
        """
        self.wsi_gcs_path = wsi_gcs_path
        self.wsi_level = wsi_level
        self.anomaly_polygon_geojson_gcs_path = anomaly_polygon_geojson_gcs_path
        self.annotation_polygon_geojson_gcs_path = annotation_polygon_geojson_gcs_path
    
    def get_anomaly_polygons(self):
        """
        Get the predicted anomaly polygons from anomaly polygon geojson file.

        Args:
            anomaly_polygon_geojson_gcs_path: str, path to the anomaly polygon geojson file stored in GCS bucket.
        
        Returns:
            anomaly_polygons_geo_df: geopandas.GeoDatFrame, a collection of anomaly polygons output from anomaly detection model inference pipeline(for example, representing predicted anomalous regions on the slide image).
            anomaly_multi_polygons: shapely.MultiPolygon object, a collection of anomaly polygons output from anomaly detection model inference pipeline(for example, representing predicted anomalous regions on the slide image).
        """
        if tf.io.gfile.exists(self.anomaly_polygon_geojson_gcs_path):
            with tf.io.gfile.GFile(self.anomaly_polygon_geojson_gcs_path) as f:
                anomaly_polygons_geojson = geojson.load(f)
            anomaly_polygons_geo_df = gpd.GeoDataFrame.from_features(anomaly_polygons_geojson)
            anomaly_multi_polygons = anomaly_polygons_geo_df.unary_union

            return anomaly_polygons_geo_df, anomaly_multi_polygons
        else:
            logging.warning("anomaly polygons geojson file was not found on the GCS bucket!")
    
    def get_annotation_polygons(self):
        """
        Get the annotation polygons from annotation polygon geojson file.

        Args:
            annotation_polygon_geojson_gcs_path: str, path to the annotated polygon geojson file stored in GCS bucket.
        
        Returns:
            annotation_polygons_geo_df: geopandas.GeoDatFrame, a collection of manual annotation polygons exported from QuPath by pathologists(for example, representing groundtruth anomalous regions on the slide image).
            annotation_multi_polygons: shapely.MultiPolygon object, a collection of manual annotation polygons exported from QuPath by pathologists(for example, representing groundtruth anomalous regions on the slide image).
        """
        with tf.io.gfile.GFile(self.annotation_polygon_geojson_gcs_path) as f:
            annotation_polygons_geojson = geojson.load(f)
        qupath_annotation_polygons_geo_df = gpd.GeoDataFrame.from_features(annotation_polygons_geojson)
        # x- and y- axis of exported annotation polygon geojson objects have been flipped by QuPath
        # flip the axis of annotation polygons based on the origin point (i.e., (0, 0))
        annotation_polygons_geo_sr = qupath_annotation_polygons_geo_df.scale(
            yfact=-1, 
            origin=(0,0)
        )
        annotation_polygons_geo_df = gpd.GeoDataFrame({"geometry":annotation_polygons_geo_sr})
        annotation_multi_polygons = annotation_polygons_geo_df.unary_union

        return annotation_polygons_geo_df, annotation_multi_polygons
    
    def get_dimensions_of_polygon(
        polygon,
        sort
    ):
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
            polygon_dimensions = (0, 0)

        rectangle = polygon.minimum_rotated_rectangle
        x, y = rectangle.exterior.coords.xy
        # Return a tuple of distances:
        distance1 = Point(x[0], y[0]).distance(Point(x[1], y[1]))
        distance2 = Point(x[1], y[1]).distance(Point(x[2], y[2]))
        if sort is True:
            polygon_dimensions = (min(distance1, distance2), max(distance1, distance2))
        
        polygon_dimensions = (distance1, distance2)

        return polygon_dimensions
    
    def get_wsi_data(self):
        """
        A general utility function to return the basic information from a WSI.

        Args:
            None: no functional specific input arguments required to execute this function.
            
        Returns:
            wsi_data: python dictionary, includes uuid, wsi, x_axis_slide_microns_per_pixel, y_axis_slide_microns_per_pixel, thumbnail, and thumbnail_dim values. The format of wsi_data is the following:
                {
                    "uuid": wsi_uuid [str],
                    "wsi": wsi [openslide WSI object],
                    "x_axis_slide_microns_per_pixel": [float],
                    "y_axis_slide_microns_per_pixel": [float],
                    "thumbnail": thumbnail [matplotlib.image.AxesImage], 
                    "thumbnail_dim": thumbnail_dimensions [tuple with float numbers]
                }
        """
        wsi_uuid = self.wsi_gcs_path.split(".")[0].split("/")[-1]

        gcs = gcsio.GcsIO()
        
        local_file = "slide_file." + self.wsi_gcs_path.split(".")[-1]
        with open(local_file, "wb") as f:
            f.write(gcs.open(self.wsi_gcs_path).read())
        wsi = openslide.OpenSlide(filename=local_file)

        thumbnail = wsi.get_thumbnail(wsi.level_dimensions[self.wsi_level])
        thumbnail_dimensions = thumbnail.size

        wsi_data = {
            "uuid": wsi_uuid,
            "wsi": wsi,
            "x_axis_slide_microns_per_pixel": float(wsi.properties["openslide.mpp-x"]),
            "y_axis_slide_microns_per_pixel": float(wsi.properties["openslide.mpp-y"]),
            "thumbnail": thumbnail, 
            "thumbnail_dim": thumbnail_dimensions
        }

        return wsi_data
    
    def file_local_to_gcs(self, file_local_path, file_gcs_path):
        """
        A general utily function to upload any file stored locally to a user defined GCS path.

        Args:
            file_local_path: str, local path to the file which needed to be uploaded to a GCS bucket.
            file_gcs_path: str, user defined GCS path to store the file which will be uploaded from a local path.

        Returns:
            None: Nothing is returned, only upload the local file to a user defined GCS path.
        """
        assert {
            os.path.exists(file_local_path)
        }, "Failed to locate the local file"

        if not tf.io.gfile.exists(file_gcs_path):
            tf.io.gfile.mkdir(file_gcs_path)
        
        assert {
            tf.io.gfile.exists(file_gcs_path)
        }, "Failed to locate the file in GCS bucket"

        file_transfer_cmd = "gsutil -m cp " + file_local_path + " " + file_gcs_path
        os.system(file_transfer_cmd)
    
    def plot_slide_and_polygons(
        self,
        figsize_factor
    ):
        """
        Plot a collection of anomaly and annotation polygons on top of a thumbnail of a WSI.

        Args:
            figsize_factor: int, Factor to magnify the entire plot by at the end of the plotting process (e.g., to make details easier
            to see in certain text editors). Defaults to 20.

        Returns:
            polygon_plot: matplotlib.image.AxesImage, the plot of a collection of anomaly and annotation polygons on top of a thumbnail of a WSI.
        """
        wsi_data = self.get_wsi_data()
        wsi = wsi_data["wsi"]
        
        if tf.io.gfile.exists(self.anomaly_polygon_geojson_gcs_path):
            anomaly_polygons_geo_df = self.get_anomaly_polygons()[0]
            annotation_polygons_geo_df = self.get_annotation_polygons()[0]

            scaled_anomalies = anomaly_polygons_geo_df.scale(
                xfact=1/wsi.level_downsamples[self.wsi_level], 
                yfact=1/wsi.level_downsamples[self.wsi_level], 
                origin=(0, 0)
            )
            scaled_annotations = annotation_polygons_geo_df.scale(
                xfact=1/wsi.level_downsamples[self.wsi_level], 
                yfact=1/wsi.level_downsamples[self.wsi_level], 
                origin=(0, 0)
            )

            thumbnail = wsi.get_thumbnail(wsi.level_dimensions[self.wsi_level])
            thumbnail_dimensions = thumbnail.size
            
            logging.info("Plotting started")
            thumbnail_dimension_ratio = min(thumbnail_dimensions) / max(
                thumbnail_dimensions
            )
            figsize = tuple(
                [x * figsize_factor for x in (1, thumbnail_dimension_ratio)]
            )

            _, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))

            gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(scaled_anomalies.unary_union)).plot(ax=ax, cmap="Reds", alpha=0.6, hatch="//"
            )
            gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(scaled_annotations.unary_union)).plot(ax=ax, cmap="Blues", alpha=0.6, hatch="\\\\"
            )

            polygon_plot = ax.imshow(
                thumbnail,
                aspect="auto",
                interpolation="bilinear",
                extent=(
                    0,  # Left
                    thumbnail_dimensions[0],  # Right
                    -thumbnail_dimensions[1],  # Bottom
                    0,  # Top
                ),
                alpha=0.75,
                cmap="gray",
            )
        else:
            logging.warning("anomaly polygons geojson file did not found on the GCS bucket!")
            annotation_polygons_geo_df = self.get_annotation_polygons()[0]
            scaled_annotations = annotation_polygons_geo_df.scale(
                xfact=1/wsi.level_downsamples[self.wsi_level], 
                yfact=1/wsi.level_downsamples[self.wsi_level], 
                origin=(0, 0)
            )

            thumbnail = wsi.get_thumbnail(wsi.level_dimensions[self.wsi_level])
            thumbnail_dimensions = thumbnail.size
            
            logging.info("Plotting started")
            thumbnail_dimension_ratio = min(thumbnail_dimensions) / max(
                thumbnail_dimensions
            )
            figsize = tuple(
                [x * figsize_factor for x in (1, thumbnail_dimension_ratio)]
            )

            _, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))

            gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(scaled_annotations.unary_union)).plot(ax=ax, cmap="Blues", alpha=0.6, hatch="\\\\"
            )

            polygon_plot = ax.imshow(
                thumbnail,
                aspect="auto",
                interpolation="bilinear",
                extent=(
                    0,  # Left
                    thumbnail_dimensions[0],  # Right
                    -thumbnail_dimensions[1],  # Bottom
                    0,  # Top
                ),
                alpha=0.75,
                cmap="gray",
            )

        return polygon_plot
    
    def polygons_thumbnail_plot_to_gcs(
        self,
        figsize_factor,
        plot_gcs_path
    ):
        """
        Upload the plot a collection of anomaly and annotation polygons on top of a thumbnail of a WSI to GCS bucket.

        Args:
            figsize_factor: int, Factor to magnify the entire plot by at the end of the plotting process (e.g., to make details easier
            to see in certain text editors). Defaults to 20.
            plot_gcs_path: str, path to the plot of a collection of anomaly and annottaion polygons on top of a thumbnail of a WSI in GCS bucket.
        
        Returns:
            None: Nothing is returned, but a plot is created and displayed.
        """
        wsi_data = self.get_wsi_data()
        polygon_plot = self.plot_slide_and_polygons(figsize_factor=figsize_factor)

        plot_name = wsi_data["uuid"] + "_slide_polygon_alignment_plot.png"
        polygon_plot.get_figure().savefig(plot_name)

        self.file_local_to_gcs(
            file_local_path = plot_name,
            file_gcs_path = plot_gcs_path,
        )
        if tf.io.gfile.exists(plot_gcs_path):
            if len(tf.io.gfile.listdir(plot_gcs_path)) != 0:
                os.remove(plot_name)

    def calculate_iou(self):
        """
        Calculate the intersection over union (IOU) by comparing the predicted and annotated anomalous multi-polygons.

        Args:
            None: no functional specific input arguments required to execute this function.

        Returns:
            iou: float, area of the intersection over union of the predicted and annotated anomalous multi-polygons.
        """
        annotation_multi_polygons = self.get_annotation_polygons()[1]
        if tf.io.gfile.exists(self.anomaly_polygon_geojson_gcs_path):
            anomaly_multi_polygons = self.get_anomaly_polygons()[1]
            intersect_polygons_area = anomaly_multi_polygons.intersection(annotation_multi_polygons).area 
            union_polygons_area = anomaly_multi_polygons.union(annotation_multi_polygons).area

            iou = intersect_polygons_area / union_polygons_area
        else:
            logging.warning("anomaly polygons geojson file did not found on the GCS bucket!")
            # when anomaly_polygons is empty, we have the intersect_polygons_area be 0.0
            # by the formula of iou calculation, the resulted iou is 0.0
            iou = 0.0

        return iou

    def calculate_confusion_matrix(self):
        """
        Calculate the confusion matrix (i.e., true/false positive/negative) by comparing the predicted and annotated anomalous multi-polygons.

        Args:
            None: no functional specific input arguments required to execute this function.

        Returns:
            confusion_matrix_dict: python dictionary, includes true positive, true negative, false positive, and false negative values. The format of confusion_matrix_dict is the following:
                {
                    "True Positive": TP [floatting number],
                    "False Positive": FP [floatting number],
                    "True Negative": TN [floatting number],
                    "False Negative": FN [floatting number]
                }
        """
        wsi = self.get_wsi_data()["wsi"]

        annotation_multi_polygons = self.get_annotation_polygons()[1]
        total_wsi_area = wsi.level_dimensions[0][0] * wsi.level_dimensions[0][1]

        if tf.io.gfile.exists(self.anomaly_polygon_geojson_gcs_path):
            anomaly_multi_polygons = self.get_anomaly_polygons()[1]
            union_anomaly_annotation_polygons = anomaly_multi_polygons.union(annotation_multi_polygons)
            union_anomaly_annotation_polygons_area = union_anomaly_annotation_polygons.area

            tp = anomaly_multi_polygons.intersection(annotation_multi_polygons)
            fp = anomaly_multi_polygons.difference(annotation_multi_polygons)
            fn = annotation_multi_polygons.difference(anomaly_multi_polygons)

            tp_area = tp.area
            fp_area = fp.area
            fn_area = fn.area
            tn_area = total_wsi_area - union_anomaly_annotation_polygons_area

            confusion_matrix_dict = {
                "True Positive": tp_area,
                "False Positive": fp_area,
                "True Negative": tn_area,
                "False Negative": fn_area
            }
        else:
            logging.warning("anomaly polygons geojson file did not found on the GCS bucket!")
            # when anomaly_polygons is empty, we have the intersect_polygons_area be 0.0
            tp_area = 0.0
            fp_area = 0.0
            fn_area = annotation_multi_polygons.area
            tn_area = total_wsi_area - fn_area

            confusion_matrix_dict = {
                "True Positive": tp_area,
                "False Positive": fp_area,
                "True Negative": tn_area,
                "False Negative": fn_area
            }

        return confusion_matrix_dict

    def anomaly_detection_inference_stat(self):
        """
        Measure the trained anomaly detection model performance by computing the intersection of union (IOU), confusion matrix (i.e., true/false positive/negative), sensitivity, specificity, precision, accuracy, and f1.

        Args:
            None: no functional specific input arguments required to execute this function.
        Returns:
            model_eval_stats_dict: python dictionary, includes iou, confusion matrix, sensitivity, specificity, precision, accuaracy, and f1 results. The format of model_eval_stats_dict is the following:
                {
                    "Intersection Over Union": iou [floatting number],
                    "Confusion Matrix": confusion matrix dictionary,
                    "Sensitivity": sensitivity [floatting number],
                    "Specificity": specificity [floatting number],
                    "Precision": precision [floatting number],
                    "Accuracy": accuracy [floatting number],
                    "F1": f1 [floatting number]
                }
            Noted:
                confusion matrix is a python dictionary with the format as:
                    {
                        "True Positive": TP [floatting number],
                        "False Positive": FP [floatting number],
                        "True Negative": TN [floatting number],
                        "False Negative": FN [floatting number]
                    }
        """
        confusion_matrix_dict = self.calculate_confusion_matrix()
        TP = confusion_matrix_dict["True Positive"]
        FP = confusion_matrix_dict["False Positive"]
        TN = confusion_matrix_dict["True Negative"]
        FN = confusion_matrix_dict["False Negative"]

        iou = self.calculate_iou()

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1 = (2 * TP) / ((2 * TP) + FP + FN)

        if tf.io.gfile.exists(self.anomaly_polygon_geojson_gcs_path):
             precision = TP / (TP + FP)
        else:
            logging.warning("anomaly polygons geojson file did not found on the GCS bucket!")
            precision = 0.0

        model_eval_stats_dict = {
            "Intersection Over Union": iou,
            "Confusion Matrix": confusion_matrix_dict,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "Accuracy": accuracy,
            "F1": f1
        }

        return model_eval_stats_dict
    
    def inference_stats_to_gcs(
        self, 
        stats_json_gcs_path
    ):
        """
        Upload the anomaly detection inference statistics measurement json file to the GCS bucket.

        Args:
            stats_json_gcs_path: str, path to the anomaly detection inference statistics measurement json file in GCS bucket.
        
        Returns:
            None: Nothing is returned, but a plot is created and displayed.
        """
        wsi_data = self.get_wsi_data()
        model_eval_stats_dict = self.anomaly_detection_inference_stat()

        stat_json_file_name = wsi_data["uuid"] + "_inference_statistics_results.json"
        with open(stat_json_file_name, "w") as f:
            json.dump(model_eval_stats_dict, f)

        self.file_local_to_gcs(
            file_local_path = stat_json_file_name,
            file_gcs_path = stats_json_gcs_path,
        )
        if tf.io.gfile.exists(stats_json_gcs_path):
            if len(tf.io.gfile.listdir(stats_json_gcs_path)) != 0:
                os.remove(stat_json_file_name)