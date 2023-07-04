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
from google.cloud import storage

import tensorflow as tf
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import openslide
import skimage.color
import skimage.filters
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import shutil
import os
import io
import math


class PCEAD_WSI_TRAIN_PREP():
    """A class of functionalities provided to extarct patches with tissue only regions from WSI file, project 
        image patches into 1024-dimensional feature space, leverage the principal component analysis (i.e., PCA) 
        to select image patches for the anomaly detection baseline model training purpose, down-sample selected 
        image patches to multi-resolution levels, and convert them into TFrecords format.
    
    Args:
        Check input args for wsi_patch_extraction class with details inside the init function.

    Functionalities:
        - Patch Extraction
        - Feature Vector Creation
        - PCA Filtering
        - Image Patches Downsampling
        - TFRecords Creation

    Methods:
        ************************************************************************************************************************
        Note: 
            - If any method is optional (i.e., dependent on user's preference to determine wheher to be executed), the notation
            "[optional]" will be added after the method name; 
            - If any method is shared by multiple functionalities, the notation "[shared]" will be added afrer the method name;
            - If no notations aded after the method name, which indicates that method is requried.
        ************************************************************************************************************************
        - General
            - __init__: used to initialize required args for wsi_patch_extraction class.
            - upload_blob [optional]: used to Upload a file to the bucket. Only used this method if source file is small.
            - local_file_to_gcp [optional]: Used to transfer local file to GCS bucket. This function is the standard approach to transfer local files to GCS
                bucket no matter the storage size and the number of files needs to be transferred.

        - Patch Extraction 
            - Functionality Specific Methods
                - get_wsi_thumbnail: gets thumbnai image from openslide WSI object.
                - otsu_method: uses otsu method to create binary mask from thumbnail.
                - create_binary_mask: used to create a list of binary mask image arrays for each patch with tissue regions.
                - patches_coords: used to generate list of x and y coordinates of starting pixel location (upper left) for each of the tissue only patches.
                - img_levels_range: used to calculate the starting and ending indexes of the image level range corresponding to the level index used in progressive GAN.
                - patch_extraction: used to extract patches with tissue regions only from WSI.
                - save_patches_local [optional][shared]: used to save all patches with tissue only regions extracted from WSI to a local file path.
                - patches_local_to_gcp [optional][shared]: Used to transfer all patches with tissue only regions extracted from WSI stored in a local path to GCS bucket.
            - Utility Static Methods
                - log2n: used to calculate the exponential factor of base number be 2.
                - img_levels: used to calculate the image level corresponding to the level index used in progressive GAN model architecture.
                
        - Feature Vector Creation
            - Functionality Specific Methods
                - customize_resnet50: used to customize user defined resenet50 model to create the image feature vectors.
                - get_image_feature: used to create image feature vectors from customized resnet50 model.

        - PCA Filtering
            - Functionality Specific Methods
                - features_pca_analysis: used to run principal component analysis (PCA) on image feature vectors.
                - pca_results_plot_local [optional]: used to create the thumbnail image of the WSI, PCA explained variance concepts visualization plot, 2D PCA scores scatter plot, and 3D PCA scores scatter plot. The store the plots locally.
                - pca_results_plot_gcs [optional]: Used to upload the thumbnail image of the WSI, PCA explained variance concepts visualization plot, 2D PCA scores scatter plot, and 3D PCA scores scatter plot to GCS bucket.
                - pca_patches_filter: used to apply the principal component analysis (PCA) to select image patch-level feature vectors, which contribute the most variation of the entire WSI.
            - Utility Static Methods
                - pca_evc_plot_local [optional]: used to create PCA explained variance concepts visualization plot and store the plot locally.
                - pca_2d_plot_local [optional]: used to create 2D PCA scores scatter plot and store the plot locally.
                - pca_3d_plot_local [optional]: used to create 3D PCA scores scatter plot and store the plot locally.
                - save_patches_local [optional][shared]: used to save the selected patches filtered by the principal component analysis (PCA) to a local file path.
                - patches_local_to_gcp [optional][shared]: used to transfer the selected patches filtered by the principal component analysis (PCA) stored in a local path to GCS bucket.

        - Image Patches Downsampling
            - Functionality Specific Methods
                - downsamle_patches: used to downsample the extracted image patches (i.e., default extracted patch size be 1024 x 1024, also noted as the level 0 image patch) to multiple resolution levels (i.e., default levels are 1, 2, ..., and up to 8, the corresponding patch size is 512 x 512, 256 x 256, and down to 4 x 4), which is required by the progressive GAN model.

        - TFRecords Creation
            - Functionality Specific Methods
                - patches_to_tfrecord_gcs: used to create tfrecords from all patches extracted from WSIs with tissue regions only, or selected patches from PCA, and store those patches to a GSC bucket.
            - Utility Static Methods
                - _bytes_feature: used to return a bytes_list from a string / byte.
                - _float_feature: used to return a float_list from a float / double.
                - _int64_feature: used to return an int64_list from a bool / enum / int / uint.
    
    Returns:
        Check returning outputs inside each of methods' functions.
    """
    def __init__(
        self,
        bucket_name,
        wsi_gcs_path,
        target_image_width,
        threshold_area_percent,
        patch_level,
        image_patch_depth,
        min_patch_size,
        max_patch_size,
        include_top, 
        weights, 
        input_shape, 
        is_trainable,
        layer_name, 
        n_components,
        pca_filter_percent,
        pca_filter_base_component,
        is_filtered_patches,
        tfrecord_gcs_path
    ):
        """Used to initialize required args for wsi_patch_extraction class.
        
        Args:
            bucket_name: str, the ID of user's GCS bucket.
            wsi_gcs_path: str, GCS path of WSI file.
            target_image_width: int, the approximate width of resultant thumbnail image.
            threshold_area_percent: float, threshold with the percentage of tissue area included within a single patch, which used to determine whether a patch will be extracted from WSI.
            patch_level: int, initial image level of each image patches, default be 0 indicates patch resolution size be (1024,1024).
            image_patch_depth: int, the depth of each image patches, represent the number of channels of each image patches, default be 3, which represents 3 RGB-channel image patches.
            min_patch_size: int, the smallest patch size required for progressive GAN training purpose, default be 4.
            max_patch_size: int, original patch size extracted from WSI, default be 1024.
            include_top: bool, whether to include the fully-connected layer at the top of the network, default be False.
            weights: str, weights of the restnet50 model pre-trained on one of the following three general options, which are None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
            input_shape: tuple, optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value. In our case, default be (1024, 1024, 3).
            is_trainable: bool, whether to re-train the restnet50 model, default be False.
            layer_name: str, user defined layer to customize the restnet50 model, which used to return the output feature vectors from the user defined layer. In our case, default be 'conv4_block1_0_conv', which used to create a customized resnet50 model based on original resnet50 model ended after the 3rd residual block.
            n_components: int, number of components to keep. If n_components is not set all components are kept. Default value be None.
            pca_filter_percent: float, the percentage of the total number of patches used to determine the number of patches will be selected from PCA for the anomaly detection model training purpose. Default value be 0.8.
            pca_filter_base_component: int, the index of the principal component ndarray from PCA used to select image patches for the anomaly detection model training purpose. Default value be 0, which indicates the 1st principal component.
            is_filtered_patches: bool, whether to save the selected patches filtered by the principal component analysis (PCA), or all patches with tissue only regions extracted from WSI to a local file path. Default be True, which intends to save PCA filtered patches.
            tfrecord_gcs_path: str, path to where the tfrecord files will be stored in GCS bucket. 
        
        Returns:
            Return None and only initialize the args of wsi_patch_extraction class.
        """
        self.bucket_name = bucket_name
        self.wsi_gcs_path = wsi_gcs_path
        self.target_image_width = target_image_width
        self.threshold_area_percent = threshold_area_percent
        self.patch_level = patch_level
        self.image_patch_depth = image_patch_depth
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.include_top = include_top
        self.weights = weights
        self.input_shape = input_shape
        self.is_trainable = is_trainable
        self.layer_name = layer_name
        self.n_components = n_components
        self.pca_filter_percent = pca_filter_percent
        self.pca_filter_base_component = pca_filter_base_component
        self.is_filtered_patches = is_filtered_patches
        self.tfrecord_gcs_path = tfrecord_gcs_path
    
    def get_wsi_thumbnail(self):
        """Gets thumbnai image from openslide WSI object.
        
        Args:
            wsi_gcs_path: str, GCS path of WSI file.
            target_image_width: int, the approximate width of resultant thumbnail image.
        
        Returns:
            wsi_uuid: str, uuid of WSI file.
            wsi: OpenSlide.obj, openslide object of WSI file.
            top_level: Python list, content is [thumbnail_height, thumbnail_width, divisor].
            thumbnail: PIL.image, the thumbnail image from openslide.
        """
        gcs = gcsio.GcsIO()
        local_file = "slide_file.svs"

        wsi_uuid = self.wsi_gcs_path.split("/")[-1].split(".")[0]

        with open(local_file, "wb") as f:
            f.write(gcs.open(self.wsi_gcs_path).read())
        wsi = openslide.OpenSlide(filename=local_file)

        # Get the ratio for the target image width.
        divisor = int(wsi.level_dimensions[0][0] / self.target_image_width)
        # Get the height and width of the thumbnail using the ratio.
        patch_size_x = int(wsi.level_dimensions[0][0] / divisor)
        patch_size_y = int(wsi.level_dimensions[0][1] / divisor)
        top_level = [patch_size_x, patch_size_y, divisor]
        # Extract the thumbnail.
        thumbnail = wsi.get_thumbnail(size=(patch_size_x, patch_size_y))
        return wsi_uuid, wsi, top_level, thumbnail
    
    def otsu_method(self):
        """Uses otsu method to create binary mask from thumbnail.

        Args:
            thumbnail: PIL.image, the thumbnail image from openslide.

        Returns:
            binary_img: Binary np.array of shape (thumbnail_width, thumbnail_height, 1).
        """
        thumbnail = self.get_wsi_thumbnail()[3]
        # Convert to grey scale image.
        gs_thumbnail = np.array(thumbnail.convert("L"))
        # Get the otsu threshold value.
        thresh = skimage.filters.threshold_otsu(image=gs_thumbnail)
        # Convert to binary mask.
        binary_img = gs_thumbnail < thresh
        binary_img = binary_img.astype(int)
        return binary_img
    
    def create_binary_mask(self):
        """Used to create a list of binary mask image arrays for each patch with tissue regions.
        
        Args:
            binary_img: Binary np.array of shape (thumbnail_width, thumbnail_height, 1).
        
        Returns:
            list_binary_img: Python list with np.array for tissue only region mask images.
        """         
        binary_img = self.otsu_method()
        
        idx = np.sum(binary_img)
        idx = np.where(binary_img == 1)
        list_binary_img = []
        for i in range(0,len(idx[0]),1):
            x = idx[1][i]
            y = idx[0][i]
            list_binary_img.append(str(x) + " " + str(y))

        return list_binary_img
    
    def patches_coords(self):
        """Used to generate list of x and y coordinates of starting pixel location (upper left) for each of the tissue only patches.

        Args:
            list_binary_img: Python list with np.array for tissue only region mask images.
            wsi: OpenSlide.obj, openslide object of WSI file.
            top_level: Python list, content is [thumbnail_height, thumbnail_width, divisor].
            patch_level: int, initial image level of each image patches, default be 0 indicates patch resolution size be (1024,1024).
            max_patch_size: int, original patch size extracted from WSI, default be 1024.
            threshold_area_percent: float, threshold with the percentage of tissue area included within a single patch, which used \
                to determine whether a patch will be extracted from WSI.
        
        Returns:
            patch_start_x_list: Python list, list of x coordinates of starting pixel location (upper left) for each of the tissue only patches.
            patch_start_y_list: Python list, list of y coordinates of starting pixel location (upper left) for each of the tissue only patches.
        """
        list_binary_img = self.create_binary_mask()
        wsi, top_level = self.get_wsi_thumbnail()[1], self.get_wsi_thumbnail()[2],

        patch_start_x_list = []
        patch_start_y_list = []

        minx = 0
        miny = 0

        condition = self.patch_level > len(wsi.level_dimensions)-1
        if condition:
            raise Exception("not enough levels " + str(self.patch_level) + " " + str(len(wsi.level_dimensions)-1))

        maxx = wsi.level_dimensions[self.patch_level][0]
        maxy = wsi.level_dimensions[self.patch_level][1]
        start_x = minx
        total_num_patches = 0
        selected_num_patches = 0

        # creating sub patches
        # Iterating through x coordinate
        while start_x + self.max_patch_size < maxx:
            # Iterating through y coordinate
            start_y = miny
            while start_y + self.max_patch_size < maxy:
                current_x = int((start_x * wsi.level_downsamples[self.patch_level]) / top_level[2])
                current_y = int((start_y * wsi.level_downsamples[self.patch_level]) / top_level[2])
                tmp_x = start_x + int(self.max_patch_size)
                tmp_y = start_y + int(self.max_patch_size)
                current_x_stop = int((tmp_x * wsi.level_downsamples[self.patch_level]) / top_level[2])
                current_y_stop = int((tmp_y * wsi.level_downsamples[self.patch_level]) / top_level[2])
                total_num_patches = total_num_patches + 1

                flag_list = [1 for i in range(current_x, current_x_stop+1) for j in range(current_y, current_y_stop+1) if str(i) + " " + str(j) in list_binary_img]

                if tmp_x <= maxx and tmp_y <= maxy and (len(flag_list) / ((current_y_stop + 1 - current_y) * (current_x_stop + 1 - current_x))) > self.threshold_area_percent:
                    patch_start_x_list.append(start_x)
                    patch_start_y_list.append(start_y)

                    selected_num_patches = selected_num_patches + 1
 
                start_y = tmp_y
            start_x = tmp_x

        return patch_start_x_list, patch_start_y_list
    
    @staticmethod
    def log2n(n):
        """Used to calculate the exponential factor of base number be 2.
        
        Args:
            n: int, input interger which supposed to be the exponent of 2.
        
        Returns:
            exp_n: int, exponential factor of base number 2.
        """
        if n > 1:
            exp_n = 1 + int(math.log2(n / 2))
        else:
            exp_n = 0

        return exp_n
    
    @staticmethod
    def img_levels(n):
        """Used to calculate the image level corresponding to the level index used in progressive GAN model architecture.
        
        Args:
            n: int, patch size, default value is one of the list [4, 8, 16, 32, 64, 128, 256, 512, 1024].
        
        Returns:
            img_level: int, image level corresponding to the level index used in progressive GAN, default value corresponding \
                to the input arg is in the list [8, 7, 6, 5, 4, 3, 2, 1, 0].
        """
        wsi_levels = list()
        for i in range(11):
            wsi_levels.append(i)

        wsi_levels = sorted(wsi_levels, reverse=True)

        img_level = wsi_levels.index(n)

        return img_level
    
    def img_levels_range(self):
        """Used to calculate the starting and ending indexes of the image level range corresponding to the level index used in progressive GAN.
        
        Args:
            min_patch_size: int, the smallest patch size required for progressive GAN training purpose, default be 4.
            max_patch_size: int, the largest patch size required for progressive GAN training purpose, default be 1024.
            
        Returns:
            min_level_range_index: int, starting index of the image level range corresponding to the level index used in progressive GAN.
            max_level_range_index: int, ending index of the image level range corresponding to the level index used in progressive GAN.
        """
        min_level_range_index = self.img_levels(self.log2n(self.max_patch_size))
        max_level_range_index = self.img_levels(self.log2n(self.min_patch_size)) + 1

        return min_level_range_index, max_level_range_index

    @staticmethod
    def upload_blob(self, source_file_name, destination_blob_name):
        """Used to Upload a file to the bucket. Only used this method if source file is small.
        
        Args:
            bucket: str, the ID of user's GCS bucket.
            source_file_name: str, path to user's local file needs to be uploaded to GCS bucket.
            destination_blob_name: str, the ID of user's GCS object (path to user's desired path to store local file in GCS bucket).
        
        Returns:
            Return N/A, only upload the local file to the user defined destinated path in GCS bucket.
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)
    
    @staticmethod
    def local_file_to_gcp(original_local_path, destination_gcs_path):
        """Used to transfer local file to GCS bucket. This function is the standard approach to transfer local files to GCS bucket 
            no matter the storage size and the number of files needs to be transferred.
        
        Args:
            original_local_path: str, path to user's local file needs to be uploaded to GCS bucket.
            destination_gcs_path: str, path to user's desired path to store local file in GCS bucket.
        
        Returns:
            Return N/A, only upload the local file to the user defined destinated path in GCS bucket.
        """
        # param -m indicates copying files parallelly
        # param -q indicates using the quiet mode to copyinfg files without outout the logging info messages
        cmd = "gsutil -m -q cp -r " + str(original_local_path) + " " + str(destination_gcs_path)
        os.system(cmd)

    def patch_extraction(self):
        """Used to extract patches with tissue regions only from WSI.
        
        Args:
            wsi_uuid: str, uuid of WSI file.
            wsi: OpenSlide.obj, openslide object of WSI file.
            patch_start_x_list: Python list, list of x coordinates of starting pixel location (upper left) for each of the tissue only patches.
            patch_start_y_list: Python list, list of y coordinates of starting pixel location (upper left) for each of the tissue only patches.
            min_level_range_index: int, starting index of the image level range corresponding to the level index used in progressive GAN.
            max_level_range_index: int, ending index of the image level range corresponding to the level index used in progressive GAN.
            patch_level: int, initial image level of each image patches, default be 0 indicates patch resolution size be (1024,1024).
            max_patch_size: int, original patch size extracted from WSI, default be 1024.
        
        Returns:
            image_patches_dict: Python dict, the format of the python dictionary is 
                {"patch_name_1": image patch object 1,
                "patch_name_2: image patch object 2,
                ...
                }
            Noted that the image object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object.
        """
        wsi_uuid, wsi = self.get_wsi_thumbnail()[0], self.get_wsi_thumbnail()[1]

        patch_start_x_list, patch_start_y_list = self.patches_coords()

        image_patch_names = list()
        image_patch_objects = list()

        for i in range(0,len(patch_start_x_list),1):
            start_x_coord = int(patch_start_x_list[i] * wsi.level_downsamples[self.patch_level])
            start_y_coord = int(patch_start_y_list[i] * wsi.level_downsamples[self.patch_level])

            patch_object = wsi.read_region(
                location=(start_x_coord, start_y_coord), 
                level=self.patch_level, 
                size=(self.max_patch_size, self.max_patch_size)
                )
            patch_object = patch_object.resize((self.max_patch_size, self.max_patch_size))

            patch_name = wsi_uuid + "_level_" + str(self.patch_level) + "_x_" + str(start_x_coord) + "_y_" + str(start_y_coord) + ".png"

            image_patch_names.append(patch_name)
            image_patch_objects.append(patch_object)
        
        image_patches_dict = dict(zip(image_patch_names, image_patch_objects))
        
        return image_patches_dict

    def customize_resnet50(self):
        """Used to customize user defined resenet50 model to create the image feature vectors.

        Args:
            input_top: bool, whether to include the fully-connected layer at the top of the network, default be False.
            weights: str, weights of the restnet50 model pre-trained on one of the following three general options, which are None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
            input_shape: tuple, optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value. In our case, default be (1024, 1024, 3).
            is_trainable: bool, whether to re-train the restnet50 model, default be False.
            layer_name: str, user defined layer to customize the restnet50 model, which used to return the output feature vectors from the user defined layer. In our case, default be 'conv4_block1_0_conv', which used to create a customized resnet50 model based on original resnet50 model ended after the 3rd residual block.
        
        Returns:
            custom_res50_model: tf.Model, a customized resnet50 model based on user provided last output layer to create image feature vectors.
            adaptive_mean_spatial_layer: tf.layer, an adaptive mean-spatial pooling intended to be added after the customized resnet50 model.
        """
        # Load the ResNet50 model
        resnet50_model = tf.keras.applications.resnet50.ResNet50(
            include_top = self.include_top,
            weights = self.weights,
            input_shape = self.input_shape
        )
        resnet50_model.trainable = self.is_trainable  # Free Training when is.trainable is False

        # Create a new resnet50 model based on original resnet50 model ended after the 3rd residual block
        # layer_name = 'conv4_block1_0_conv'
        custom_res50_model = tf.keras.Model(
            inputs = resnet50_model.input, 
            outputs = resnet50_model.get_layer(self.layer_name).output
        )

        # Add adaptive mean-spatial pooling after the new model
        adaptive_mean_spatial_layer = tf.keras.layers.GlobalAvgPool2D()
        
        return custom_res50_model, adaptive_mean_spatial_layer
    
    def get_image_feature(self):
        """Used to create image feature vectors from customized resnet50 model.
        
        Args:
            image_patch_dict: Python dict, the format of the python dictionary is 
                {"patch_name_1": image patch object 1,
                "patch_name_2: image patch object 2,
                ...
                }
            Noted the image object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object.
            custom_res50_model: tf.Model, a customized resnet50 model based on user provided last output layer to create image feature vectors.
            adaptive_mean_spatial_layer: tf.layer, an adaptive mean-spatial pooling intended to be added after the customized resnet50 model.
        
        Returns:
            patch_features_dict: Python dict, the format of the python dictionary is
                {"patch_names": ["patch_name_1", "patch_name_2", ...]
                "patch_features_flat": (patch_feature_tensor_1, patch_feature_tensor_2, ...)
                }
            Noted that the patch_feature_tensor is (1, 1024) dimensional feature vector corrrespoing to each of the image patch PIL object. The patch_features_flat is a ndarray with shape be (#image patches, 1024). 
        """
        custom_res50_model, adaptive_mean_spatial_layer = self.customize_resnet50()

        image_patches_dict = self.patch_extraction()
        patch_features = list()
        for patch_name in image_patches_dict.keys():
            patch_tensor = tf.convert_to_tensor(value=image_patches_dict[patch_name])[:, :, :3]

            image_patch_array = np.array(object=patch_tensor)
            image_batch = np.expand_dims(a=image_patch_array, axis=0)
            image_patch = tf.keras.applications.resnet50.preprocess_input(image_batch.copy())
            predicts = custom_res50_model.predict(image_patch)
            image_patch_feature_array = adaptive_mean_spatial_layer(predicts)

            patch_feature = image_patch_feature_array.numpy()
            patch_feature = tf.convert_to_tensor(value=patch_feature)
            
            patch_features.append(patch_feature)

        patch_features_flat = [
            tf.reshape(
                tensor=i, 
                shape=(i.shape[1], )
            ).numpy() 
            for i in patch_features
        ]
        patch_features_flat = np.array(object=patch_features_flat)

        patch_names = list(image_patches_dict.keys())

        patch_features_dict = {
            "patch_names": patch_names,
            "patch_features_flat": patch_features_flat
        }
        
        return patch_features_dict
    
    def features_pca_analysis(self):
        """Used to run principal component analysis (PCA) on image feature vectors. Use Sklearn PCA package
        to run principal component analysis. Check PCA functionality documentation in SKlearn, the link is 
        in below,
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        
        Args:
            patch_features_dict: Python dict, the format of the python dictionary is
                {"patch_names": ["patch_name_1", "patch_name_2", ...]
                "patch_features_flat": (patch_feature_tensor_1, patch_feature_tensor_2, ...)
                }
            Noted that the patch_feature_tensor is (1, 1024) dimensional feature vector corrrespoing to each of the image patch PIL object. The patch_features_flat is a ndarray with shape be (#image patches, 1024). 
            n_components: int, number of components to keep. If n_components is not set all components are kept. Default value be None.
        
        Returns:
            features_pca: ndarray, transformed values returned by fitting the model with the input training data (i.e., array-like training data 
            with the shape be (n_sample, n_features), where n_samples is the number of samples and n_features is the number of features), where 
            the shape is (n_samples, n_components).
            exp_var_pca: ndarray, percentage of variance explained by each of the selected components with the shape be (n_components,). If n_components
            is not set then all components are stored and the sum of the ratios is equal to 1.0.
            cum_sum_eigenvalues: ndarray, the cumulative sum of the elements along a given axis. The returned value is a new array holding the 
            result is returned unless out is specified, in which case a reference to out is returned. The result has the same size as the input 
            array, and the same shape as the input array if axis is not None or the input array is a 1-d array.
        """
        pca = PCA(n_components=self.n_components)

        patch_features_dict = self.get_image_feature()
        patch_features_flat = patch_features_dict["patch_features_flat"]
        
        # standardize features by removing the mean and scalling to unit variance using StandardScaler function in SKlearn package
        # check SKlearn package StandardScaler documentation page:
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        sc = StandardScaler()
        # compute the mean and std to be used for later scaling
        # input is the array-like data used to compute the mean and standard deviation used for later scaling along the features axis
        # return the fitted scaler object
        sc.fit(X=patch_features_flat)
        # perform standardization by centering and scaling the input data (i.e., array-like, sparse matrix of 
        # shape (n_samples, n_features), used to scale along the features axis), and return the transformed array
        transform_features = sc.transform(X=patch_features_flat)
        # fit to data, then transform it
        # fits transformer to input samples and returns a transformed version of input samples
        features_pca = pca.fit_transform(transform_features) 

        # Determine explained variance using explained_variance_ration_ attribute
        exp_var_pca = pca.explained_variance_ratio_

        # Cumulative sum of eigenvalues; This will be used to create step plot
        # for visualizing tnhe variance explained by each principal component.
        cum_sum_eigenvalues = np.cumsum(a=exp_var_pca)

        return features_pca, exp_var_pca, cum_sum_eigenvalues

    @staticmethod
    def pca_evc_plot_local(
        pca_plot_local_path,
        exp_var_pca,
        cum_sum_eigenvalues
    ):
        """Used to create PCA explained variance concepts visualization plot and store the plot locally.
        
        Args:
            pca_plot_local_path: str, path to where the PCA explained variance concepts visualization plot will be stored locally.
            exp_var_pca: ndarray, percentage of variance explained by each of the selected components with the shape be (n_components,). If n_components
            is not set then all components are stored and the sum of the ratios is equal to 1.0.
            cum_sum_eigenvalues: ndarray, the cumulative sum of the elements along a given axis. The returned value is a new array holding the 
            result is returned unless out is specified, in which case a reference to out is returned. The result has the same size as the input 
            array, and the same shape as the input array if axis is not None or the input array is a 1-d array.
        
        Returns:
            Return N/A, only save PCA explained variance concepts visualization plot locally.
        """
        fig_pca_evc = plt.figure()
        plt.bar(
            x=range(0,len(exp_var_pca)), 
            height=exp_var_pca, 
            alpha=0.5, 
            align="center", 
            label="Individual Explained Variance"
        )

        plt.step(
            x=range(0,len(cum_sum_eigenvalues)), 
            y=cum_sum_eigenvalues, 
            where="mid",
            label="Cumulative Explained Variance"
        )

        plt.title("PCA Explained Variance Concepts Plot")
        plt.xlabel("Principal Component Index")
        plt.ylabel("Explained Variance Ratio")
        plt.legend(loc="best")
        plt.tight_layout()

        fig_pca_evc.savefig(os.path.join(pca_plot_local_path, "pca_evc_plot.png"))
    
    @staticmethod
    def pca_2d_plot_local(
        pca_plot_local_path, 
        features_pca, 
        x_pca_index, 
        y_pca_index
    ):
        """Used to create 2D PCA scores scatter plot and store the plot locally.
        
        Args:
            pca_plot_local_path: str, path to where the 2D PCA scores plot will be stored locally.
            features_pca: ndarray, transformed values returned by fitting the model with the input training data (i.e., array-like training data 
            with the shape be (n_sample, n_features), where n_samples is the number of samples and n_features is the number of features), where 
            the shape is (n_samples, n_components).
            x_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the x-axis.
            y_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the y-axis.
        
        Returns:
            Return N/A, only save 2D PCA scores scatter plot locally.
        """
        fig_pca_2d = plt.figure()
        sns.scatterplot(
            x=features_pca[:, x_pca_index], 
            y=features_pca[:, y_pca_index]
        )
        plt.title("2D PCA Scatter Plot")
        plt.xlabel("Principal Component - " + str(x_pca_index + 1))
        plt.ylabel("Principal Component - " + str(y_pca_index + 1))

        fig_pca_2d.savefig(os.path.join(pca_plot_local_path, "pca_2d_plot.png"))

    @staticmethod
    def pca_3d_plot_local(
        pca_plot_local_path,
        features_pca, 
        x_pca_index, 
        y_pca_index, 
        z_pca_index
    ):
        """Used to create 3D PCA scores scatter plot and store the plot locally.
        
        Args:
            pca_plot_local_path: str, path to where the 3D PCA scores plot will be stored locally.
            features_pca: ndarray, transformed values returned by fitting the model with the input training data (i.e., array-like training data 
            with the shape be (n_sample, n_features), where n_samples is the number of samples and n_features is the number of features), where 
            the shape is (n_samples, n_components).
            x_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the x-axis.
            y_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the y-axis.
            z_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the z-axis.
        
        Returns:
            Return N/A, only save 3D PCA scores scatter plot locally.
        """
        fig_pca_3d = plt.figure(figsize = (16, 9))
        ax = plt.axes(projection ="3d")
        
        # Add x, y gridlines
        ax.grid(
            b = True, 
            color ="grey",
            linestyle ="-.", 
            linewidth = 0.3,
            alpha = 0.2)
        
        # Creating color map
        my_cmap = plt.get_cmap("hsv")
        
        # Creating plot
        sctt = ax.scatter3D(
            xs=features_pca[:, x_pca_index], 
            ys=features_pca[:, y_pca_index], 
            zs=features_pca[:, z_pca_index], 
            alpha = 0.8,
            c = (features_pca[:, x_pca_index] + features_pca[:, y_pca_index] + features_pca[:, z_pca_index]),
            cmap = my_cmap,
            marker ="^"
        )
        
        plt.title("3D PCA Scatter Plot")
        ax.set_xlabel(
            xlabel="Principal Component - " + str(x_pca_index + 1), 
            fontweight ="bold"
        )
        ax.set_ylabel(
            ylabel="Principal Component - " + str(y_pca_index + 1), 
            fontweight ="bold"
        )
        ax.set_zlabel(
            zlabel="Principal Component - " + str(z_pca_index + 1), 
            fontweight ="bold"
        )
        fig_pca_3d.colorbar(
            mappable=sctt, 
            ax=ax, 
            shrink=0.5, 
            aspect=5
        )
 
        fig_pca_3d.savefig(os.path.join(pca_plot_local_path, "pca_3d_plot.png"))
    
    def pca_results_plot_local(self, x_pca_index, y_pca_index, z_pca_index):
        """Used to create the thumbnail image of the WSI, PCA explained variance concepts visualization plot, 2D PCA scores scatter plot, and 3D PCA
        scores scatter plot. The store the plots locally.
        
        Args:
            pca_plot_local_path: str, path to where the plots (listed in the description of the function) will be stored locally.
            thumbnail: PIL.image, the thumbnail image from openslide.
            features_pca: ndarray, transformed values returned by fitting the model with the input training data (i.e., array-like training data 
            with the shape be (n_sample, n_features), where n_samples is the number of samples and n_features is the number of features), where 
            the shape is (n_samples, n_components).
            exp_var_pca: ndarray, percentage of variance explained by each of the selected components with the shape be (n_components,). If n_components
            is not set then all components are stored and the sum of the ratios is equal to 1.0.
            cum_sum_eigenvalues: ndarray, the cumulative sum of the elements along a given axis. The returned value is a new array holding the 
            result is returned unless out is specified, in which case a reference to out is returned. The result has the same size as the input 
            array, and the same shape as the input array if axis is not None or the input array is a 1-d array.
            x_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the x-axis.
            y_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the y-axis.
            z_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the z-axis.
        
        Returns:
            Return N/A, only save the plots (listed in the description of the function) locally.
        """
        pca_plot_local_path, thumbnail = self.get_wsi_thumbnail()[0], self.get_wsi_thumbnail()[-1]
        features_pca, exp_var_pca, cum_sum_eigenvalues = self.features_pca_analysis()

        if not os.path.exists(pca_plot_local_path):
            os.mkdir(pca_plot_local_path)

        thumbnail.save(os.path.join(pca_plot_local_path, "thumbnail.png"))
        self.pca_evc_plot_local(
            pca_plot_local_path=pca_plot_local_path,
            exp_var_pca=exp_var_pca,
            cum_sum_eigenvalues=cum_sum_eigenvalues
        )
        self.pca_2d_plot_local(
            pca_plot_local_path=pca_plot_local_path,
            features_pca=features_pca,
            x_pca_index=x_pca_index,
            y_pca_index=y_pca_index
        )
        self.pca_3d_plot_local(
            pca_plot_local_path=pca_plot_local_path,
            features_pca=features_pca,
            x_pca_index=x_pca_index,
            y_pca_index=y_pca_index,
            z_pca_index=z_pca_index
        )
    
    def pca_results_plot_gcs(self, x_pca_index, y_pca_index, z_pca_index, pca_plot_gcs_path):
        """Used to upload the thumbnail image of the WSI, PCA explained variance concepts visualization plot, 2D PCA scores scatter plot, and 3D PCA
        scores scatter plot to GCS bucket.
        
        Args:
            pca_plot_local_path: str, path to where the plots (listed in the description of the function) has been stored locally.
            x_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the x-axis.
            y_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the y-axis.
            z_pca_index: int, index to determine which principla component related ndarray features will be used to plot on the z-axis.
            pca_plot_gcs_path: str, path to where the plots (listed in the description of the function) will be stored in GCS bucket.
        
        Returns:
            Return N/A, only upload the plots (listed in the description of the function) from local path to GCS bucket.
        """
        pca_plot_local_path = self.get_wsi_thumbnail()[0]

        if not os.path.exists(pca_plot_local_path):
            self.pca_results_plot_local(
                x_pca_index=x_pca_index,
                y_pca_index=y_pca_index,
                z_pca_index=z_pca_index
            )
            
        self.local_file_to_gcp(
            original_local_path=pca_plot_local_path, 
            destination_gcs_path=pca_plot_gcs_path
        )

        # remove patches stored locally only after image patches have been uploaded to GCS bucket
        pca_gcs_exist_condition = tf.io.gfile.exists(os.path.join(pca_plot_gcs_path, pca_plot_local_path))
        
        if pca_gcs_exist_condition:
            pca_upload_gcs_pass_condition = len(
                tf.io.gfile.listdir(os.path.join(pca_plot_gcs_path, pca_plot_local_path))
            ) == len(os.listdir(pca_plot_local_path))
            if pca_upload_gcs_pass_condition:
                shutil.rmtree(pca_plot_local_path)
    
    def pca_patches_filter(self):
        """Used to apply the principal component analysis (PCA) to select image patch-level feature vectors, which contribute the
        most variation of the entire WSI.
        
        Args:
            image_patches_dict: Python dict, the format of the python dictionary is 
                {"patch_name_1": image patch object 1,
                "patch_name_2: image patch object 2,
                ...
                }
            Noted that the image object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object.
            features_pca: ndarray, transformed values returned by fitting the model with the input training data (i.e., array-like training data 
            with the shape be (n_sample, n_features), where n_samples is the number of samples and n_features is the number of features), where 
            the shape is (n_samples, n_components).
            pca_filter_percent: float, the percentage of the total number of patches used to determine the number of patches will be selected from PCA for the anomaly detection model training purpose. Default value be 0.8.
            pca_filter_base_component: int, the index of the principal component ndarray from PCA used to select image patches for the anomaly detection model training purpose. Default value be 0, which indicates the 1st principal component.
        Returns:
            filtered_patches_dict: Python dict, the format of the python dictionary is 
                {"selected_patch_name_1": selected image patch object 1,
                "selected_patch_name_2: selected image patch object 2,
                ...
                }
            Noted that the selected image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object.
        """
        image_patches_dict = self.patch_extraction()
        features_pca = self.features_pca_analysis()[0]

        num_samples = len(image_patches_dict)
        num_pca_filter_samples = int(num_samples * self.pca_filter_percent)
        pca_filter_patch_index = list(
            features_pca[:, self.pca_filter_base_component].argsort()[-(num_pca_filter_samples):][::-1]
        )

        image_patches_names = list(image_patches_dict.keys())
        filtered_patches_names = list()
        for i in pca_filter_patch_index:
            filtered_patch_name = image_patches_names[i]
            filtered_patches_names.append(filtered_patch_name)
        
        filtered_patches_objects = list()
        for i in filtered_patches_names:
            filtered_patch_object = image_patches_dict[i]
            filtered_patches_objects.append(filtered_patch_object)
        
        filtered_patches_dict = dict(zip(filtered_patches_names, filtered_patches_objects))
        return filtered_patches_dict
    
    def save_patches_local(self):
        """Used to save either all patches with tissue only regions extracted from WSI, or the selected patches filtered 
        by the principal component analysis (PCA) to a local file path.
        
        Args:
            patch_local_path: str, path to patches stored locally.
            is_filtered_patches: bool, whether to save the selected patches filtered by the principal component analysis (PCA), or all patches with tissue only regions extracted from WSI to a local file path. Default be True, which intends to save PCA filtered patches.
            image_patches_dict: Python dict, python dictionary with key be patch name and value be corresponding patch PIL object. The format of 
            this dictionary is:
                {"selected_patch_name_1": selected image patch feature vector 1,
                "selected_patch_name_2: selected image patch feature vector 2,
                ...
                },
            when is_filtered_patches be True (noted that the selected image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object);
            The format of this dictionary is:
                {"patch_name_1": image patch object 1,
                "patch_name_2: image patch object 2,
                ...
                },
            when is_filtered_patches be False (noted that the image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object). 
        
        Returns:
            Return N/A, only save image patches in ".png" format to a local file path.
        """
        patch_local_path = self.get_wsi_thumbnail()[0]

        if not os.path.exists(patch_local_path):
            os.mkdir(patch_local_path)
        
        if self.is_filtered_patches:
            image_patches_dict = self.pca_patches_filter()
        else:
            image_patches_dict = self.patch_extraction()

        for image_patch_name in image_patches_dict.keys():
            image_patch_object = image_patches_dict[image_patch_name]
            image_patch_object.save(os.path.join(patch_local_path, image_patch_name))

    def patches_local_to_gcp(self, patch_gcs_path):
        """Used to transfer either all patches with tissue only regions extracted from WSI, or the selected patches filtered 
        by the principal component analysis (PCA) stored in a local path to GCS bucket.
        
        Args:
            is_filtered_patches: bool, whether to save the selected patches filtered by the principal component analysis (PCA), or all patches with tissue only regions extracted from WSI to a local file path. Default be True, which intends to save PCA filtered patches.
            patch_local_path: str, path to patches stored locally.
            patch_gcs_path: str, path to patches expected to be stored in GCS bucket.
        
        Returns:
            Return N/A, only transfer image patches stored locally to GCS bucket and remove the local stored image patches files.
        """
        patch_local_path = self.get_wsi_thumbnail()[0]

        if not os.path.exists(patch_local_path):
            self.save_patches_local()
            
        self.local_file_to_gcp(
            original_local_path=patch_local_path, 
            destination_gcs_path=patch_gcs_path
        )

        # remove patches stored locally only after image patches have been uploaded to GCS bucket
        patch_gcs_exist_condition = tf.io.gfile.exists(os.path.join(patch_gcs_path, patch_local_path))
        
        if patch_gcs_exist_condition:
            patch_upload_gcs_pass_condition = len(
                tf.io.gfile.listdir(os.path.join(patch_gcs_path, patch_local_path))
            ) == len(os.listdir(patch_local_path))
            if patch_upload_gcs_pass_condition:
                shutil.rmtree(patch_local_path)
    
    def downsamle_patches(self):
        """Used to downsample the extracted image patches (i.e., default extracted patch size be 1024 x 1024, also noted as the level 0 image patch) to multiple resolution levels (i.e., default levels are 1, 2, ..., and up to 8, the corresponding patch size is 512 x 512, 256 x 256, and down to 4 x 4), which is required by the progressive GAN model.
        
        Args:
            min_level_range_index: int, starting index of the image level range corresponding to the level index used in progressive GAN.
            max_level_range_index: int, ending index of the image level range corresponding to the level index used in progressive GAN.
            is_filtered_patches: bool, whether to save the selected patches filtered by the principal component analysis (PCA), or all patches with tissue only regions extracted from WSI to a local file path. Default be True, which intends to save PCA filtered patches.
            max_image_patches_dict: Python dict, python dictionary with key be patch name and value be corresponding patch PIL object. The format of 
            this dictionary is:
                {"selected_patch_name_1": selected image patch object 1,
                "selected_patch_name_2: selected image patch object 2,
                ...
                },
            when is_filtered_patches be True (noted that the selected image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object);
            The format of this dictionary is:
                {"patch_name_1": image patch object 1,
                "patch_name_2: image patch object 2,
                ...
                },
            when is_filtered_patches be False (noted that the image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object). 
        Returns:
            image_patches_dict: Python dict, python dictionary with key be the image level, and corresponding value be another python dictionary with the key be the patch name and value be corresponding patch PIL object. The format of 
            this dictionary is:
                {"L0": 
                    {
                    "selected_patch_name_1": selected image patch object 1,
                    "selected_patch_name_2: selected image patch object 2,
                    ...
                    },
                "L1":
                    {
                    "selected_patch_name_1": selected image patch object 1,
                    "selected_patch_name_2: selected image patch object 2,
                    ...
                    },
                ...
                },
            when is_filtered_patches be True (noted that the selected image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object);
            The format of this dictionary is:
                {"L0": 
                    {
                    "patch_name_1": image patch object 1,
                    "patch_name_2: image patch object 2,
                    ...
                    },
                "L1":
                    {
                    "patch_name_1": image patch object 1,
                    "patch_name_2: image patch object 2,
                    ...
                    },
                ...
                },
            when is_filtered_patches be False (noted that the image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object). 
        """
        min_level_range_index, max_level_range_index = self.img_levels_range()
        
        if self.is_filtered_patches:
            max_image_patches_dict = self.pca_patches_filter()
        else:
            max_image_patches_dict = self.patch_extraction()
        
        #rescaleing image
        max_size_patch_objects = list(max_image_patches_dict.values())
        max_size_patch_names = list(max_image_patches_dict.keys())

        image_patch_names = [[] for i in range(min_level_range_index, max_level_range_index)]
        image_patch_files = [[] for i in range(min_level_range_index, max_level_range_index)]

        for index in range(len(max_image_patches_dict)):
            for level in range(min_level_range_index, max_level_range_index):
                level_size = int(self.max_patch_size / (2 ** level))
                patch_size = (level_size, level_size)
                patch_object = max_size_patch_objects[index].resize(patch_size)

                wsi_uuid = max_size_patch_names[index].split("_level_")[0]
                patch_object_level = max_size_patch_names[index].split("_level_")[-1].split("_x_")[0]
                start_xy_coord = max_size_patch_names[index].split("_level_")[-1].split("_x_")[-1]
                patch_name = wsi_uuid + "_level_" +  patch_object_level + "_x_" + start_xy_coord

                image_patch_files[level].append(patch_object)
                image_patch_names[level].append(patch_name)
        image_patches_dict = {"L" + str(i): dict(zip(image_patch_names[i], image_patch_files[i])) for i in range(min_level_range_index, max_level_range_index)}
        
        return image_patches_dict
    
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def patches_to_tfrecord_gcs(self):
        """Used to create tfrecords from all patches extracted from WSIs with tissue regions only, or selected patches from PCA, and 
        store those patches to a GSC bucket.
        
        Args:
            is_filtered_patches: bool, whether to save the selected patches filtered by the principal component analysis (PCA), or all patches with tissue only regions extracted from WSI to a local file path. Default be True, which intends to save PCA filtered patches.
            image_patches_dict: Python dict, python dictionary with key be the image level, and corresponding value be another python dictionary with the key be the patch name and value be corresponding patch PIL object. The format of 
            this dictionary is:
                {"L0": 
                    {
                    "selected_patch_name_1": selected image patch object 1,
                    "selected_patch_name_2: selected image patch object 2,
                    ...
                    },
                "L1":
                    {
                    "selected_patch_name_1": selected image patch object 1,
                    "selected_patch_name_2: selected image patch object 2,
                    ...
                    },
                ...
                },
            when is_filtered_patches be True (noted that the selected image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object);
            The format of this dictionary is:
                {"L0": 
                    {
                    "patch_name_1": image patch object 1,
                    "patch_name_2: image patch object 2,
                    ...
                    },
                "L1":
                    {
                    "patch_name_1": image patch object 1,
                    "patch_name_2: image patch object 2,
                    ...
                    },
                ...
                },
            when is_filtered_patches be False (noted that the image patch PIL object is the maximum image patch PIL oject, default be 1024 x 1024 patch size image patch PIL object). 
            tfrecord_gcs_path: str, path to where the tfrecord files will be stored in GCS bucket. 
            
        Returns:
            Return N/A, and only create and save tfrecords in GCS bucket.
        """
        image_patches_dict = self.downsamle_patches()
        wsi_uuid = list(list(image_patches_dict.values())[0].keys())[0].split("_")[0]

        if not tf.io.gfile.exists(os.path.join(self.tfrecord_gcs_path, wsi_uuid)):
            tf.io.gfile.mkdir(os.path.join(self.tfrecord_gcs_path, wsi_uuid))
        
        for level_str in list(image_patches_dict.keys()):
            tfrecord_full_path = os.path.join(self.tfrecord_gcs_path, wsi_uuid, wsi_uuid + "_" + level_str + ".tfrecords")
            tf_writer = tf.io.TFRecordWriter(tfrecord_full_path)

            for patch_name in list(image_patches_dict[level_str].keys()):
                patch_file = image_patches_dict[level_str][patch_name]
                patch_byte_arry = io.BytesIO()
                patch_file.save(patch_byte_arry, format="PNG")
                patch_byte_arry = patch_byte_arry.getvalue()

                patch_format = "png"
                patch_height = int(self.max_patch_size / (2 ** int(level_str.split("L")[-1]))) 
                patch_width = patch_height

                feature = {
                    "image/name": self._bytes_feature(patch_name.encode("utf8")),
                    "image/format": self._bytes_feature(patch_format.encode("utf8")),
                    "image/height": self._int64_feature(patch_height),
                    "image/width": self._int64_feature(patch_width),
                    "image/depth": self._int64_feature(self.image_patch_depth),
                    "image/encoded": self._bytes_feature(patch_byte_arry)
                }
                Example = tf.train.Example(features=tf.train.Features(feature=feature))
                Serialized = Example.SerializeToString()
                tf_writer.write(Serialized)
            tf_writer.close()