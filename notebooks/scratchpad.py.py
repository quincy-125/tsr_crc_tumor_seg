import os
import logging

# os.chdir("../")

# from proganomaly_modules.load_var_module import env_var
# from dotenv import load_dotenv

# load_dotenv("notebooks/.env")
# env_var_dict = env_var.get_env_var()


def flatten(object: dict):
    for key in object.keys():
        value = object[key]
        if type(value) == dict:
            flatten(value)
        else:
            logging.info('Adding "%s"', key)
            globals().update({key: value})


flatten(config)

# Disable GPUs (for our DLVMs):
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# wsi_stitch_gcs_path = ""

# wsi_stitch_gcs_path = (
#     ""
# )
execfile("proganomaly_modules/beam_image_stitch/components/pre_inference_wsi.py")
