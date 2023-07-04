# This follows the instructions at
# https://cloud.google.com/dataflow/docs/guides/using-custom-containers#python

# This can be built with, e.g.:
# export PROJECT=
# export REPO=
# export TAG=0.0.1
# export IMAGE_URI=
# docker build -f Dockerfile . --tag $IMAGE_URI
# docker push $IMAGE_URI

# Note that building this image may need to happen outside of a DLVM, given
# firewall restrictions.


FROM apache/beam_python3.7_sdk


RUN apt-get update && \
  apt-get --assume-yes install \
    openslide-tools \
    python3-openslide && \
  pip3 install --upgrade pip && \
  pip3 install \
    opencv-python-headless==4.5.5.64 \
    openslide-python==1.1.2 \
    matplotlib==3.5.1 \
    scikit-image==0.19.2 \
    shapely==1.8.1.post1 \
    geojson==2.5.0 \
    python-dotenv==0.19.2 \
    geopandas==0.10.2 \
    rtree==0.9.7 \
    pygeos==0.12.0 \
    pyfarmhash==0.3.2

# Useful for confirming the underlying release that the base image is
# built on; this is just retained for ease of debugging:
# RUN cat /etc/os-release && sleep 10
