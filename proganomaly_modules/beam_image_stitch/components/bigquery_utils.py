"""
Functions for interacting with BigQuery in Google Cloud Platform (GCP) / AI
Factory (AIF).

See https://cloud.google.com/bigquery/docs/samples/bigquery-insert-geojson and
https://github.com/googleapis/python-bigquery for examples and upstream
documentation.
"""

import ctypes
from enum import IntEnum, unique
import farmhash  # From pyfarmhash
from google.cloud import bigquery
import logging
import sys
from typing import List, Tuple, Union

bigquery_client = None
bigquery_client_project_id = ""

nucleus_tumor_threshold_default = 7.0


# The use of farmhash.fingerprint64 follows the example at
# https://gist.github.com/pcejrowski/6abcb9813f3125b63da1f8ddda986bde
def string_to_partition(string: str, max_return: int = 4000) -> int:
    """
    Use Google's farmhash.fingerprint64 hash function, which deterministically
    transforms a string into an int (e.g., to create a BigQuery partition
    number from a slide UUID).

    *Multiple strings will return the same output int.*

    Args:
        string (str): The string to hash.

        max_return (int, optional): The highest acceptable number to return.
        Defaults to 4000.

    Returns:
        int: An integer between 0 and max_return.
    """
    # We need to use something like c_long() here because fingerprint64()
    # returns an unsigned int, which is different from how the equivalent
    # command within BigQuery works:
    hash_value = abs(int(ctypes.c_long(farmhash.fingerprint64(string)).value))
    return abs(hash_value % max_return)


def create_bigquery_client(project_id: str) -> None:
    """
    Create a client connection to BigQuery.
    (See https://googleapis.dev/python/bigquery/latest/generated/google.cloud.bigquery.client.Client.html)

    Args:
        project_id (str): The name of the BigQuery project to connect to.

    Returns:
        None: (Nothing is returned, but the global variable "bigquery_client"
        is updated to be a google.cloud.bigquery.client.Client object.)
    """
    # Affect the global client variable.
    global bigquery_client
    global bigquery_client_project_id
    if bigquery_client is None or bigquery_client_project_id != project_id:
        bigquery_client = bigquery.Client(project=project_id)
        bigquery_client_project_id = project_id
    return


def delete_slide_rows(
    project_id: str,
    table_name: str,
    slide_uuid: str,
    dataset_name: str = "phi_main_us_p",
) -> None:
    """
    Delete rows associated with slide_uuid from project_id.table_name in
    BigQuery.

    Args:
        project_id (str): The name of the BigQuery project to query.

        table_name (str): The name of the BigQuery table within
        the dataset_name dataset.

        slide_uuid (str): A unique, consistent identifier for a slide, used to
        look up that slide's values in BigQuery.

        dataset_name (str, optional): The name of the BigQuery dataset to query.
        Defaults to "phi_main_us_p".

    Returns:
        None: (Nothing is returned, but rows are deleted from BigQuery)
    """
    # Note that one may get an error related to deleting rows
    # "in the streaming buffer". According to
    # https://sql.info/d/solved-bigquery-error-update-or-delete-statement-over-table-would-affect-rows-in-the-streaming-buffer-which-is-not-supported ,
    # rows written within the last 30 minutes are typically unable to be
    # deleted.
    partition_number = string_to_partition(slide_uuid)

    sql_statement = f"""
        DELETE FROM `{project_id}.{dataset_name}.{table_name}`
        WHERE partition_number={partition_number}
        AND slide_uuid="{slide_uuid}";
    """

    output = run_sql(
        project_id,
        sql_statement,
        description=f'removing records from table `{project_id}.{dataset_name}.{table_name}` for partition_number "{partition_number}" and slide_uuid "{slide_uuid}"',
    )

    logging.info(
        'Output from removing records for partition_number "%d" and slide_uuid "%s": "%s"',
        partition_number,
        slide_uuid,
        "; ".join(output),
    )


def pixels_to_meters(pixels: float, microns_per_pix: float) -> float:
    """
    Convert number of pixels into the equivalent number of meters for a given
    slide image.

    To calculate meters per pixel, run `1/pixels_to_meters(...)`

    Args:
        pixels (float): The number of pixels to convert into number of meters.

        microns_per_pix (float): The number of microns a
        pixel represents. Using OpenSlide, for example, one can retrieve this
        value from a slide image with `openslide.OpenSlide(slide_uri).mpp-x`
        or `openslide.OpenSlide(slide_uri).mpp-y`.

    Returns:
        float: The number of meters represented by pixels.
    """
    microns = pixels * microns_per_pix
    meters = microns * 1e-6
    return meters


@unique
class Slide_Type(IntEnum):
    SKIN = 0
    COLON = 1
    LUNG = 2


def slide_to_bigquery(
    slide_uuid: str,
    slide_type: Union[Slide_Type, int],
    project_id: str,
    slide_microns_per_pixel: float,
    slide_dimensions: Tuple[int, int],
    dataset_name: str = "phi_main_us_p",
) -> None:
    """
    Insert a slide (if it does not already exist) into the 'slides' table in the
    project_id.dataset_name dataset in BigQuery.

    Args:
        slide_uuid (str): A unique, consistent identifier for a slide, used to
        look up that slide's values in BigQuery.

        slide_type (Union[Slide_Type, int]): An indicator of the type of slide:
            '0' indicates Skin
            '1' indicates Colon
            '2' indicates Lung

        slide_microns_per_pixel (float): The number of microns per pixel for
        the Whole Slide Image.

        slide_dimensions (Tuple[int, int]): The x and y dimensions of the Whole
        Slide Image, in pixels.

        project_id (str): The name of the BigQuery project to connect to.

        dataset_name (str, optional): The name of the BigQuery dataset to query.
        Defaults to "phi_main_us_p".

    Returns:
        None: (Nothing is returned, but data is uploaded into BigQuery)
    """

    # This uses a MERGE to only insert a record if the slide_uuid does not
    # already exist in the table. For examples of MERGE statements in BigQuery,
    # see https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_examples
    sql_statement = f"""
        MERGE `{project_id}.{dataset_name}.slides` as existing
        USING (
            SELECT 
                "{slide_uuid}" as slide_uuid,
                {int(slide_type)} as slide_type,
                {slide_microns_per_pixel} as microns_per_pixel,
                {pixels_to_meters(slide_dimensions[0], microns_per_pix=slide_microns_per_pixel)} as x_dimensions,
                {pixels_to_meters(slide_dimensions[1], microns_per_pix=slide_microns_per_pixel)} as y_dimension
            ) as new_slide
        ON existing.slide_uuid = new_slide.slide_uuid
        WHEN NOT MATCHED THEN
            INSERT ROW;
    """
    run_sql(
        project_id,
        sql_statement,
        description=f"inserting slide with uuid {slide_uuid}",
    )


def run_sql(project_id: str, sql_statement: str, description: str) -> List[str]:
    """
    Execute an SQL statement in BigQuery.

    This assumes that create_bigquery_client() has been used to update
    the global `bigquery_client` variable.

    Args:
        project_id (str): The name of the BigQuery project to query.

        sql_statement (str): The SQL statement to execute.

        description (str): A brief description of the goal of sql_statement.
        This is used to create the text of a RuntimeError if something goes
        wrong when executing sql_statement.

    Raises:
        RuntimeError: An error occured when executing sql_statement.

    Returns:
        list[str, ...]: A list of string outputs from executing sql_statement.
    """
    create_bigquery_client(project_id=project_id)

    logging.info(
        """Executing SQL statement:
    
%s

    """,
        sql_statement,
    )
    query_job = bigquery_client.query(sql_statement)
    try:
        output = query_job.result()
    except Exception as e:
        raise RuntimeError(f'Error {description}: "{e}"')

    return list(output)


def create_bigquery_tables(
    project_id: str,
    dataset_name: str = "phi_main_us_p",
    drop_tables_first: bool = False,
) -> List[str]:
    """
    Create tables (if they do not already exist) for data storage in BigQuery.
    This function will not overwrite existing tables.

    The table names within project_id.dataset_name will be "anomaly_polygons",
    "nuclear_polygons", and "slides".

    Note that BigQuery does not support transactions for table creation. Thus,
    errors in table creation cannot be rolled-back.

    Args:
        project_id (str): The name of the BigQuery project.

        dataset_name (str, optional): The name of the BigQuery dataset.
        Defaults to "phi_main_us_p".

        drop_tables_first (bool, optional): Whether to delete the tables before
        re-creating them. Setting this to True will require manual input in
        response to a confirmation prompt. Defaults to False.

    Returns:
        list[str, ...]: A list of string outputs from executing sql_statement.
    """
    # Following the BigQuery documentation at
    # https://cloud.google.com/bigquery/docs/information-schema-tables,
    # updates to the SQL statements below can be facilitated by running the
    # following within BigQuery's Cloud Console:
    #     SELECT
    #       table_name, ddl
    #     FROM
    #       {dataset_name}.INFORMATION_SCHEMA.TABLES;

    # Note, following the documentation at
    # https://cloud.google.com/bigquery/docs/geospatial-data#partitioning_and_clustering_geospatial_data,
    # that "If you store GEOGRAPHY data in a table and your queries filter data
    # by using a spatial predicate, ensure that the table is clustered by the
    # GEOGRAPHY column."

    # Note, per the documentation at
    # https://cloud.google.com/bigquery/quotas#partitioned_tables, that up to
    # 4000 partitions are allowed per table. We therefore use the approach
    # defined in utils.string_to_partition(), of hashing the slide UUID into
    # an int, and then taking the modulo of it and 4000 to get a stable
    # partition number.

    create_bigquery_client(project_id=project_id)

    if drop_tables_first is True:
        from random import randint

        tables = ["slides", "nuclear_polygons", "anomaly_polygons"]

        random_number = randint(0, 100000)
        drop_table_confirmation_prompt = f"delete tables from dataset {project_id}.{dataset_name}; random number: {random_number}"
        confirmation_prompt_full_text = f"""You are about to delete the following tables, including ALL DATA that they contain: {'; '.join(tables)}.

To confirm that you want to delete these tables and ALL DATA that they contain, please type the following, and then press Enter:

"{drop_table_confirmation_prompt}"
"""
        logging.warning("Respond to input prompt before proceeding.")
        logging.info('(Input prompt text: "%s"', drop_table_confirmation_prompt)

        confirmation_text = input(confirmation_prompt_full_text)

        if confirmation_text != drop_table_confirmation_prompt:
            sys.exit(1)

        logging.info(f'Dropping tables {", ".join(tables)}...')
        sql_drop_statement = "; ".join(
            [
                f"DROP TABLE IF EXISTS {project_id}.{dataset_name}.{table_name}"
                for table_name in tables
            ]
        )
        drop_table_output = run_sql(
            project_id,
            sql_drop_statement,
            description=f"dropping tables",
        )
        logging.info(
            'Output from dropping tables: "%s"',
            "; ".join(drop_table_output),
        )

    partition_number_explanation = "Created as an INT to allow partitioning, using `ABS(MOD(FARM_FINGERPRINT(slide_uuid),4000)))`. NOTE that this ID is not necessarily unique across slides."

    coordinate_system_explanation = (
        "(In this coordinate system, 1.0 units = 1.0 meters)"
    )

    sql_statement = f"""
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_name}.slides`
    (
        slide_uuid STRING NOT NULL OPTIONS(description="Slide UUID."),

        slide_type INT64 NOT NULL OPTIONS(description="Type / project of the slide. 0=Skin, 1=Colon; 2=Lung"),

        microns_per_pixel FLOAT64 OPTIONS(description="Number of microns per pixel. From the original slide image's metadata."),
        
        x_dimension FLOAT64 OPTIONS(description="Length of the X-axis of the slide. {coordinate_system_explanation}"),
        
        y_dimension FLOAT64 OPTIONS(description="Length of the Y-axis of the slide. {coordinate_system_explanation}")
    );

    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_name}.anomaly_polygons`
    (
        slide_uuid STRING NOT NULL OPTIONS(description="Slide UUID."),

        partition_number INT64 NOT NULL OPTIONS(description="{partition_number_explanation}"),

        segment GEOGRAPHY NOT NULL OPTIONS(description="Geometr(ies) / polygon(s) of the anomaly. {coordinate_system_explanation}")
    )
    PARTITION BY RANGE_BUCKET(partition_number, GENERATE_ARRAY(0, 4000, 1))
    CLUSTER BY segment, slide_uuid;

    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_name}.nuclear_polygons`
    (
        slide_uuid STRING NOT NULL OPTIONS(description="Slide UUID."),

        partition_number INT64 NOT NULL OPTIONS(description="{partition_number_explanation}"),

        centroid GEOGRAPHY OPTIONS(description="Centroid of the nucleus. {coordinate_system_explanation}"),

        diameter FLOAT64 OPTIONS(description="The diameter, in meters, of the nucleus. (Specifically, the length of the largest dimension of the minimum rotated rectangle that can be drawn to encompass the nucleus.) {coordinate_system_explanation}"),

        is_tumor BOOL OPTIONS(description="Whether, based on the size of the nucleus, we suspect that the nucleus is tumorous.")
    )
    PARTITION BY RANGE_BUCKET(partition_number, GENERATE_ARRAY(0, 4000, 1))
    CLUSTER BY centroid, slide_uuid, is_tumor;
    """

    output = run_sql(project_id, sql_statement, description="creating tables")
    return output
