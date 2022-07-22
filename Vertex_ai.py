https://chrisalbon.com/code/machine_learning/preprocessing_text/bag_of_words/

# Load library
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#https://towardsdatascience.com/adapting-to-changes-of-data-by-building-mlops-pipeline-in-vertex-ai-3f8ebd19a869

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.dsl import pipeline
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient

PROJECT_ID = “YOUR_GCP_PROJECT_ID”
REGION = “GCP_REGION_TO_RUN_PIPELINE”
PIPELINE_ROOT = “LOCATION_RUN_METADATA_IS_GOING_TO_BE_STORED”
DATASET_META_PATH = “LOCATION_DATASET_METADATA_IS_STORED”

@pipeline(name=”my-pipeline”)
def pipeline(project: str = PROJECT_ID):
   ds_op = gcc_aip.ImageDatasetCreateOp(
      project=project,
      display_name=”DATASET_NAME_TO_APPEAR”,
      gcs_source=DATASET_META_PATH,
      import_schema_uri=\
         aiplatform.schema.dataset.ioformat.image.bounding_box,
   )
   training_job_run_op = gcc_aip.AutoMLImageTrainingJobRunOp(
      project=project,
      display_name=”my-daughter-od-training”,
      prediction_type=”object_detection”,
      model_type=”CLOUD”,
      base_model=None,
      dataset=ds_op.outputs[“dataset”],
      model_display_name=”my-daughter-od-model”,
      training_fraction_split=0.6,
      validation_fraction_split=0.2,
      test_fraction_split=0.2,
      budget_milli_node_hours=20000,
   )
   endpoint_op = gcc_aip.ModelDeployOp(
      project=project, model=training_job_run_op.outputs[“model”]
   )

compiler.Compiler().compile(
   pipeline_func=pipeline, package_path=PIPELINE_SPEC_PATH
)
api_client = AIPlatformClient(project_id=PROJECT_ID, region=REGION)
response = api_client.create_run_from_job_spec(
 PIPELINE_SPEC_PATH,
 pipeline_root=PIPELINE_ROOT,
 parameter_values={“project”: PROJECT_ID},
)

from kfp.v2.google.client import AIPlatformClient
PROJECT_ID = “YOUR_GCP_PROJECT_ID”
REGION = “GCP_REGION_TO_RUN_PIPELINE”
PIPELINE_ROOT = “LOCATION_RUN_METADATA_IS_GOING_TO_BE_STORED”
PIPELINE_SPEC_PATH = “LOCATION_PIPELINE_SPEC_IS_STORED”
def vertex_ai_pipeline_trigger(event, context):
   print(‘File: {}’.format(event[‘name’]))
   print(‘Extension: {}’.format(event[‘name’].split(“.”)[-1]))
   if event[‘name’].split(“.”)[-1] == “jsonl”:
      print(“target file extension”)
      api_client = AIPlatformClient(
         project_id=PROJECT_ID,
         region=REGION
      )
      print(“api_client is successfully instantiated”)
      response = api_client.create_run_from_job_spec(
         PIPELINE_SPEC_PATH,
         pipeline_root=PIPELINE_ROOT,
         parameter_values={“project”: PROJECT_ID},
      )
      print(‘response: {}’.format(response))

gclcloud functions deploy YOUR_FUNCTION_NAME \
 — trigger-resource YOUR_TRIGGER_BUCKET_NAME \
 — trigger-event providers/cloud.storage/eventTypes/object.finzlize

