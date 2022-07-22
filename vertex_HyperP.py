# https://cloud.google.com/blog/topics/developers-practitioners/distributed-training-and-hyperparameter-tuning-tensorflow-vertex-ai?utm_content=buffer48771&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer

# https://cloud.google.com/sdk/gcloud/reference

pip install cloudml-hypertune

PROJECT_ID = "AIIIIIIIIIIIIIIIIIIIIIIIII"

# Get your Google Cloud project ID from gcloud
if not os.getenv("IS_LOVETESTING"):
    shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
    PROJECT_ID = shell_output[0]
    print("Project ID: ", PROJECT_ID)

