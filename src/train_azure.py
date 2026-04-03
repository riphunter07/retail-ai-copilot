from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
   "24cf5cf7-052a-4c4b-8284-885db4abe7c5",
    "ml-resource-group",
    "ml-workspace"
)

job = command(
    code="./src",
    command="python train.py --data ${{inputs.data}}",
    inputs={"data": Input(type="uri_file", path="azureml:sales-data:1")},
    environment="azureml://registries/azureml/environments/sklearn-1.0/labels/latest",
    compute="cpu-cluster",
    display_name="retail-training-job"
)

ml_client.jobs.create_or_update(job)

print("Job submitted!")