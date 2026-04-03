from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

subscription_id = "24cf5cf7-052a-4c4b-8284-885db4abe7c5"
resource_group = "ml-resource-group"
workspace = "ml-workspace"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace
)

print("Connected to Azure ML")