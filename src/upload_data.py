from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


ml_client = MLClient(
    DefaultAzureCredential(),
    "24cf5cf7-052a-4c4b-8284-885db4abe7c5",
    "ml-resource-group",
    "ml-workspace"
)

data = Data(
    path="data/sales.csv",
    type=AssetTypes.URI_FILE,
    name="sales-data",
    description="Retail sales dataset"
)

ml_client.data.create_or_update(data)

print("Data uploaded to Azure")