import sys
import typing
import google.cloud.compute_v1 as compute_v1

# first you have to run the bash command :
# gcloud auth application-default login
# a json file will be created with credentials
# Credentials saved to file: [/home/heba/.config/gcloud/application_default_credentials.json]


def stop_instance(project_id, zone, machine_name):
    instance_client = compute_v1.InstancesClient()
    operation_client = compute_v1.ZoneOperationsClient()

    print(f"Stopping {machine_name} from {zone}...")
    operation = instance_client.stop(
        project=project_id, zone=zone, instance=machine_name
    )
    while operation.status != compute_v1.Operation.Status.DONE:
        operation = operation_client.wait(
            operation=operation.name, zone=zone, project=project_id
        )
    if operation.error:
        print("Error during stop:", operation.error, file=sys.stderr)
    if operation.warnings:
        print("Warning during stop:", operation.warnings, file=sys.stderr)
    print(f"Instance {machine_name} stopped.")
    return


stop_instance('kayanhr-staging', 'europe-west1-b', 'kayanhr-staging-ai-workload-1')


# """
# BEFORE RUNNING:
# ---------------
# 1. If not already done, enable the Compute Engine API
#    and check the quota for your project at
#    https://console.developers.google.com/apis/api/compute
# 2. This sample uses Application Default Credentials for authentication.
#    If not already done, install the gcloud CLI from
#    https://cloud.google.com/sdk and run
#    `gcloud beta auth application-default login`.
#    For more information, see
#    https://developers.google.com/identity/protocols/application-default-credentials
# 3. Install the Python client library for Google APIs by running
#    `pip install --upgrade google-api-python-client`
# """
# from pprint import pprint
#
# from googleapiclient import discovery
# from oauth2client.client import GoogleCredentials
#
# credentials = GoogleCredentials.get_application_default()
#
# service = discovery.build('compute', 'v1', credentials=credentials)
#
# # Project ID for this request.
# project = 'my-project'  # TODO: Update placeholder value.
#
# # The name of the zone for this request.
# zone = 'my-zone'  # TODO: Update placeholder value.
#
# # Name of the instance resource to stop.
# instance = 'my-instance'  # TODO: Update placeholder value.
#
# request = service.instances().stop(project=project, zone=zone, instance=instance)
# response = request.execute()
#
# # TODO: Change code below to process the `response` dict:
# pprint(response)