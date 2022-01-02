from DataStore import DataStore
from azure.storage.blob import BlobServiceClient

# class to read data from csv stored in azure blobs
class AzureBlobDataStore(DataStore):
    #enter connection string
    CONNECTIONSTRING = ''
    CONTAINERNAME = 'datasets'
    FILENAME = 'heart.csv'

    def readData(self):
        blob_service_client = BlobServiceClient.from_connection_string(self.CONNECTIONSTRING)
        container_client = blob_service_client.get_container_client(self.CONTAINERNAME)
        download_stream = container_client.download_blob(self.FILENAME)
        data = download_stream.readall()
        return data