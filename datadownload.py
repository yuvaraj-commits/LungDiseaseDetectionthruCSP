import requests
import os
import requests, zipfile, io
import shutil

def dataset_download(zip_file_url='https://storage.googleapis.com/kaggle-data-sets/627146/1117472/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210327%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210327T183327Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=6c2859dadf47066feacb7e99c2e73d1a8fe6e78d73e2958e2ac7834c9bc78b5f4a8228c2350d8b1468c8dc40f295d0a9f9018b6a915d33e18912173e94509cfdafaeb36fd438ac11c9932defa37dff58b441e4d5a395edc275c7a7e82beabee06032cb163ecdd36d79845b8e6a1105108ebc7e186bdce2334394b2631e98d1ee579bfd4ba346d9c9657e668d4e8b307cb2bcf10d84884baabe65edd9d517179eb3be990836fdd4573baf3489d88e903b112554efc054da63100692a896aadf5f685e04a15cb435b583be0273877071abe9d0860f8284207b868a1a7bb8ce830689bcc2c488ab33c295e25fada3b16cb0f2d471fe546b8b61f6919e7fdae30659', folder_name='kaggle_dataset'):
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    download_path = os.path.join(os.getcwd(), folder_name)
    if os.path.exists(download_path):
        shutil.rmtree(download_path)
    os.makedirs(download_path)
    z.extractall(download_path)
    print("data download to ", download_path)
    
zip_file_url = 'https://storage.googleapis.com/kaggle-data-sets/627146/1117472/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210327%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210327T183327Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=6c2859dadf47066feacb7e99c2e73d1a8fe6e78d73e2958e2ac7834c9bc78b5f4a8228c2350d8b1468c8dc40f295d0a9f9018b6a915d33e18912173e94509cfdafaeb36fd438ac11c9932defa37dff58b441e4d5a395edc275c7a7e82beabee06032cb163ecdd36d79845b8e6a1105108ebc7e186bdce2334394b2631e98d1ee579bfd4ba346d9c9657e668d4e8b307cb2bcf10d84884baabe65edd9d517179eb3be990836fdd4573baf3489d88e903b112554efc054da63100692a896aadf5f685e04a15cb435b583be0273877071abe9d0860f8284207b868a1a7bb8ce830689bcc2c488ab33c295e25fada3b16cb0f2d471fe546b8b61f6919e7fdae30659'#'https://www.kaggle.com/pranavraikokte/covid19-image-dataset/download'
folder_name = 'kaggle_dataset'

# dataset_download(zip_file_url, folder_name)
