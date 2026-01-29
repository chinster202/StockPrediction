# Upload all .pkl files in the models/ directory to S3
import os
import boto3
from botocore.exceptions import NoCredentialsError
#from dotenv import load_dotenv
#load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)
import glob
import sys
import traceback

def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3',
                             aws_access_key_id="AKIAWKPMSDWP7JGING2F",  #os.getenv('AWS_ACCESS_KEY_ID'),
                             aws_secret_access_key="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")  #os.getenv('AWS_SECRET_ACCESS_KEY'))
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        logging.info(f"Upload Successful: {file_name} to {bucket}/{object_name}")
        return True
    except FileNotFoundError:
        logging.error(f"The file {file_name} was not found.")
        return False
    except NoCredentialsError:
        logging.error("Credentials not available.")
        return False
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return False
    
if __name__ == "__main__":
    bucket_name = 'my-stock-model-data'  # Replace with your S3 bucket name
    model_files = glob.glob('models/*.pkl')

    for model_file in model_files:
        upload_to_s3(model_file, bucket_name)