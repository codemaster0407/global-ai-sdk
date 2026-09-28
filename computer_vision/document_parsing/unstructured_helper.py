# One-time setup:
# pip install unstructured-transform-client
# Replace the placeholder key below.

from unstructured_transform_client import TransformClient
from dotenv import load_dotenv 
import os 
from unstructured.partition.pdf import partition_pdf

load_dotenv()

client = TransformClient(
    api_key=os.getenv('UNSTRUCTURED_API_KEY'),
    server_url="https://transform.unstructured.io",
)



def cloud_pdf_reader(filepath: str):
    with open(filepath, "rb") as f:
        result = client.parse.run(input=f)

    return result.markdown


def local_pdf_reader(filepath : str, verbose : bool = True):
    elements = partition_pdf(filename = filepath)

    if verbose:
        for i, element in enumerate(elements):
            text = (element.text or "").strip()
            if not text:
                continue
            print(f"[{i}] {element.category}: {text}")

    return elements

