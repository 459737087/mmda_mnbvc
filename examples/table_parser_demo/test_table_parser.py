import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(parent_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)
import base64
from pprint import pprint
from mmda.predictors.paddle_predictors.table_predictor import TablePredictor

def image_to_base64(image_path: str) -> str:
    """
    Convert an image to its base64 representation.
    Parameters:
        image_path (str): The path to the image file.
    Returns:
        str: The base64 representation of the image.
    """
    with open(image_path, "rb") as image_file:
        # Convert the binary data to base64 encoded string
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

table_predictor = TablePredictor()
img_path = 'data/test/table_en.png'
img = image_to_base64(img_path)
res  = table_predictor.get_html_result(img)
pprint(res)