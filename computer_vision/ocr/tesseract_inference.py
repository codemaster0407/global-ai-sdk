from PIL import Image
import pytesseract


def ocr_inference(image_path):
# Load your image
    image = Image.open(image_path)

    # Extract text
    text = pytesseract.image_to_string(image)

    return text 