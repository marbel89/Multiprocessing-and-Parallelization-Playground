import cv2  # OpenCV library for image processing
import pytesseract  # Library for Optical Character Recognition (OCR)
import pathlib  # Library for handling filesystem paths
import time  # Library for tracking time
from multiprocessing import Pool  # Library for parallel processing

# Specify the path to the Tesseract OCR executable. Generally (py)tesseract works a bit wonky,
# but this way seems to work reliable.
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Define the path to the dataset folder
path = pathlib.Path("large-receipt-image-dataset-SRD")
# Recursively find all files in the dataset folder
files = list(path.rglob("*"))
# Filter the files to include only JPG images
files = [str(file) for file in files if ".jpg" in file.name]


def scrape_text(file):
    """
    Extract text from an image file using Tesseract OCR.

    Parameters
    ----------
    file : str
        The path to the image file.

    Returns
    -------
    str
        The text extracted from the image.
    """
    # Read the image file
    image = cv2.imread(file)
    # Use Tesseract to convert the image to a string
    return pytesseract.image_to_string(image)


# For Windows environments, and even if there's no windows environment, for compatibility reasons
if __name__ == "__main__":

    start = time.time()
    # Create a pool of 8 worker processes for parallel processing
    pool_count = 8
    cores_pool = Pool(pool_count)
    # Use the pool to apply the scrape_text function to all image files
    results = cores_pool.map(scrape_text, files)
    # Close the pool to free up resources
    cores_pool.close()
    end = time.time()
    print(f"Time taken with {pool_count} workers: {end - start}")

    for file, text in zip(files, results):
        print(f"Text from {file}:\n{text}\n")
