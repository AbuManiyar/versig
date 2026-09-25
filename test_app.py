from app import preprocess_image, verify_signatures
from PIL import Image


def test_preprocess_image():
    image = Image.new("RGB", (220, 155), color="white")

    tensor = preprocess_image(image)

    assert tensor.shape == (1, 1, 155, 220)
