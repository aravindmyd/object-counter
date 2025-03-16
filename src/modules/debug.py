import os

from PIL import ImageDraw, ImageFont


def draw(predictions, image, image_name):
    draw_image = ImageDraw.Draw(image, "RGBA")
    image_width, image_height = image.size

    # Try to load a default font if a system font is unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Use system font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default PIL font

    for prediction in predictions:
        box = prediction.box
        draw_image.rectangle(
            [
                (box.xmin * image_width, box.ymin * image_height),
                (box.xmax * image_width, box.ymax * image_height),
            ],
            outline="red",
        )
        class_name = prediction.class_name
        text_position = (box.xmin * image_width, box.ymin * image_height - 10)

        draw_image.text(
            text_position,
            f"{class_name}: {prediction.score:.2f}",
            font=font,
            fill="black",
        )

    os.makedirs("tmp/debug", exist_ok=True)
    image.save(f"tmp/debug/{image_name}", "JPEG")
