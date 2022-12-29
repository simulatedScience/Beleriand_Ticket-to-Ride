from PIL import Image, ImageDraw, ImageFont
import os

# Load the "Middle-earth" font
filepath = os.path.join(os.path.dirname(__file__), "beleriand_ttr", "MiddleEarth.ttf")
with open(filepath, "r") as file:
    print("file read")
font = ImageFont.truetype(filepath, size=24)

# Get the size of the text
text = "Hello, World!"
text_width, text_height = font.getsize(text)

print(f"Text width: {text_width}")
print(f"Text height: {text_height}")
