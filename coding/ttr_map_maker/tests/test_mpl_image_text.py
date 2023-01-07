from PIL import Image, ImageDraw, ImageFont

def render_text(text, font_size, font_file, text_color):
    # Load the font
    font = ImageFont.truetype(font_file, font_size)
    # Get the size of the text
    text_size = font.getsize(text)
    # Create an image with a transparent background
    image = Image.new("RGBA", text_size, (0, 0, 0, 0))
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    # Draw the text onto the image
    draw.text((0, 0), text, font=font, fill=text_color)
    return image

def draw_text_on_plot(ax, x, y, text, font_size, font_file, text_color):
    # Render the text as an image
    image = render_text(text, font_size, font_file, text_color)
    # Draw the image onto the plot using imshow
    ax.imshow(image, extent=(x, x + image.size[0], y - image.size[1], y))

# Example usage

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Draw some text on the plot
draw_text_on_plot(ax, 0, 0, "Hello, World!", 24, "arial.ttf", (255, 0, 0))

plt.show()
