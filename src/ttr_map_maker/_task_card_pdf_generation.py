from typing import List, Tuple
import os
import tkinter as tk
from tkinter import filedialog

from pylatex import Document, NoEscape, MiniPage, Command, Package

def add_horizontal_crop_marks(
        doc: Document,
        x_pos: float = 0.0,
        y_pos: float = 0.0,
        content_height: float = 65.0,
        content_width: float = 91.0,
        crop_mark_length: float = 5.0):
    """
    Adds horizontal crop marks for one card to the document.

    Args:
        doc (Document): The LaTeX document.
        crop_mark_length (float): Length of the crop marks in mm.
    """
    # Calculate positions for horizontal crop marks (top and bottom of the card)
    top_y_pos = y_pos
    bottom_y_pos = y_pos + content_height
    left_x_pos = x_pos - 1
    right_x_pos = x_pos + content_width + 1
    # begin tikzpicture
    doc.append(NoEscape(r'\begin{tikzpicture}[overlay, remember picture]'))
    # Top left horizontal crop mark
    doc.append(NoEscape(f"\\draw ([xshift={left_x_pos}mm,yshift=-{top_y_pos}mm]current page.north west) -- ++(-{crop_mark_length}mm,0); % top left horizontal"))
    # Top right horizontal crop mark
    doc.append(NoEscape(f"\\draw ([xshift={right_x_pos}mm,yshift=-{top_y_pos}mm]current page.north west) -- ++({crop_mark_length}mm,0); % top right horizontal"))
    # Bottom left horizontal crop mark
    doc.append(NoEscape(f"\\draw ([xshift={left_x_pos}mm,yshift=-{bottom_y_pos}mm]current page.north west) -- ++(-{crop_mark_length}mm,0); % bottom left horizontal"))
    # Bottom right horizontal crop mark
    doc.append(NoEscape(f"\\draw ([xshift={right_x_pos}mm,yshift=-{bottom_y_pos}mm]current page.north west) -- ++({crop_mark_length}mm,0); % bottom right horizontal"))
    # end tikzpicture
    doc.append(NoEscape(r'\end{tikzpicture}'))

def add_vertical_crop_marks(
        doc: Document,
        x_pos: float = 0.0,
        y_pos: float = 0.0,
        content_height: float = 65.0,
        content_width: float = 91.0,
        crop_mark_length: float = 5.0):
    """
    Adds vertical crop marks for one page of cards to the document.
    
    Args:
        doc (Document): The LaTeX document.
        x_pos (float): x position of the top left corner of the top card in mm.
        y_pos (float): y coordinate of the top left corner of the top card in mm.
        content_height (float): Height of the combined cards in mm.
        content_width (float): Width of the combined cards in mm.
        crop_mark_length (float): Length of the crop marks in mm.
    """
    # Calculate positions for vertical crop marks (left and right of the card)
    top_y_pos = y_pos - 1
    bottom_y_pos = y_pos + content_height + 1
    left_x_pos = x_pos
    right_x_pos = x_pos + content_width
    # begin tikzpicture
    doc.append(NoEscape(r'\begin{tikzpicture}[overlay, remember picture]'))
    # Top left vertical crop mark
    doc.append(NoEscape(f"\\draw ([xshift={left_x_pos}mm,yshift=-{top_y_pos}mm]current page.north west) -- ++(0,{crop_mark_length}mm); % top left vertical"))
    # Top right vertical crop mark
    doc.append(NoEscape(f"\\draw ([xshift={right_x_pos}mm,yshift=-{top_y_pos}mm]current page.north west) -- ++(0,{crop_mark_length}mm); % top right vertical"))
    # Bottom left vertical crop mark
    doc.append(NoEscape(f"\\draw ([xshift={left_x_pos}mm,yshift=-{bottom_y_pos}mm]current page.north west) -- ++(0,-{crop_mark_length}mm); % bottom left vertical"))
    # Bottom right vertical crop mark
    doc.append(NoEscape(f"\\draw ([xshift={right_x_pos}mm,yshift=-{bottom_y_pos}mm]current page.north west) -- ++(0,-{crop_mark_length}mm); % bottom right vertical"))
    # end tikzpicture
    doc.append(NoEscape(r'\end{tikzpicture}'))

def add_card(
        doc: Document,
        x_pos: float,
        y_pos: float,
        front_path: str,
        back_path: str,
        flip_backside: bool = False,
        card_width: float = 88.9,
        card_height: float = 63.5,
        gap: float = 0.1,
        vspace: float = 2.0):
    """
    Adds a single card (front and back) to the document.

    Args:
        doc (Document): The LaTeX document.
        front_path (str): File path to the front image.
        back_path (str): File path to the back image.
        flip_backside (bool): Whether to flip the back image.
        card_width (float): Width of the card in mm.
        card_height (float): Height of the card in mm.
        gap (float): Gap between front and back in mm.
        vspace (float): Vertical space between cards in mm.
    """# Start tikzpicture for precise image placement
    doc.append(NoEscape(r'\begin{tikzpicture}[overlay, remember picture]'))

    # Adjust shifts to correct the positioning for the back image
    x_shift_back = x_pos + card_width / 2
    y_shift_back = y_pos + card_height / 2

    if flip_backside:
        doc.append(NoEscape(
            f'\\node[anchor=center,rotate=180] at ([xshift={x_shift_back}mm,yshift=-{y_shift_back}mm]current page.north west) '
            f'{{\\includegraphics[height={card_height}mm,width={card_width}mm,keepaspectratio=FALSE]{{{back_path}}}}};'
        ))
    else:
        doc.append(NoEscape(
            f'\\node[anchor=center] at ([xshift={x_shift_back}mm,yshift=-{y_shift_back}mm]current page.north west) '
            f'{{\\includegraphics[height={card_height}mm,width={card_width}mm,keepaspectratio=FALSE]{{{back_path}}}}};'
        ))

    # Adjust shifts to correct the positioning for the front image
    x_shift_front = x_pos + card_width + gap + card_width / 2
    y_shift_front = y_pos + card_height / 2

    doc.append(NoEscape(
        f'\\node[anchor=center] at ([xshift={x_shift_front}mm,yshift=-{y_shift_front}mm]current page.north west) '
        f'{{\\includegraphics[height={card_height}mm,width={card_width}mm,keepaspectratio=FALSE]{{{front_path}}}}};'
    ))

    # End tikzpicture
    doc.append(NoEscape(r'\end{tikzpicture}'))

    # Vertical space between cards
    doc.append(NoEscape(f'\\vspace{{{vspace}mm}}'))

def generate_latex_document(
        fronts: List[str], 
        back: str, 
        flip_backside: bool = True,
        target_filepath: str = "debug_graph/task_cards/task_cards",
        card_width: float = 91,
        card_height: float = 65,
        border: Tuple[float, float] = (10, 10),
        vspace: float = 2.0,
        gap: float = 0.1,
        crop_mark_length: float = 20) -> Tuple[Document, str]:
    """
    Generates a LaTeX document for printing custom cards.

    Args:
        fronts (List[str]): List of file paths for the front images.
        back (str): File path for the back image.
        flip_backside (bool): Whether to flip the back image.
        border (Tuple[float, float]): Vertical and horizontal borders in cm.
        vspace (float): Vertical space between cards in mm.

    Returns:
        Tuple[Document, str]: The generated document object and the filepath where it was saved.
    """

    geometry_options = {
        "margin": f"{border[0]}mm"
    }
    doc = Document(documentclass='article', document_options='a4paper', geometry_options=geometry_options)
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('tikz'))
    doc.append(NoEscape(r'\pagestyle{empty}'))

    y_pos = border[0]
    x_pos = border[1]
    for i, front in enumerate(fronts):
        doc.append(NoEscape(r'\noindent'))
        add_card(
            doc,
            x_pos=x_pos,
            y_pos=y_pos,
            front_path=front,
            back_path=back,
            flip_backside=flip_backside,
            card_width=card_width,
            card_height=card_height,
            gap=gap,
            vspace=vspace)
        add_horizontal_crop_marks(
            doc,
            x_pos=x_pos,
            y_pos=y_pos,
            content_height=card_height,
            content_width=2*card_width + gap,
            crop_mark_length=crop_mark_length)
        y_pos += card_height
        if not i == len(fronts) - 1:
            y_pos += vspace
        if i % 4 != 3: # add linebreak between cards
            doc.append(NoEscape(r'\\'))
        if i % 4 == 3 or i == len(fronts) - 1:
            add_vertical_crop_marks(
                doc,
                x_pos=x_pos,
                y_pos=border[0],
                content_height=y_pos - border[0],
                content_width=2*card_width + gap,
                crop_mark_length=crop_mark_length)
            if i != len(fronts) - 1:
                doc.append(NoEscape(r'\newpage'))
                y_pos = border[0]
    # Output as .tex file
    target_filepath_stripped = target_filepath.strip(".tex")
    doc.generate_tex(target_filepath_stripped)
    return doc, target_filepath

def task_cards_to_latex(
        front_images: list[str] = None,
        back_image: str = None,
        flip_backside: bool = False,
        card_height: float = 65.0,
        card_width: float = 91.0,
        target_filepath: str = None):
    """


    Args:
        front_images (list[str]): 
        back_image (str): 
        card_height (float, optional): 
        card_width (float, optional): 
        flip_backside (bool, optional): 
    """
    tk.Tk().withdraw()
    # file picker to choose folder with front images
    front_images_path = filedialog.askdirectory(title="Choose folder with front images")
    front_images = os.listdir(front_images_path)
    # file picker to choose back image
    back_image = filedialog.askopenfilename(title="Choose back image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    # remove back image from front images
    try:
        # remove basename of back image from front images
        front_images.remove(os.path.basename(back_image))
    except ValueError:
        pass
    # remove non-image files from front images
    # front_images = [f"{front_images_path}/{img}" for img in front_images if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")]
    front_images = [img for img in front_images if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")]
    if target_filepath is None: # save 
        target_filepath = os.path.join(front_images_path, "task_cards")

    doc, filepath = generate_latex_document(
        fronts=front_images,
        back=back_image,
        flip_backside=flip_backside,
        target_filepath=target_filepath,
        card_width=card_width,
        card_height=card_height,
    )
    print(f"Generated LaTeX document at {filepath}.tex")
    # Compile to pdf
    filepath = filepath.strip(".tex")
    doc.generate_pdf(
        filepath,
        compiler="pdflatex",
        clean_tex=False)
    print(f"Generated PDF at {filepath}.pdf")


if __name__ == "__main__":
    # Example usage
    task_cards_to_latex(flip_backside=False)
    # front_images = ["img1.png", "img1.png", "img1.png", "img1.png", "img1.png"]
    # back_image = "back.png"
    # generate_latex_document(front_images, back_image, flip_backside=False)
