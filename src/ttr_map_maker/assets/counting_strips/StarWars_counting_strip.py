from counting_strip_generator import generate_counting_strip


# Beleriand counting strip
strip_minmax = [
    (1, 25),
    (26, 50),
]
image_size = [9000, 6767]
strip_lengths = [image_size[1], image_size[1]]
for i, ((min_number, max_number), strip_length_px) in enumerate(zip(strip_minmax, strip_lengths)):
    
    strip_image = generate_counting_strip(
        min_number=min_number,
        max_number=max_number,
        length_px=strip_length_px,
        # cell_images=[
        #     "assets\\counting_strips\\2d_oak_base.png",
        #     "assets\\counting_strips\\2d_palantir.png",
        #     "assets\\counting_strips\\2d_palantir_on_oak.png"],
        cell_images=[None],
        # cell_images=[
        #     "assets\\counting_strips\\2d_runestone.png",
        #     "assets\\counting_strips\\2d_palantir.png"],
        # number_folders=["assets\\points_images\\points_standard"],
        number_folders=["../../projects/Star_Wars_TTR/points_images/counting_strip_numbers_bg"],
        # number_heights=[0.4, 0.5, 0.5],
        number_heights=[1, 1, 1],
        # number_heights=[0.5, 0.5, 0.5],
        # number_offset=(0., 0.1),
        number_offset=(0., 0.),
        number_rotation=0,
        empty_first_cell=False,
        save_path_prefix=f"../../projects/Star_Wars_TTR/board design/Star_Wars_")