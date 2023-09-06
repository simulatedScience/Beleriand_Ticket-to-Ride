from counting_strip_generator import generate_counting_strip


# Beleriand counting strip
strip_minmax = [
    (1, 45),
    (46, 75),
    (76, 120),
    (121, 150)]
# image_size = [11975, 8445] # for large map (including Angband), no empty_first_cell
image_size = [10887, 7766] # for small map (excluding Angband), empty_first_cell
for i, ((min_number, max_number), strip_length_px) in enumerate(zip(strip_minmax, image_size*2)):
    
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
        number_folders=["assets\\points_images\\points_standard"],
        # number_heights=[0.4, 0.5, 0.5],
        number_heights=[0.5, 0.5, 0.5],
        number_offset=(0., 0.1),
        number_rotation=45 * (-1)**i,
        empty_first_cell=True,
        save_path_prefix=f"assets\\counting_strips\\beleriand_counting_strips\\beleriand_")