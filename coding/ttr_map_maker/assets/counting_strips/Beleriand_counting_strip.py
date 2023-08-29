from counting_strip_generator import generate_counting_strip


# Beleriand counting strip
strip_minmax = [
    (1, 35),
    (36, 75),
    (76, 110),
    (111, 150)]
image_size = [11975, 8445]
for (min_number, max_number), strip_length_px in zip(strip_minmax, image_size*2):
    
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
        number_heights=[0.4, 0.5, 0.5],
        number_offset=(0., 0.1))
    strip_image.save(f"assets\\beleriand_counting_{min_number}_{max_number}.png")