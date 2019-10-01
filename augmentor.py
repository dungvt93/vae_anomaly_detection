import Augmentor
p = Augmentor.Pipeline("./train_origin/unknown")
p.rotate(probability=0.5,max_left_rotation=10,max_right_rotation=10)
p.zoom(probability=0.5,min_factor=1.1,max_factor=1.5)
p.skew_corner(probability=0.5)
p.random_distortion(probability=0.5,grid_width=4, grid_height=4, magnitude=1)
p.rotate_random_90(probability=0.5)
p.shear(probability=0.5,max_shear_left=10,max_shear_right=10)
p.flip_random(probability=0.5)
p.sample(5000)