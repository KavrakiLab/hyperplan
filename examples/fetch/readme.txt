See the corresponding images of each dataset for a visualization of start and goal. 


pathxxxx.yaml is a valid trajectory
requestxxxx.yaml contains the start and goal configurations in joint_space
scenexxxx.yaml contains the geometric description of the obstacles 
scene_sensedxxxx.yaml contains the octomap representation of the obstacles 

The box_pick and table_under_pick contain problems that are in "open space" so I would assume similar
parameters work best for them 

The tall_pick, small_pick, thin_vert_pick are bookcases that are narrow

tall_place and thin_vert_place are similar to the "pick" versions but instead of the stow position the robot
starts inside a random shelf

The thin_vert_place is the hardest one of all so far "especially the octomap version" 
