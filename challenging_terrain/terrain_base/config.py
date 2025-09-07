class terrain_config:
        mesh_type = "terrain" # [NOTE] need to finetune later.  "terrain"
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.2, 0.2]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 
        height = [0.02, 0.06]   # roughness height
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        if mesh_type != "None":
                curriculum = True
        else:   
                curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True # 生成采样的高度点
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        max_init_terrain_level = 0 # starting curriculum state
        terrain_length = 10 # [NOTE] more steps scenario 20 #10.
        terrain_width = 4.
        platform_size = 2.5
        num_rows= 10 #20 #20 #10 # [NOTE] number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 20 #10 #10 #20 # [NOTE] number of terrain cols (types)

        num_goals = 10 # [NOTE] more steps scenario 20 #10

        dataset_points_x = [-1.5, -1.35, -1.2, -1.05, -0.9, -0.75, -0.6, -0.45,
                            -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2,
                            1.35, 1.5, 1.65, 1.8, 1.95, 2.1, 2.25, 2.4, 2.55, 2.7, 2.85, 3.0, 3.15, 3.3, 3.45, 3.6]
        dataset_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]

        slope_treshold = 1.5 # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True