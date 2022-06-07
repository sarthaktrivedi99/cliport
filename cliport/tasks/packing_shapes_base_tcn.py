"""Packing Shapes task."""

import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class PackingShapes_Base_TCN(Task):
    """Packing Shapes base class for TCN."""

    def __init__(self):
        super().__init__()
        # self.ee = 'suction'
        self.max_steps = 1
        # self.metric = 'pose'
        # self.primitive = 'pick_place'
        self.train_set = np.arange(3, 4)
        self.test_set = np.arange(3,4)
        self.homogeneous = False

        self.lang_template = "pack the {obj} in the brown box"
        self.task_completed_desc = "done packing shapes."

    def reset(self, env):
        super().reset(env)

        # Shape Names:
        shapes = {
            3: "square",
        }

        n_objects = 1
        if self.mode == 'train':
            obj_shapes = np.random.choice(self.train_set, n_objects, replace=False)
        else:
            if self.homogeneous:
                obj_shapes = [np.random.choice(self.test_set, replace=False)] * n_objects
            else:
                obj_shapes = np.random.choice(self.test_set, n_objects, replace=False)

        # Shuffle colors to avoid always picking an object of the same color
        color_names = self.get_colors()
        colors = [utils.COLORS[cn] for cn in color_names]
        np.random.shuffle(colors)

        # Add container box.
        zone_size = self.get_random_size(0.1, 0.15, 0.1, 0.15, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf):
            os.remove(container_urdf)

        # Add objects.
        objects = []
        template = 'kitting/object-template.urdf'
        object_points = {}
        for i in range(n_objects):
            shape = obj_shapes[i]
            size = (0.08, 0.08, 0.02)
            pose= self.get_random_pose(env, size)
            fname = f'{shape:02d}.obj'
            fname = os.path.join(self.assets_root, 'kitting', fname)
            scale = [0.003, 0.003, 0.001]  # .0005
            replace = {'FNAME': (fname,),
                       'SCALE': scale,
                       'COLOR': colors[i]}
            urdf = self.fill_template(template, replace)
            block_id = env.add_object(urdf, pose)
            if os.path.exists(urdf):
                os.remove(urdf)
            object_points[block_id] = self.get_box_object_points(block_id)
            objects.append((block_id, (0, None)))

        # Pick the first shape.
        num_objects_to_pick = 1
        for i in range(num_objects_to_pick):
            obj_pts = dict()
            obj_pts[objects[i][0]] = object_points[objects[i][0]]

            self.goals.append(([objects[i]], np.int32([[1]]), [zone_pose],
                               False, True, 'zone',
                               (obj_pts, [(zone_pose, zone_size)]),
                               1 / num_objects_to_pick))
            self.lang_goals.append(self.lang_template.format(obj=shapes[obj_shapes[i]]))

    def get_colors(self):
        # return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
        return ['blue']