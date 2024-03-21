import numpy as np
def generate_path(c2w, focal, sc, length=None):
    # hwf = c2w[:, 4:5]
    num_novelviews = 60
    max_disp = 48.0
    # H, W, focal = hwf[:, 0]
    # downsample = 2.0
    # focal = (854 / 2 * np.sqrt(3)) / float(downsample)

    max_trans = max_disp / focal[0] * sc
    dolly_poses = []
    dolly_focals = []

    # Dolly zoom
    for i in range(30):
        x_trans = 0.0
        y_trans = 0.0
        z_trans = max_trans * 2.5 * i / float(30 // 2)
        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )
        i_pose = np.linalg.inv(i_pose)
        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )
        render_pose = np.dot(ref_pose, i_pose)
        dolly_poses.append(render_pose[:3, :])
        new_focal = focal[0] - focal[0] * 0.1 * z_trans / max_trans / 2.5
        dolly_focals.append(new_focal)
    dolly_poses = np.stack(dolly_poses, 0)[:, :3]

    zoom_poses = []
    zoom_focals = []