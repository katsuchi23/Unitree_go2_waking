import mujoco
import mujoco.viewer
import numpy as np
import time

# Load your MJCF model
model = mujoco.MjModel.from_xml_path("/home/rey/isaacsim_ws/src/mujoco_menagerie/unitree_go2/scene_mjx.xml")
data = mujoco.MjData(model)
data.qpos[:] = np.array([ 
                    0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0,
                    0.1,  0.8, -1.5,  # FL
                    -0.1,  0.8, -1.5,  # FR
                    0.1,  1.0, -1.5,  # RL
                    -0.1,  1.0, -1.5   # RR
                ])
mujoco.mj_forward(model, data)
viewer = mujoco.viewer.launch_passive(model, data)
# print(model.nu, model.nq, model.nv)

      # Adjust the sleep time as needed
while True:

    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.001)

