import mujoco
import mujoco.viewer
import numpy as np
import time

# Load your MJCF model
model = mujoco.MjModel.from_xml_path("source/unitree_go2_direct/unitree_go2_direct/usd/scene.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)

while True:
    mujoco.mj_step(model, data)
    viewer.sync()
    # get_wrench()
    time.sleep(0.001)

