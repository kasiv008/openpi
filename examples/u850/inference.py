import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import u850_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import time
import cv2
from xarm.wrapper import XArmAPI
import numpy as np

def init_xarm(ip,init_pos=None):
    arm = XArmAPI(ip)
    #arm.reset()
    arm.clean_error()
    arm.clean_warn()
    # arm.clean_gripper_error()
    
    # # xarm gripper
    # arm.set_collision_tool_model(1) # xarm gripper
    # arm.set_gripper_enable(enable=True)
    # arm.set_gripper_mode(0)

    #2f140
    arm.set_collision_tool_model(5)
    arm.robotiq_reset()
    arm.robotiq_set_activate()

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(2)
    if init_pos == None:
        _, init_pos = tuple(arm.get_initial_point())
    arm.set_servo_angle(angle=init_pos,wait=True,is_radian=False)
    return arm, init_pos

config = _config.get_config("pi0_fast_u850")
checkpoint_dir = download.maybe_download("/home/kasi/Desktop/l5vel/openpi/checkpoints/pi0_fast_u850/my_experiment_robotiq_fixed/29999")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

top_camera = cv2.VideoCapture(0)
wrist_camera = cv2.VideoCapture(1)
prompt = "pick the toy from ground and place it on the table"

arm_ip = "172.16.0.11"
arm, init_pos = init_xarm(arm_ip)

while True:
    start = time.time()
    # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
    top_image = top_camera.read()[1]
    wrist_image = wrist_camera.read()[1]
    code, angles = arm.get_servo_angle()
    pos_gripper = arm.robotiq_status['gPO'] #2f140
    code_gripper = arm.robotiq_status['gFLT']

    state = np.array(angles[:7], dtype=np.float32)  # Get the first 7 angles
    state = np.concatenate((state, [pos_gripper]), axis=0)  # Add gripper position
    
    observation = {
        "state": state,
        "image": top_image,
        "wrist_image": wrist_image,
        "prompt": prompt,
    }
    result = policy.infer(observation)
    print("Result:", result["action"])
    #del policy
    print("Actions shape:", result["action"].shape)
    print("Inference time:", time.time() - start)


# Delete the policy to free up memory.


