import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco
import mujoco.viewer
import numpy as np
import time

class Env():
    def __init__(self, headless=True):
        self.MAX_STEP = 10_000  # max steps per episode
        self.MAX_EPISODE = 10000000  # max episodes
        self.progressed_flag = True
        self.not_progressed_start_time = 0
        self.current_step = 0
        self.model_path = "/home/rey/isaacsim_ws/src/mujoco_menagerie/unitree_go2/scene_mjx.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        self.default_position = np.array([ 
                                    0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0,
                                    0.1,  0.8, -1.5,  # FL
                                    -0.1,  0.8, -1.5,  # FR
                                    0.1,  1.0, -1.5,  # RL
                                    -0.1,  1.0, -1.5   # RR
                                ])
        self.data.qpos[:] = self.default_position.copy()
        self.initial_position = self.default_position.copy()
        self.initial_action = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
        self.prev_pos = self.data.qpos[0]
        self.contact_wrench = {
                  "FL": np.random.rand(6),
                  "FR": np.random.rand(6),
                  "RL": np.random.rand(6),
                  "RR": np.random.rand(6)}
        self.headless = headless
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.command = np.zeros(3)
        self.previous_action = self.initial_action.copy()

    def reset(self):
        self.current_step = 0
        self.progressed_flag = True
        self.not_progressed_start_time = 0
        self.data.qpos[:] = self.default_position.copy()
        self.prev_pos = self.data.qpos[0]
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self.contact_wrench = {
            "FL": np.random.rand(6),
            "FR": np.random.rand(6),
            "RL": np.random.rand(6),
            "RR": np.random.rand(6)}
        for i in range(300):
            action = np.zeros(12)
            self.previous_action = action.copy()
            self.step(action)
        self.initial_position = self.data.qpos[7:].copy()
        self.command[0] = 0.0
        # print(self.initial_position)
    
    def get_rotation_matrix_from_quaternion(self, quat):
        """Convert quaternion to rotation matrix"""
        # Assuming quat is [w,x,y,z] format - adjust if your MuJoCo uses [x,y,z,w]
        w, x, y, z = quat
        
        # Create rotation matrix
        rot_mat = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return rot_mat

    def get_wrench(self):
        for j in range(self.data.ncon):
            contact = self.data.contact[j]
            f_contact = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]
            mujoco.mj_contactForce(self.model, self.data, j, f_contact)

            if contact.geom1 == 0:  # Check if geom1 is the ground
                if contact.geom2 == 20:
                    self.contact_wrench['FL'] = f_contact
                elif contact.geom2 == 32:
                    self.contact_wrench['FR'] = f_contact
                elif contact.geom2 == 44:
                    self.contact_wrench['RL'] = f_contact
                elif contact.geom2 == 56:
                    self.contact_wrench['RR'] = f_contact

    def step(self, action=None, scale=0.5):
        if action is not None:
            self.data.ctrl[:] = action
            self.data.ctrl[0] = action[0] * scale + self.initial_action[0]
            self.data.ctrl[1] = action[4] * scale + self.initial_action[4]
            self.data.ctrl[2] = action[8] * scale + self.initial_action[8]
            self.data.ctrl[3] = action[1] * scale + self.initial_action[1]
            self.data.ctrl[4] = action[5] * scale + self.initial_action[5]
            self.data.ctrl[5] = action[9] * scale + self.initial_action[9]
            self.data.ctrl[6] = action[2] * scale + self.initial_action[2]
            self.data.ctrl[7] = action[6] * scale + self.initial_action[6]
            self.data.ctrl[8] = action[10] * scale + self.initial_action[10]
            self.data.ctrl[9] = action[3] * scale + self.initial_action[3]
            self.data.ctrl[10] = action[7] * scale + self.initial_action[7]
            self.data.ctrl[11] = action[11] * scale + self.initial_action[11]
            self.previous_action = action.copy()
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        if not self.headless:
            self.viewer.sync()
        self.current_step += 1

    def reward(self):  # reward based on forward motion in x direction
        forward_position = self.data.qpos[0]
        delta_forward_position = forward_position - self.prev_pos
        self.prev_pos = forward_position

        forward_velocity = self.data.qvel[0]
        forward_velocity_sign = 1 if forward_velocity > 0 else -1
        return forward_velocity_sign * (0.5 * forward_velocity ** 2) + delta_forward_position

    def terminate(self):
        return False

    def get_state(self):
        # Get base orientation (rotation matrix from world to base frame)
        base_quat = self.data.qpos[3:7]  # Base quaternion [w,x,y,z] or [x,y,z,w] depending on MuJoCo version
        base_rot_mat = self.get_rotation_matrix_from_quaternion(base_quat)
        
        # Root linear velocity in BASE frame (not world frame)
        root_linear_vel_world = self.data.qvel[:3].copy()
        root_linear_vel = base_rot_mat.T @ root_linear_vel_world  # Transform to base frame
        
        # Root angular velocity in BASE frame (not world frame)  
        root_angular_vel_world = self.data.qvel[3:6].copy()
        root_angular_vel = base_rot_mat.T @ root_angular_vel_world  # Transform to base frame
        
        # Projected gravity - unit vector in BASE frame
        gravity_world = np.array([0, 0, -1])  # Unit vector pointing down in world frame
        projected_gravity = base_rot_mat.T @ gravity_world  # Transform to base frame
        
        # Joint positions and velocities (your existing logic)
        position_dummy = self.data.qpos[7:].copy()
        velocity_dummy = self.data.qvel[6:].copy()
        position = np.zeros(12)
        velocity = np.zeros(12)

        # Your existing joint reordering logic
        position[0] = position_dummy[0].copy() - self.initial_position[0].copy()
        position[1] = position_dummy[3].copy() - self.initial_position[3].copy()
        position[2] = position_dummy[6].copy() - self.initial_position[6].copy()
        position[3] = position_dummy[9].copy() - self.initial_position[9].copy()
        position[4] = position_dummy[1].copy() - self.initial_position[1].copy()
        position[5] = position_dummy[4].copy() - self.initial_position[4].copy()
        position[6] = position_dummy[7].copy() - self.initial_position[7].copy()
        position[7] = position_dummy[10].copy() - self.initial_position[10].copy()
        position[8] = position_dummy[2].copy() - self.initial_position[2].copy()
        position[9] = position_dummy[5].copy() - self.initial_position[5].copy()
        position[10] = position_dummy[8].copy() - self.initial_position[8].copy()
        position[11] = position_dummy[11].copy() - self.initial_position[11].copy()

        velocity[0] = velocity_dummy[0]
        velocity[1] = velocity_dummy[3]
        velocity[2] = velocity_dummy[6]
        velocity[3] = velocity_dummy[9]
        velocity[4] = velocity_dummy[1]
        velocity[5] = velocity_dummy[4]
        velocity[6] = velocity_dummy[7]
        velocity[7] = velocity_dummy[10]
        velocity[8] = velocity_dummy[2]
        velocity[9] = velocity_dummy[5]
        velocity[10] = velocity_dummy[8]
        velocity[11] = velocity_dummy[11]
        

        # Rest of your code...
        acceleration = self.data.qacc
        contact_wrench = np.array([])
        self.get_wrench()
        for i in self.contact_wrench:
            contact_wrench = np.append(contact_wrench, self.contact_wrench[i])

        state = np.concatenate((root_linear_vel, root_angular_vel, projected_gravity, self.command, position, velocity, self.previous_action))
        reward = self.reward()
        done = self.terminate()
        # print(state)
        return state, reward, done


# class SharedModel(nn.Module):
#     """Exact replica of Isaac Lab's SharedModel for consistent inference"""
    
#     def __init__(self, obs_dim, act_dim):
#         super().__init__()
#         self.num_actions = act_dim
        
#         # Match Isaac Lab network architecture exactly
#         self.net_container = nn.Sequential(
#             nn.Linear(obs_dim, 512),  # LazyLinear gets replaced with regular Linear
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#             nn.Linear(256, 128),
#             nn.ELU(),
#         )
#         self.policy_layer = nn.Linear(128, self.num_actions)
#         self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
#         self.value_layer = nn.Linear(128, 1)

#         # Isaac Lab GaussianMixin parameters
#         self.clip_log_std = True
#         self.min_log_std = -20.0
#         self.max_log_std = 2.0
#         self.reduction = "sum"  # This affects how log_prob is computed

#     def forward(self, x):
#         """Forward pass matching Isaac Lab's compute method"""
#         features = self.net_container(x)
#         mean = self.policy_layer(features)
        
#         # Apply log_std clipping like Isaac Lab
#         log_std = self.log_std_parameter
#         if self.clip_log_std:
#             log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
#         std = torch.exp(log_std)
#         value = self.value_layer(features)
#         return mean, std, value

#     def act(self, states, deterministic=False):
#         """Action selection matching Isaac Lab's GaussianMixin.act()"""
#         mean, std, _ = self.forward(states)
        
#         if deterministic:
#             return mean
#         else:
#             # Sample from normal distribution
#             return torch.normal(mean, std)

#     def compute_log_prob(self, states, actions):
#         """Compute log probability matching Isaac Lab's method"""
#         mean, std, _ = self.forward(states)
        
#         # Compute log probability
#         var = std.pow(2)
#         log_prob = -((actions - mean).pow(2)) / (2 * var) - 0.5 * torch.log(2 * torch.pi * var)
        
#         # Apply reduction
#         if self.reduction == "sum":
#             return log_prob.sum(dim=-1)
#         elif self.reduction == "mean":
#             return log_prob.mean(dim=-1)
#         else:
#             return log_prob


# def load_checkpoint_correct_format(checkpoint_path, obs_dim, act_dim, device):
#     """Load checkpoint and handle potential format differences"""
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     # Create model
#     policy = SharedModel(obs_dim=obs_dim, act_dim=act_dim).to(device)
    
#     # Try to load state dict - handle different possible formats
#     if "policy" in checkpoint:
#         # Standard format from your training
#         try:
#             policy.load_state_dict(checkpoint["policy"], strict=True)
#         except RuntimeError as e:
#             print(f"Strict loading failed: {e}")
#             # Try non-strict loading
#             policy.load_state_dict(checkpoint["policy"], strict=False)
#     elif "model_state_dict" in checkpoint:
#         # Alternative format
#         policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
#     else:
#         # Checkpoint might be the state dict itself
#         policy.load_state_dict(checkpoint, strict=True)
    
#     return policy


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load checkpoint with correct model architecture
#     checkpoint_path = "/home/rey/fyp/isaac_sim/unitree_go2_direct/logs/skrl/cartpole_direct/2025-08-24_08-15-52_ppo_torch/checkpoints/best_agent.pt"
#     policy = load_checkpoint_correct_format(checkpoint_path, obs_dim=48, act_dim=12, device=device)
#     policy.eval()

#     env = Env(headless=False)
#     num_episodes = 1000

#     for episode in range(num_episodes):
#         env.reset()
#         state, _, _ = env.get_state()
#         total_reward = 0

#         for step in range(env.MAX_STEP):
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

#             action = env.previous_action.copy()
#             if (step) % 4 == 0:
#                 with torch.no_grad():
#                     # Use the same action selection as Isaac Lab
#                     action_tensor = policy.act(state_tensor, deterministic=False)  # Set to True for deterministic
#                     action = action_tensor.cpu().numpy().flatten()
#                     # print(action)
#             # action = np.zeros(12)

#             # Apply the same scaling as Isaac Lab (scale=0.5)
#             scaled_action = action * 0.5
#             # [print(f"Step {step}, Action: {scaled_action}")]
#             # print(state)
#             env.step(scaled_action)
#             next_state, reward, done = env.get_state()
#             total_reward += reward
#             state = next_state
            
#             if done:
#                 print(f"Episode {episode+1}, total reward: {total_reward:.2f}, steps: {step+1}")
#                 break
#         else:
#             print(f"Episode {episode+1} reached max steps {env.MAX_STEP}, total reward: {total_reward:.2f}")

#     print("Inference complete.")