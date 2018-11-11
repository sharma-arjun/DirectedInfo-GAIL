from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np

max_frame = 90

width = 480
height = 480
video = np.zeros((90, height, 2 * width, 3), dtype=np.uint8)

# Load one task:
env = suite.load(domain_name="cartpole", task_name="swingup")

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
while not time_step.last():
  for i in range(max_frame):
    action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
    time_step = env.step(action)
    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])
    #print(time_step.reward, time_step.discount, time_step.observation)
  for i in range(max_frame):
    img = plt.imshow(video[i])
    plt.pause(0.01)  # Need min display time > 0.0.
    plt.draw()
