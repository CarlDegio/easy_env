from dm_control import suite
from dm_control import manipulation
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.animation import FuncAnimation

max_len = max(len(d) for d, _ in suite.BENCHMARKING)
for domain, task in suite.BENCHMARKING:
    print(f'{domain:<{max_len}}  {task}')

random_state = np.random.RandomState(42)
env = suite.load('hopper', 'stand', task_kwargs={'random': random_state})

# Simulate episode with random actions
duration = 4  # Seconds
frames = []
ticks = []
rewards = []
observations = []

spec = env.action_spec()
time_step = env.reset()

while env.physics.data.time < duration:
    action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)

    camera0 = env.physics.render(camera_id=0, height=200, width=200)
    camera1 = env.physics.render(camera_id=1, height=200, width=200)
    frames.append(np.hstack((camera0, camera1)))
    rewards.append(time_step.reward)
    observations.append(copy.deepcopy(time_step.observation))
    ticks.append(env.physics.data.time)


# Show video and plot reward and observations
num_sensors = len(time_step.observation)

fig, ax = plt.subplots(1 + num_sensors, 1, sharex=True, figsize=(4, 8))
ax[0].plot(ticks, rewards)
ax[0].set_ylabel('reward')
ax[-1].set_xlabel('time')

for i, key in enumerate(time_step.observation):
    data = np.asarray([observations[j][key] for j in range(len(observations))])
    ax[i+1].plot(ticks, data, label=key)
    ax[i+1].set_ylabel(key)

# 新增代码：创建新的 figure 用于播放视频
video_fig, video_ax = plt.subplots(figsize=(4, 4))

def update(frame):
    video_ax.clear()  # 清除之前的帧
    video_ax.imshow(frames[frame])
    video_ax.axis('off')  # 关闭坐标轴
    return video_ax,

ani = FuncAnimation(video_fig, update, frames=len(frames), interval=1000 / 30)  # 30 FPS
plt.show()
