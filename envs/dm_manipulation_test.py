from dm_control import manipulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dm_control import viewer

print('\n'.join(manipulation.ALL))
print('--------------------------------')
print('\n'.join(manipulation.get_environments_by_tag('vision')))

env = manipulation.load('reach_site_vision', seed=42)
# viewer.launch(env)
action_spec = env.action_spec()


def sample_random_action():
    return env.random_state.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
    ).astype(action_spec.dtype, copy=False)


frames = []
timestep = env.reset()
frames.append(timestep.observation['front_close'].squeeze())
while not timestep.last():
    # print(env.physics.data.time)
    timestep = env.step(sample_random_action())
    env.physics.render(height=200, width=200, camera_id=0)
    frames.append(timestep.observation['front_close'].squeeze())

video_fig, video_ax = plt.subplots(figsize=(4, 4))

def update(frame):
    video_ax.clear()  # 清除之前的帧
    video_ax.imshow(frames[frame])
    video_ax.axis('off')  # 关闭坐标轴
    return video_ax,

ani = FuncAnimation(video_fig, update, frames=len(
    frames), interval=1000 / 30)  # 30 FPS
plt.show()
