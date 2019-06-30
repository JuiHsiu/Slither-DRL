import gym
from universe.wrappers import EpisodeID, BlockingReset
from universe.vectorized import core
import pyglet
import numpy as np
from PIL import Image

def resize(screen,shape):
	im = Image.fromarray(screen, mode='RGB')
	im = im.resize(size=shape, resample=Image.BILINEAR) # size = (width, height)
	im = np.asarray(im)
	return im

def multiActionTransform(action):
    """tranform array of action into pointer location and key event
    
    Args:
        action (ndarray): 1-D array integer, from 0-23
    
    Returns:
        list: list of actions
    """
    angles = action%12
    spaces = action//12
    off_x, off_y = 268, 234
    x = (off_x + 100*np.cos(angles*30* np.pi / 180. )).astype(int)
    y = (off_y + 100*np.sin(angles*30* np.pi / 180. )).astype(int)
    return [[('PointerEvent', x, y, False),('KeyEvent','space', space)]for (x,y,space) in zip(x,y,spaces)]

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True

        nchannels = arr.shape[-1]
        if nchannels == 1:
            _format = "I"
        elif nchannels == 3:
            _format = "RGB"
        else:
            raise NotImplementedError
        image = pyglet.image.ImageData(self.width, self.height, "RGB", arr.tobytes())

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()

class MultiWrapper(core.Wrapper):
    """
    Take a vectorized environment with any batch size and turn it into an unvectorized environment. (doing crop screen at the same time)
    """
    autovectorize = False
    metadata = {'runtime.vectorized': False}
    def __init__(self,env, resize):
        super(MultiWrapper, self).__init__(env)
        self.viewer = None
        self.resize = resize
        
    def _reset(self):
        self.observation_n = self.env.reset()
        self.observation_n = [ob['vision'][84:384, 18:518, :] for ob in self.observation_n]
        if self.resize != None:
            self.resize_obs = np.stack([resize(ob,shape=self.resize) for ob in self.observation_n],axis=0)/255
            self.observation_n = np.stack(self.observation_n,axis=0)
        else:
            self.observation_n = np.stack(self.observation_n,axis=0)
            self.resize_obs = self.observation_n/255
        
        return self.resize_obs

    def _step(self, actions):
        self.observation_n, reward_n, done_n, info = self.env.step(actions)
        self.observation_n = [ob['vision'][84:384, 18:518, :] for ob in self.observation_n]
        if self.resize != None:
            self.resize_obs = np.stack([resize(ob,shape=self.resize) for ob in self.observation_n],axis=0)/255
            self.observation_n = np.stack(self.observation_n,axis=0)
        else:
            self.observation_n = np.stack(self.observation_n,axis=0)
            self.resize_obs = self.observation_n/255

        reward_n = np.stack(reward_n,axis=0)
        done_n = np.stack(done_n,axis=0)
#         info = info['n']
        return self.resize_obs, reward_n, done_n, info

    def _seed(self, seed):
        return self.env.seed([seed])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                for viewer in self.viewer:
                    viewer.close()
                self.viewer = None
            return
        imgs = [img.astype(np.uint8) for img in self.observation_n]
        if mode == 'rgb_array':
            return imgs[0]
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = [SimpleImageViewer() for _ in range(len(imgs))]
            for idx,viewer in enumerate(self.viewer):
                viewer.imshow(imgs[idx][::-1])

def MultiEnv(resize=None):
    env = gym.make('internet.SlitherIO-v0')
    env = BlockingReset(env)
    env = MultiWrapper(env,resize)
    return env