from typing import List, Generator, Union
from abc import ABC
from collections.abc import Sequence
import types

import cv2
import numpy as np

__all__ = [
    'FullSampler',
    'SystematicSampler',
    'RandomSampler',
    'OnceRandomSampler',
    'RandomTemporalSegmentSampler',
    'OnceRandomTemporalSegmentSampler',
    'LambdaSampler',
    'synchronize_state',
]


class _MediaCapture:
    def __init__(self, source):
        self.source = source
        if isinstance(source, Sequence) and not isinstance(source, str):
            self.paths = list(source)
            self.is_video = False
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_video = True
        self._frame_id = 0

    @classmethod
    def from_video_capture(cls, cap):
        raise NotImplementedError

    def is_opened(self):
        if self.is_video:
            return self.cap.isOpened()
        else:
            return len(self.paths) > 0

    def get(self, prop):
        if self.is_video:
            return self.cap.get(prop)

    def set(self, prop, value):
        if self.is_video:
            return self.cap.set(prop, value)

    def read(self):
        if self.is_video:
            self._frame_id = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok, frame = self.cap.read()
        else:
            frame = cv2.imread(self.paths[self._frame_id])
            ok = frame is not None
        self._frame_id += 1
        return ok, frame

    def release(self):
        if self.is_video:
            self.cap.release()
        else:
            self.paths.clear()

    def seek(self, frame_id):
        if self.is_video:
            if frame_id == self._frame_id:
                return
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self._frame_id = frame_id

    @property
    def frame_count(self):
        if self.is_video:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return len(self.paths)

    @property
    def fps(self):
        if self.is_video:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.

    @property
    def frame_id(self):
        return self._frame_id

    def sample(self, frame_ids):
        frames = []
        for frame_id in frame_ids:
            self.seek(frame_id)
            ok, frame = self.read()
            if not ok:
                if self.is_video:
                    raise RuntimeError(f'Unable to read frame {frame_id} of {self.source}.')
                else:
                    raise RuntimeError(f'Unable to read file {self.paths[frame_id]}.')
            frames.append(frame)
        return frames

    def __str__(self):
        ret = f'{self.__class__.__name__}'
        ret += f'(source="{self.source}")' if self.is_video else f'(source={self.source})'
        return ret


class _BaseSampler(ABC):
    def __init__(self, n_frames=16):
        if not n_frames:
            raise ValueError(f'n_frames must be positive number, got {n_frames}.')
        self.n_frames = n_frames
        self._presampling_hooks = []

    def __call__(self, source, start_frame=None, end_frame=None, sample_id=None):
        cap = _MediaCapture(source)
        if not cap.is_opened():
            raise RuntimeError(f'{source} is invalid.')
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = cap.frame_count - 1
        elif end_frame > cap.frame_count - 1:
            end_frame = cap.frame_count - 1

        for hook in self._presampling_hooks:
            hook(source, start_frame, end_frame, sample_id)
        sampled_frame_ids = self._get_sampled_frame_ids(source, start_frame, end_frame, sample_id)
        return cap.sample(sampled_frame_ids)

    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id):
        raise NotImplementedError

    def register_presampling_hook(self, hook):
        self._presampling_hooks.append(hook)

    def clear_presampling_hooks(self):
        self._presampling_hooks.clear()


class _BaseMemorizedSampler(_BaseSampler, ABC):
    def __init__(self, n_frames=16):
        super(_BaseMemorizedSampler, self).__init__(n_frames)
        self.memory = {}

    def __call__(self, source, start_frame=None, end_frame=None, sample_id=None):
        if sample_id is None:
            raise RuntimeError('sample_id is required.')
        return super(_BaseMemorizedSampler, self).__call__(source, start_frame, end_frame, sample_id)

    def clear(self):
        self.memory.clear()


class FullSampler(_BaseSampler):
    """Sample all frames"""

    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        return list(range(start_frame, end_frame))


class SystematicSampler(_BaseSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        sampled_frame_ids = np.linspace(start_frame, end_frame, self.n_frames)
        return sampled_frame_ids.round().astype(np.int64)


class RandomSampler(_BaseSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        sampled_frame_ids = start_frame + np.random.rand(self.n_frames) * (end_frame - start_frame)
        sampled_frame_ids.sort()
        return sampled_frame_ids.round().astype(np.int64)


class OnceRandomSampler(_BaseMemorizedSampler, RandomSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        if sample_id in self.memory:
            return self.memory[sample_id]
        sampled_frame_ids = RandomSampler._get_sampled_frame_ids(self, source, start_frame, end_frame)
        self.memory[sample_id] = sampled_frame_ids
        return sampled_frame_ids


class RandomTemporalSegmentSampler(_BaseSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        segments = np.linspace(start_frame, end_frame, self.n_frames + 1)
        segment_length = (end_frame - start_frame) / self.n_frames
        sampled_frame_ids = segments[:-1] + np.random.rand(self.n_frames) * segment_length
        return sampled_frame_ids.round().astype(np.int64)


class OnceRandomTemporalSegmentSampler(_BaseMemorizedSampler, RandomTemporalSegmentSampler):
    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        if sample_id in self.memory:
            return self.memory[sample_id]
        sampled_frame_ids = RandomTemporalSegmentSampler._get_sampled_frame_ids(self, source, start_frame, end_frame)
        self.memory[sample_id] = sampled_frame_ids
        return sampled_frame_ids


class LambdaSampler(_BaseSampler):
    def __init__(self, get_sampled_frame_ids_func):
        super(LambdaSampler, self).__init__(n_frames=0)
        self.get_sampled_frame_ids_func = get_sampled_frame_ids_func

    def _get_sampled_frame_ids(self, source, start_frame, end_frame, sample_id=None):
        return self.get_sampled_frame_ids_func(source, start_frame, end_frame)


class synchronize_state:
    def __init__(self, samplers: Union[List[_BaseSampler], Generator]):
        if isinstance(samplers, types.GeneratorType):
            samplers = list(samplers)
        self.samplers = samplers
        self._random_state = None

    def __enter__(self):
        self._random_state = np.random.get_state()
        for sampler in self.samplers:
            sampler.register_presampling_hook(self._reuse_numpy_state)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for sampler in self.samplers:
            sampler.clear_presampling_hooks()
        self._random_state = None

    def _reuse_numpy_state(self, *args, **kwargs):
        np.random.set_state(self._random_state)
