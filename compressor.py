"""Compressor Class"""
from __future__ import annotations

from typing import Any, Final

import torch
import numpy as np
from myNet import CompressNet
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import BaseEstimator, RegressorMixin
from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.signal_processing import compute_rms, resample
from clarity.utils.audiogram import Audiogram



EPS: Final = 1e-8


class Compressor:
    def __init__(
        self,
        fs: float = 44100.0,
        attack: float = 5.0,
        release: float = 20.0,
        threshold: float = 1.0,
        attenuation: float = 0.0001,
        rms_buffer_size: float = 0.2,
        makeup_gain: float = 1.0,
        **_kwargs,
    ) -> None:
        """Instantiate the Compressor Class.

        Args:
            fs (float): (default = 44100.0)
            attack (float): (default = 5.0)
            release float: (default = 20.0)
            threshold (float): (default = 1.0)
            attenuation (float): (default = 0.0001)
            rms_buffer_size (float): (default = 0.2)
            makeup_gain (float): (default = 1.0)
        """
        self.fs = fs
        self.rms_buffer_size = rms_buffer_size
        self.set_attack(attack)
        self.set_release(release)
        self.threshold = threshold
        self.attenuation = attenuation
        self.makeup_gain = makeup_gain

        # window for computing rms
        self.win_len = int(self.rms_buffer_size * self.fs)
        self.window = np.ones(self.win_len)

    def set_attack(self, t_msec: float) -> None:
        """DESCRIPTION

        Args:
            t_msec (float): DESCRIPTION

        Returns:
            float: DESCRIPTION
        """
        t_sec = t_msec / 1000.0
        reciprocal_time = 1.0 / t_sec
        self.attack = reciprocal_time / self.fs

    def set_release(self, t_msec: float) -> None:
        """DESCRIPTION

        Args:
            t_msec (float): DESCRIPTION

        Returns:
            float: DESCRIPTION
        """
        t_sec = t_msec / 1000.0
        reciprocal_time = 1.0 / t_sec
        self.release = reciprocal_time / self.fs

    def process(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Any]]:
        """DESCRIPTION

        Args:
            signal (np.array): DESCRIPTION

        Returns:
            np.array: DESCRIPTION
        """
        padded_signal = np.concatenate((np.zeros(self.win_len - 1), signal))
        rms = np.sqrt(
            np.convolve(padded_signal**2, self.window, mode="valid") / self.win_len
            + EPS
        )
        comp_ratios: list[float] = []
        curr_comp: float = 1.0
        for rms_i in rms:
            if rms_i > self.threshold:
                temp_comp = (rms_i * self.attenuation) + (
                    (1.0 - self.attenuation) * self.threshold
                )
                curr_comp = curr_comp * (1.0 - self.attack) + (temp_comp * self.attack)
            else:
                curr_comp = self.release + curr_comp * (1 - self.release)
            comp_ratios.append(curr_comp)
        return (signal * np.array(comp_ratios) * self.makeup_gain), rms, comp_ratios

class SidechainCompressor(Compressor):
    def __init__(
            self,
        fs: float = 44100.0,
        attack: float = 5.0,
        release: float = 20.0,
        threshold: float = 1.0,
        attenuation: float = 0.0001,
        rms_buffer_size: float = 0.2,
        makeup_gain: float = 1.0,
        ratio: float = 1.0
    ):
        super().__init__(fs, attack, release, threshold, 
                         attenuation,rms_buffer_size, makeup_gain)
        self.ratio = ratio


    def process(self, signal, sidechain_signal):
        if sidechain_signal.ndim == 2:
            print("sidechain_signal.ndim == 2")
            comp_ratios_list = []
            for ch in range(sidechain_signal.shape[1]):
                comp_ratios = self._compute_compression_ratios(sidechain_signal[:, ch])
                comp_ratios_list.append(comp_ratios)
            comp_ratios = np.mean(comp_ratios_list, axis=0)
        else:
            print("sidechain_signal.ndim != 2,", sidechain_signal.ndim)
            comp_ratios = self._compute_compression_ratios(sidechain_signal)
        compressed_signal = signal * comp_ratios[:len(signal)] * self.makeup_gain
        in_rms =  self.calculate_rms(signal)
        out_rms = self.calculate_rms(compressed_signal)

        if out_rms > 0:  # 避免除以零
            print("out_rms:", out_rms)
            self.makeup_gain = in_rms / out_rms
        else:
            self.makeup_gain = 1.0  # 保持增益不变

        # apply the final makeup_gain
        compressed_signal *= self.makeup_gain
        return compressed_signal, comp_ratios

    def _compute_compression_ratios(self, sidechain_signal):
        padded_sidechain = np.concatenate((np.zeros(self.win_len - 1), sidechain_signal))
        rms = np.sqrt(np.convolve(padded_sidechain**2, self.window, mode="valid") / self.win_len + 1e-10)
        comp_ratios = []
        ratio = 3
        curr_comp = 1.0
        
        for rms_i in rms:

            if rms_i > self.threshold:
                # print("rms_i:", rms_i)
                # 计算增益衰减（dB）
                gain_reduction_db = (rms_i - self.threshold) * (self.ratio - 1)
                temp_comp = 10 ** (-gain_reduction_db / 20)  # 转换为线性增益
            else:
                temp_comp = 1.0  # 未超过阈值，不压缩
            
            curr_comp = curr_comp * (1.0 - self.attack) + (temp_comp * self.attack)
            curr_comp = max(curr_comp, 0.0)  # 确保 curr_comp 为正
            
            # print("curr_comp:", curr_comp)
            comp_ratios.append(curr_comp)

        return np.array(comp_ratios)

    def calculate_rms(self, signal):
        
        # 确保信号为 numpy 数组
        signal = np.asarray(signal)

        # 计算 RMS 值
        rms_value = np.sqrt(np.mean(signal**2))

        return rms_value

class CompressorEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, attack=5.0, release=20.0, threshold=0.001, attenuation=0.8):
        self.attack = attack
        self.release = release
        self.threshold = threshold
        self.attenuation = attenuation

    def fit(self, X, y=None):
        return self

    def score(self, X, y=None):
        data = X[0]
        signal = data['signal'] # 背景音乐
        sidechain_signal = data['sidechain_signal'] # 侧链音乐
        reference_signal = data['reference_signal'] # mixture音乐
        audiogram = data['audiogram']
        
        compressor = SidechainCompressor(fs=44100, attack=self.attack, release=self.release, threshold=self.threshold, attenuation=self.attenuation)
        compressed_signal, _ = compressor.process(signal, sidechain_signal)
        # loss = np.mean((compressed_signal - signal) ** 2)
        # print("loss:", loss)
        # haaqi_score = compute_haaqi(
        #     processed_signal=compressed_signal,
        #     reference_signal=reference_signal,
        #     processed_sample_rate=44100,
        #     reference_sample_rate=44100,
        #     audiogram=audiogram,
        #     equalisation=1,
        #     level1=65.0,
        # )

        # Compute the scores

        haaqi_score = compute_haaqi(
            processed_signal=resample(
                compressed_signal,
                32000,
                24000,
            ),
            reference_signal=resample(
                reference_signal, 44100, 24000
            ),
            processed_sample_rate=24000,
            reference_sample_rate=24000,
            audiogram=audiogram,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_signal)),
        )

        # right_score = compute_haaqi(
        #     processed_signal=resample(
        #         enhanced_signal[:, 1],
        #         config.remix_sample_rate,
        #         config.HAAQI_sample_rate,
        #     ),
        #     reference_signal=resample(
        #         right_reference, config.sample_rate, config.HAAQI_sample_rate
        #     ),
        #     processed_sample_rate=config.HAAQI_sample_rate,
        #     reference_sample_rate=config.HAAQI_sample_rate,
        #     audiogram=listener.audiogram_right,
        #     equalisation=2,
        #     level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 1])),
        # )
        print("haaqi score:",haaqi_score)
        return haaqi_score

    def predict(self, X):
        data = X[0]
        signal = data['signal']
        sidechain_signal = data['sidechain_signal']
        
        compressor = SidechainCompressor(fs=44100, attack=self.attack, release=self.release, threshold=self.threshold, attenuation=self.attenuation)
        compressed_signal, _ = compressor.process(signal, sidechain_signal)
        return compressed_signal