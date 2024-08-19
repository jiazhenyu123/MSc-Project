import numpy as np          # Used for numerical computations and array manipulations
import hydra                # Used for reading configuration files
import json                 # Used for handling JSON data
import gym                  # Used for creating and managing reinforcement learning environments
import csv                  # Used for reading and saving CSV files
import os                   # Used for file and directory operations
import torch                # Used for deep learning operations
from gym import spaces      # Used for defining state and action spaces in reinforcement learning environments
from stable_baselines3 import PPO  # Proximal Policy Optimization algorithm for reinforcement learning
import logging              # Used for logging messages and debugging
import librosa              # Used for audio processing tasks
from clarity.enhancer.compressor import SidechainCompressor  # Custom sidechain compressor class
from clarity.enhancer.nalr import NALR  # Custom audio enhancer class
from clarity.enhancer.compressor import Compressor  # Custom compressor class
from clarity.utils.file_io import read_signal  # Utility function to read audio signals
from recipes.cad_icassp_2024.baseline.sidechain_demo_test import decompose_signal, process_remix_for_listener  # Custom functions for signal processing
from clarity.utils.source_separation_support import get_device, separate_sources  # Utility functions for source separation
from clarity.utils.audiogram import Listener  # Custom class for listener audiograms
from clarity.evaluator.haaqi import compute_haaqi  # Function to compute the HAAQI score
from clarity.utils.signal_processing import resample  # Utility function to resample audio signals
from clarity.utils.signal_processing import compute_rms  # Function to compute the root mean square of a signal
from evaluate import load_reference_stems  # Function to load reference audio stems
from evaluate import set_song_seed  # Function to set the random seed for a song
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB  # Pre-trained model for source separation
from stable_baselines3.common.env_util import make_vec_env  # Utility to create vectorized environments
from stable_baselines3.common.vec_env import SubprocVecEnv  # Subprocess-based vectorized environment
from stable_baselines3.common.callbacks import BaseCallback  # Base class for custom callbacks

from pathlib import Path  # Provides an object-oriented interface to handle filesystem paths
from clarity.utils.signal_processing import (
    clip_signal,
    denormalize_signals,
    normalize_signal,
    resample,
    to_16bit,
)  # Importing several signal processing utilities
from recipes.cad_icassp_2024.baseline.evaluate import (
    apply_gains,
    apply_ha,
    make_scene_listener_list,
    remix_stems,
)  # Importing custom functions for evaluation
from concurrent.futures import ProcessPoolExecutor  # Used for parallel execution of tasks

logger = logging.getLogger(__name__)  # Setting up the logger

# Custom environment class for audio compression, inheriting from the gym.Env class
class AudioCompressionEnv(gym.Env):
    def __init__(self, config, scene_listener_pair_idx):
        print("-------------------init AudioCompressionEnv-------------------")
        super(AudioCompressionEnv, self).__init__()
        
        # Define the observation space, consisting of audio features and compressor parameters
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(192,))
        
        # Define the action space for the environment, representing the compressor parameters
        ratio_discrete_steps = np.arange(2.0, 7.1, 0.1).tolist()  # Discrete steps for ratio parameter
        n_ratio_discrete = len(ratio_discrete_steps)

        self.action_space = spaces.MultiDiscrete([40, n_ratio_discrete, 86, 2001] * 6)

        self.config = config  # Store the configuration
        # Load and initialize the source separation model based on the configuration
        if config.separator.model == "demucs":
            self.separation_model = HDEMUCS_HIGH_MUSDB.get_model()
            self.model_sample_rate = HDEMUCS_HIGH_MUSDB.sample_rate
            self.sources_order = self.separation_model.sources
            self.normalise = True
            print("Demucs model loaded successfully.")
        elif config.separator.model == "openunmix":
            self.separation_model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq", niter=0)
            self.model_sample_rate = self.separation_model.sample_rate
            self.sources_order = ["vocals", "drums", "bass", "other"]
            self.normalise = False
            print("OpenUnmix model loaded successfully.")
        else:
            raise ValueError(f"Separator model {config.separator.model} not supported.")
        
        # Set the device for processing and move the model to this device
        self.device, _ = get_device(config.separator.device)
        self.separation_model.to(self.device)
        # Set the output directory where processed signals will be saved
        self.enhanced_folder = Path("enhanced_signals")
        self.enhanced_folder.mkdir(parents=True, exist_ok=True)

        # Load data from configuration files (gains, scenes, listeners, etc.)
        with Path(config.path.gains_file).open("r", encoding="utf-8") as file:
            self.gains = json.load(file)
        with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
            self.scenes = json.load(file)
        with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
            self.scenes_listeners = json.load(file)
        with Path(config.path.music_file).open("r", encoding="utf-8") as file:
            self.songs = json.load(file)

        # Select a batch to process
        self.scene_listener_pairs = make_scene_listener_list(
            self.scenes_listeners, config.evaluate.small_test
        )
        self.scene_listener_pairs = self.scene_listener_pairs[
            config.evaluate.batch :: config.evaluate.batch_size
        ]

        # Load listener audiograms and songs, enhancer and compressor models
        self.listener_dict = Listener.load_listener_dict(config.path.listeners_file)
        self.enhancer = NALR(**config.nalr)
        self.compressor = Compressor(**config.compressor)
        
        # Initialize variables to store HAAQI scores
        self.current_haaqi_score = None
        self.current_haaqi_score_left = None
        self.current_haaqi_score_right = None
        self.listener = None

        # Initialize the index for scene-listener pairs
        self.scene_listener_pair_idx = scene_listener_pair_idx

        self.current_audio_stems = None  # Store the current audio stems
        self.current_compressed_audio_stems = None  # Store the compressed audio stems
        self.mixture_signal = None  # Store the mixture signal
        self.scene_id = None  # Store the current scene ID

        # Load previous HAAQI scores from a file
        self.previous_haaqi_score = self._load_previous_scores("/Users/jiazhenyu/Documents/GitHub/clarity-0.4.1/recipes/cad_icassp_2024/baseline/haaqi_score.txt")
        
        # Initialize parameters for incrementing the HAAQI target
        self.initial_haaqi_increment = 0.01
        self.current_haaqi_increment = self.initial_haaqi_increment
        self.increase_rate = 0.01
        self.episodes_per_increase = 5

        # Counter for the number of completed episodes
        self.episode_count = 0
        self.target_haaqi_increment = self.current_haaqi_increment

        # Initialize the log file for recording training data
        self.log_file = "training_log_small_version.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Step", "Reward", "Bass_params_left", "Bass_params_right", "Other_params_left", "Other_params_right", "Drums_params_left", "Drums_params_right", "HAAQI_score_left", "HAAQI_score_right", "HAAQI_score", "Original HAAQI_score", "Done"])
        

    def seed(self, seed=None):
        np.random.seed(seed)
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

    def reset(self):
        print("---------------------reset---------------------")
        # Reset the episode counter
        self.episode_count += 1

        # Increase the target HAAQI increment every few episodes
        if self.episode_count % self.episodes_per_increase == 0:
            self.current_haaqi_increment += self.increase_rate
            self.target_haaqi_increment = self.current_haaqi_increment

        self.episode = self.scene_listener_pair_idx  # Store the current episode index
        self.step_count = 0  # Initialize the step counter

        # Load a new audio sample and reset the environment
        self.current_audio_stems = self._load_new_audio_sample()
        self.current_compressed_audio_stems = self.current_audio_stems.copy()  # Initialize with the original audio
        
        state = self._get_initial_state()
        return state
    
    def _extract_audio_features(self, audio, sr=44100):
        # Extract various audio features from the given audio signal
        # Root Mean Square energy
        rms = np.mean(librosa.feature.rms(y=audio))  
        # Spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)) 
        # Peak value of the audio 
        peak = np.max(np.abs(audio))  
        # Mel-Frequency Cepstral Coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  
        mfcc_mean = np.mean(mfccs, axis=1)
        # Spectral bandwidth
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))  
        # Spectral roll-off
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))  
        # Zero-crossing rate
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))  
        # Combine all features into a single array
        features = np.array([rms, spectral_centroid, peak, spectral_bandwidth, spectral_rolloff, zero_crossing_rate])
        features = np.concatenate((features, mfcc_mean))
        
        return features
    
    def _load_previous_scores(self, score_file):
        # Load previous HAAQI scores from a CSV file and store them in a dictionary
        previous_scores = {}
        with open(score_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = (row['scene'], row['song'], row['listener'])
                previous_scores[key] = float(row['score'])
        return previous_scores
    
    def _load_new_audio_sample(self):
        # Load a new audio sample based on the current index
        self.scene_id, listener_id = self.scene_listener_pairs[self.scene_listener_pair_idx]
        scene = self.scenes[self.scene_id]
        song_name = f"{scene['music']}-{scene['head_loudspeaker_positions']}"
        print(f"Loading song: {song_name}")

        num_scenes = len(self.scene_listener_pairs)
        logger.info(
            f"[{self.scene_listener_pair_idx + 1:03d}/{num_scenes:03d}] "
            f"Processing {self.scene_id}: {song_name} for listener {listener_id}"
        )
        # Get the listener's audiogram
        self.listener = self.listener_dict[listener_id]

        # Load and separate the audio signal
        print("Loading mixture signal")
        print("mixture signal filename path:", self.config.path.music_dir + '/' + self.songs[song_name]["Path"] + '/' + "mixture.wav")
        self.mixture_signal = read_signal(
            filename=Path(self.config.path.music_dir) / 
            self.songs[song_name]["Path"] /
              "mixture.wav",
            sample_rate=self.config.sample_rate,
            allow_resample=True,
        )

        # Decompose the signal into stems using the separation model
        stems = decompose_signal(
            model=self.separation_model,
            model_sample_rate=self.model_sample_rate,
            signal=self.mixture_signal,
            signal_sample_rate=self.config.sample_rate,
            device=self.device,
            sources_list=self.sources_order,
            listener=None,  # Listener is ignored in this baseline system
            normalise=self.normalise,
        )

        stems = apply_gains(stems, self.config.sample_rate, self.gains[scene["gain"]])

        # Load reference signals for comparison
        self.reference_stems, self.original_mixture = load_reference_stems(
            Path(self.config.path.music_dir) / self.songs[song_name]["Path"]
        )
        self.reference_stems = apply_gains(
            self.reference_stems, self.config.sample_rate, self.gains[scene["gain"]]
        )
        self.reference_mixture = remix_stems(
            self.reference_stems, self.original_mixture, self.config.sample_rate
        )

        # Update the scene listener pair index for the next sample
        self.scene_listener_pair_idx += 1
        if self.scene_listener_pair_idx >= num_scenes:
            self.song_index = 0  # Loop back to the first song

        return stems

    def step(self, action):
        print("---------------------step---------------------")
        # Extract compressor parameters from the action
        ratio_discrete_steps = np.arange(2.0, 7.1, 0.1).tolist()

        # Convert action values into actual compressor parameters
        threshold_values = -action[::4] - 1
        ratio_values = [ratio_discrete_steps[idx] for idx in action[1::4]]
        attack_values = action[2::4] + 15
        release_values = action[3::4] + 20

        # Group parameters for each channel
        bass_left_params = np.array([threshold_values[0], ratio_values[0], attack_values[0], release_values[0]])
        bass_right_params = np.array([threshold_values[1], ratio_values[1], attack_values[1], release_values[1]])
        other_left_params = np.array([threshold_values[2], ratio_values[2], attack_values[2], release_values[2]])
        other_right_params = np.array([threshold_values[3], ratio_values[3], attack_values[3], release_values[3]])
        drums_left_params = np.array([threshold_values[4], ratio_values[4], attack_values[4], release_values[4]])
        drums_right_params = np.array([threshold_values[5], ratio_values[5], attack_values[5], release_values[5]])

        # Apply compression using the selected parameters
        compressed_audio_stems = self._apply_compression(bass_left_params, bass_right_params,
                                                   other_left_params, other_right_params,
                                                   drums_left_params, drums_right_params)
        
        self.current_compressed_audio_stems = compressed_audio_stems
        
        # Remix stems to get the final audio signal
        current_compressed_audio = remix_stems(compressed_audio_stems, self.mixture_signal, self.model_sample_rate)

        self.current_compressed_audio = process_remix_for_listener(
            signal=current_compressed_audio,
            enhancer=self.enhancer,
            compressor=self.compressor,
            listener=self.listener,
            apply_compressor=False,
        )

        # Extract new audio features for the next state
        new_state = self._get_next_state(self.current_compressed_audio_stems, action)
        
        # Calculate the reward based on the current audio quality
        reward = self._calculate_reward()
        
        # Check if the episode is done
        done = self._is_done(self.current_haaqi_score)
        print("is done:", done)
        
        self.step_count += 1  # Increment the step counter
        # Log the current step's data
        key_list = list(self.previous_haaqi_score.keys())
        current_key = key_list[self.scene_listener_pair_idx - 1]
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.episode, self.step_count, reward, bass_left_params, bass_right_params, 
                             other_left_params, other_right_params, drums_left_params, drums_right_params, 
                             self.current_haaqi_score_left, self.current_haaqi_score_right, self.current_haaqi_score, 
                             self.previous_haaqi_score[current_key], done])

        return new_state, reward, done, {}

    def _get_initial_state(self):
        # Extract audio features and combine them with initial compressor parameters to form the initial state
        print("---------------------_get_initial_state---------------------")
        vocals_features_left = self._extract_audio_features(self.current_audio_stems["vocals"][:, 0])
        vocals_features_right = self._extract_audio_features(self.current_audio_stems["vocals"][:, 1])
        bass_features_left = self._extract_audio_features(self.current_audio_stems["bass"][:, 0])
        bass_features_right = self._extract_audio_features(self.current_audio_stems["bass"][:, 1])
        drums_features_left = self._extract_audio_features(self.current_audio_stems["drums"][:, 0])
        drums_features_right = self._extract_audio_features(self.current_audio_stems["drums"][:, 1])
        other_features_left = self._extract_audio_features(self.current_audio_stems["other"][:, 0])
        other_features_right = self._extract_audio_features(self.current_audio_stems["other"][:, 1])
        
        # Generate random initial parameters for the compressors
        ratio_discrete_steps = np.arange(2.0, 7.1, 0.1).tolist()
        n_ratio_discrete = len(ratio_discrete_steps)

        threshold_initial = -np.random.randint(1, 41, size=6)  # [1, 40]
        ratio_initial = np.random.randint(0, n_ratio_discrete, size=6)  # [0, n_ratio_discrete - 1]
        attack_initial = np.random.randint(10, 201, size=6)  # [10, 200]
        release_initial = np.random.randint(20, 2001, size=6)  # [20, 2000]

        initial_params = np.concatenate([threshold_initial, ratio_initial, attack_initial, release_initial])
        print("initial_params:", initial_params)
        
        # Extract Listener features (audiogram)
        audiogram_left = self.listener.audiogram_left.levels
        audiogram_right = self.listener.audiogram_right.levels
        listener_features = np.concatenate([audiogram_left, audiogram_right])
                
        # Combine all features and parameters to form the state
        state = np.concatenate((vocals_features_left, vocals_features_right,
                                bass_features_left, bass_features_right,
                                drums_features_left, drums_features_right,
                                other_features_left, other_features_right,
                                listener_features,  # Add listener features
                                initial_params))
        return state
    
    def _apply_compression(self, bass_left_params, bass_right_params,
                           other_left_params, other_right_params,
                           drums_left_params, drums_right_params):
        # Apply sidechain compression to the audio stems using the provided parameters
        print("---------------------_apply_compression---------------------")
        print("bass_left_params:", bass_left_params)
        print("bass_right_params:", bass_right_params)
        print("other_left_params:", other_left_params)
        print("other_right_params:", other_right_params)
        print("drums_left_params:", drums_left_params)
        print("drums_right_params:", drums_right_params)
        sidechain_bass_compressor_left = SidechainCompressor(fs=44100, threshold=bass_left_params[0], ratio=bass_left_params[1], attack=bass_left_params[2], release=bass_left_params[3])
        sidechain_bass_compressor_right = SidechainCompressor(fs=44100, threshold=bass_right_params[0], ratio=bass_right_params[1], attack=bass_right_params[2], release=bass_right_params[3])
        
        sidechain_other_compressor_left = SidechainCompressor(fs=44100, threshold=other_left_params[0], ratio=other_left_params[1], attack=other_left_params[2], release=other_left_params[3])
        sidechain_other_compressor_right = SidechainCompressor(fs=44100, threshold=other_right_params[0], ratio=other_right_params[1], attack=other_right_params[2], release=other_right_params[3])
        
        sidechain_drums_compressor_left = SidechainCompressor(fs=44100, threshold=drums_left_params[0], ratio=drums_left_params[1], attack=drums_left_params[2], release=drums_left_params[3])
        sidechain_drums_compressor_right = SidechainCompressor(fs=44100, threshold=drums_right_params[0], ratio=drums_right_params[1], attack=drums_right_params[2], release=drums_right_params[3])

        # Apply compression to each stem and return the compressed audio stems
        print("compressed_bass_left")
        compressed_bass_left, _ = sidechain_bass_compressor_left.process(self.current_audio_stems["bass"][:, 0], self.current_audio_stems["vocals"][:, 0])
        print("compressed_bass_right")
        compressed_bass_right, _ = sidechain_bass_compressor_right.process(self.current_audio_stems["bass"][:, 1], self.current_audio_stems["vocals"][:, 1])
        print("compressed_other_left")
        compressed_other_left, _ = sidechain_other_compressor_left.process(self.current_audio_stems["other"][:, 0], self.current_audio_stems["vocals"][:, 0])
        print("compressed_other_right")
        compressed_other_right, _ = sidechain_other_compressor_right.process(self.current_audio_stems["other"][:, 1], self.current_audio_stems["vocals"][:, 1])
        print("compressed_drums_left")
        compressed_drums_left, _ = sidechain_drums_compressor_left.process(self.current_audio_stems["drums"][:, 0], self.current_audio_stems["vocals"][:, 0])
        print("compressed_drums_right")
        compressed_drums_right, _ = sidechain_drums_compressor_right.process(self.current_audio_stems["drums"][:, 1], self.current_audio_stems["vocals"][:, 1])
        
        # Combine the compressed stems into a dictionary
        compressed_audio_stem = {
            "vocals": self.current_audio_stems["vocals"],
            "bass": np.stack([compressed_bass_left, compressed_bass_right], axis=1),
            "other": np.stack([compressed_other_left, compressed_other_right], axis=1),
            "drums": np.stack([compressed_drums_left, compressed_drums_right], axis=1)
        }

        return compressed_audio_stem

    def _get_next_state(self, compressed_audio_stems, action):
        # Extract audio features from the compressed audio stems to form the next state
        print("---------------------_get_next_state---------------------")
        vocals_features_left = self._extract_audio_features(compressed_audio_stems["vocals"][:, 0])
        vocals_features_right = self._extract_audio_features(compressed_audio_stems["vocals"][:, 1])
        bass_features_left = self._extract_audio_features(compressed_audio_stems["bass"][:, 0])
        bass_features_right = self._extract_audio_features(compressed_audio_stems["bass"][:, 1])
        drums_features_left = self._extract_audio_features(compressed_audio_stems["drums"][:, 0])
        drums_features_right = self._extract_audio_features(compressed_audio_stems["drums"][:, 1])
        other_features_left = self._extract_audio_features(compressed_audio_stems["other"][:, 0])
        other_features_right = self._extract_audio_features(compressed_audio_stems["other"][:, 1])
        
        # Extract the current compressor parameters
        current_compressor_params = np.array(action)

        # Extract Listener features
        audiogram_left = self.listener.audiogram_left.levels
        audiogram_right = self.listener.audiogram_right.levels
        listener_features = np.concatenate([audiogram_left, audiogram_right])
        
        # Combine all features and parameters to form the next state
        state = np.concatenate((vocals_features_left, vocals_features_right,
                                bass_features_left, bass_features_right,
                                drums_features_left, drums_features_right,
                                other_features_left, other_features_right,
                                listener_features,  # Add listener features
                                current_compressor_params))
        return state
    
    def _calculate_reward(self):
        # Calculate the reward based on the HAAQI score of the current audio
        print("---------------------_calculate_reward---------------------")
        signal = self.current_compressed_audio
        # Resample the signal to the expected output sample rate
        if self.config.sample_rate != self.config.remix_sample_rate:
            signal = resample(signal, self.config.sample_rate, self.config.remix_sample_rate)
        # Clip the signal to prevent clipping distortion
        signal, n_clipped = clip_signal(signal, False)
        # Convert the signal to 16-bit integer format
        signal = to_16bit(signal)
        signal = (signal / 32768.0).astype(np.float32)
        # Apply hearing aid processing to the reference signals
        left_reference = apply_ha(
            enhancer=self.enhancer,compressor=None,
            signal=self.reference_mixture[:, 0], audiogram=self.listener.audiogram_left,
            apply_compressor=False,sidechain_signal=None
        )
        right_reference = apply_ha(
            enhancer=self.enhancer, compressor=None,
            signal=self.reference_mixture[:, 1], audiogram=self.listener.audiogram_right,
            apply_compressor=False, sidechain_signal=None
        )
        # Set the random seed for the scene
        if self.config.evaluate.set_random_seed:
            set_song_seed(self.scene_id)

        # Compute the HAAQI scores for the left and right channels
        left_score = compute_haaqi(
            processed_signal=resample(
                signal[:, 0],
                self.config.remix_sample_rate,self.config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                left_reference, self.config.sample_rate, self.config.HAAQI_sample_rate
            ),
            processed_sample_rate=self.config.HAAQI_sample_rate,
            reference_sample_rate=self.config.HAAQI_sample_rate,
            audiogram=self.listener.audiogram_left, equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(self.reference_mixture[:, 0])),
        )
        right_score = compute_haaqi(
            processed_signal=resample(
                signal[:, 1],self.config.remix_sample_rate,self.config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                right_reference, self.config.sample_rate, self.config.HAAQI_sample_rate
            ),
            processed_sample_rate=self.config.HAAQI_sample_rate,
            reference_sample_rate=self.config.HAAQI_sample_rate,
            audiogram=self.listener.audiogram_right, equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(self.reference_mixture[:, 1])),
        )
        # Calculate the mean HAAQI score for both channels
        haaqi_score = np.mean([left_score, right_score])
        self.current_haaqi_score = haaqi_score
        key_list = list(self.previous_haaqi_score.keys())
        current_key = key_list[self.scene_listener_pair_idx - 1]
        # Calculate the reward as the difference between the current and previous HAAQI scores
        return (haaqi_score - self.previous_haaqi_score[current_key]) * 100000

    def _is_done(self,current_haaqi_score):
        # Determine if the episode is done (e.g., based on the number of processed scenes)
        if self.scene_listener_pair_idx >= 10:  # Example: done condition is processing 10 scenes
            return True
        if self.previous_haaqi_score is not None:
            print("self.previous_haaqi_score:", self.previous_haaqi_score)
            key_list = list(self.previous_haaqi_score.keys())
            current_key = key_list[self.scene_listener_pair_idx - 1]

            target_score = self.previous_haaqi_score[current_key] + self.target_haaqi_increment
            print("target_score:", target_score)
            if current_haaqi_score >= target_score:
                return True
        return False  # Continue the episode if not done

# Function to create a new environment instance
def make_env(audio_file, config):
    def _init():
        env = AudioCompressionEnv(config, audio_file)
        return env
    return _init

# Custom callback class for logging during training
class CustomCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_file = open(f"{self.log_dir}/training_log.csv", mode="w", newline="")
        self.writer = csv.writer(self.log_file)
        # Write header
        self.writer.writerow(["timesteps", "loss", "reward", "value_loss", "policy_loss"])

    def _on_step(self) -> bool:
        # Collect and log information at each step
        self.writer.writerow([self.num_timesteps,self.locals.get("loss"),
                              self.locals.get("rewards"),self.locals.get("value_loss"),
                              self.locals.get("policy_gradient_loss")])
        return True

    def _on_training_end(self) -> None:
        self.log_file.close()

# Main function to train the PPO model
@hydra.main(config_path="", config_name="config")
def train_ppo_module(config):
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    # Create custom audio compression environments for multiple audio files
    audio_files = [0, 1, 2, 3]
    envs = [make_env(audio_file, config) for audio_file in audio_files]
    vec_env = SubprocVecEnv(envs)

    # Use a custom callback to log training information
    callback = CustomCallback(log_dir=log_dir)
    # Create a PPO model with the specified parameters
    model = PPO("MlpPolicy", vec_env, verbose=1, 
                learning_rate=0.0002,      # Reduced learning rate
                n_steps=8,                 # Keep steps the same for frequent updates
                batch_size=4,              # Keep batch size the same
                n_epochs=5,                # Increase number of training epochs
                gamma=0.99, 
                gae_lambda=0.95, 
                clip_range=0.1)            # Increase the clip range
    # Start training the model
    model.learn(total_timesteps=256, callback=callback)
    # Save the trained model
    model.save("ppo_audio_compression_model")
    # Close the environment
    vec_env.close()

if __name__ == "__main__":
    train_ppo_module()
