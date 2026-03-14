"""flash-eeg: GPU-accelerated EEG-to-image transforms for large-scale AI training."""
import math
import torch
from torch import nn
from typing import Optional
from functools import lru_cache

__version__ = "0.4.1"
__all__ = ["spectrogram", "morlet", "connectivity", "reshape", "bandpass", "clear_cache",
           "Spectrogram", "Morlet", "Connectivity", "Reshape", "Bandpass"]

_COMPILE_GPUS = ("A100", "H100", "H200", "L40", "RTX 4090", "RTX 5090")


def _should_auto_compile() -> bool:
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name()
    return any(gpu in name for gpu in _COMPILE_GPUS)


def _canonical_device(device_str):
    """Normalize device strings so 'cuda' and 'cuda:0' hash the same in lru_cache."""
    return str(torch.device(device_str))


# --- Cached module constructors ---

@lru_cache(maxsize=32)
def _get_spectrogram_module(sfreq, n_fft, hop_length, freq_min, freq_max,
                             freq_breakpoint, n_freqs_out, output_size, device, compile):
    from .transforms import SpectrogramSTFT
    m = SpectrogramSTFT(sfreq=sfreq, n_fft=n_fft, hop_length=hop_length, freq_min=freq_min,
                         freq_max=freq_max, freq_breakpoint=freq_breakpoint, n_freqs_out=n_freqs_out,
                         target_size=output_size, device=device)
    if compile:
        m = torch.compile(m, mode="max-autotune")
    return m


@lru_cache(maxsize=32)
def _get_morlet_module(sfreq, freq_min, freq_max, n_freqs, n_cycles, n_samples, decim, output_size, device, compile):
    from .transforms import SpectrogramMorlet
    m = SpectrogramMorlet(sfreq=sfreq, freq_min=freq_min, freq_max=freq_max, n_freqs=n_freqs,
                           n_cycles=n_cycles, n_samples=n_samples, decim=decim, target_size=output_size, device=device)
    if compile:
        m = torch.compile(m, mode="max-autotune")
    return m


@lru_cache(maxsize=32)
def _get_connectivity_module(sfreq, n_channels, n_samples, n_fft, NW, num_tapers, output_size, device, compile):
    from .transforms import ConnectivityWPLI
    m = ConnectivityWPLI(sfreq=sfreq, n_channels=n_channels, n_samples=n_samples, n_fft=n_fft,
                          NW=NW, num_tapers=num_tapers, target_size=output_size, device=device)
    if compile:
        m = torch.compile(m, mode="max-autotune")
    return m


@lru_cache(maxsize=32)
def _get_reshape_module(n_samples, n_channels, output_size, block_size, device, compile):
    from .transforms import ReshapeSquare
    m = ReshapeSquare(n_samples=n_samples, n_channels=n_channels, target_size=output_size,
                       block_size=block_size, device=device)
    if compile:
        m = torch.compile(m, mode="max-autotune")
    return m


def clear_cache():
    """Free all cached modules and their GPU memory."""
    _get_spectrogram_module.cache_clear()
    _get_morlet_module.cache_clear()
    _get_connectivity_module.cache_clear()
    _get_reshape_module.cache_clear()


# --- Functional API ---

def spectrogram(x, sfreq=250.0, n_fft=1024, hop_length=128, freq_min=1.0, freq_max=100.0,
                freq_breakpoint=30.0, n_freqs_out=224, output_size=224, output="image", compile=None):
    """STFT spectrogram. [B,C,T] -> [B,C,output_size,output_size] or raw dB with output='raw'."""
    device = _canonical_device(x.device)
    if compile is None:
        compile = _should_auto_compile() and "cuda" in device
    m = _get_spectrogram_module(sfreq, n_fft, hop_length, freq_min, freq_max,
                                 freq_breakpoint, n_freqs_out, output_size, device, compile)
    return m(x, output=output)


def morlet(x, sfreq=250.0, freq_min=1.0, freq_max=100.0, n_freqs=50, n_cycles=None, decim=4,
           output_size=224, output="image", compile=None):
    """Morlet wavelet spectrogram. [B,C,T] -> [B,C,output_size,output_size] or raw dB with output='raw'."""
    device = _canonical_device(x.device)
    n_samples = x.shape[-1]
    if compile is None:
        compile = _should_auto_compile() and "cuda" in device
    m = _get_morlet_module(sfreq, freq_min, freq_max, n_freqs, n_cycles, n_samples, decim, output_size, device, compile)
    return m(x, output=output)


def connectivity(x, sfreq=250.0, n_fft=512, NW=2.0, num_tapers=3, output_size=224, compile=None):
    """WPLI connectivity. [B,C,T] -> [B,1,output_size,output_size]."""
    device = _canonical_device(x.device)
    n_channels, n_samples = x.shape[1], x.shape[2]
    if compile is None:
        compile = _should_auto_compile() and "cuda" in device
    m = _get_connectivity_module(sfreq, n_channels, n_samples, n_fft, NW, num_tapers, output_size, device, compile)
    return m(x)


def reshape(x, output_size=224, block_size=32, compile=None):
    """Signal-to-image reshape. [B,C,T] -> [B,C,output_size,output_size]."""
    device = _canonical_device(x.device)
    n_channels, n_samples = x.shape[1], x.shape[2]
    if compile is None:
        compile = _should_auto_compile() and "cuda" in device
    m = _get_reshape_module(n_samples, n_channels, output_size, block_size, device, compile)
    return m(x)


def bandpass(x, sfreq=250.0, low=1.0, high=50.0, rolloff=2.0):
    """FFT bandpass filter with cosine rolloff. [B,C,T] -> [B,C,T]."""
    orig_dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq, device=x.device)
    mask = torch.ones_like(freqs)
    if rolloff > 0:
        low_rise = torch.clamp((freqs - (low - rolloff)) / rolloff, 0.0, 1.0)
        high_fall = torch.clamp(((high + rolloff) - freqs) / rolloff, 0.0, 1.0)
        mask = 0.5 * (1 - torch.cos(math.pi * low_rise)) * 0.5 * (1 - torch.cos(math.pi * high_fall))
    else:
        mask = ((freqs >= low) & (freqs <= high)).float()
    result = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * mask, n=x.shape[-1], dim=-1)
    return result.to(orig_dtype) if orig_dtype != torch.float32 else result


# --- Class API (nn.Module) ---

class Spectrogram(nn.Module):
    """Reusable STFT spectrogram transform."""
    def __init__(self, sfreq=250.0, n_fft=1024, hop_length=128, freq_min=1.0, freq_max=100.0,
                 freq_breakpoint=30.0, n_freqs_out=224, output_size=224, device="cuda", compile=None):
        super().__init__()
        if compile is None:
            compile = _should_auto_compile() and "cuda" in device
        from .transforms import SpectrogramSTFT
        self.module = SpectrogramSTFT(sfreq=sfreq, n_fft=n_fft, hop_length=hop_length, freq_min=freq_min,
                                       freq_max=freq_max, freq_breakpoint=freq_breakpoint,
                                       n_freqs_out=n_freqs_out, target_size=output_size, device=device)
        if compile:
            self.module = torch.compile(self.module, mode="max-autotune")

    def forward(self, x, output="image"):
        return self.module(x, output=output)


class Morlet(nn.Module):
    """Reusable Morlet wavelet spectrogram transform."""
    def __init__(self, sfreq=250.0, freq_min=1.0, freq_max=100.0, n_freqs=50, n_cycles=None,
                 n_samples=7500, decim=4, output_size=224, device="cuda", compile=None):
        super().__init__()
        if compile is None:
            compile = _should_auto_compile() and "cuda" in device
        from .transforms import SpectrogramMorlet
        self.module = SpectrogramMorlet(sfreq=sfreq, freq_min=freq_min, freq_max=freq_max, n_freqs=n_freqs,
                                         n_cycles=n_cycles, n_samples=n_samples, decim=decim, target_size=output_size, device=device)
        if compile:
            self.module = torch.compile(self.module, mode="max-autotune")

    def forward(self, x, output="image"):
        return self.module(x, output=output)


class Connectivity(nn.Module):
    """Reusable WPLI connectivity transform."""
    def __init__(self, sfreq=250.0, n_channels=8, n_samples=7500, n_fft=512,
                 NW=2.0, num_tapers=3, output_size=224, device="cuda", compile=None):
        super().__init__()
        if compile is None:
            compile = _should_auto_compile() and "cuda" in device
        from .transforms import ConnectivityWPLI
        self.module = ConnectivityWPLI(sfreq=sfreq, n_channels=n_channels, n_samples=n_samples, n_fft=n_fft,
                                        NW=NW, num_tapers=num_tapers, target_size=output_size, device=device)
        if compile:
            self.module = torch.compile(self.module, mode="max-autotune")

    def forward(self, x):
        return self.module(x)


class Reshape(nn.Module):
    """Reusable signal-to-image reshape transform."""
    def __init__(self, n_samples=7500, n_channels=8, output_size=224, block_size=32, device="cuda", compile=None):
        super().__init__()
        if compile is None:
            compile = _should_auto_compile() and "cuda" in device
        from .transforms import ReshapeSquare
        self.module = ReshapeSquare(n_samples=n_samples, n_channels=n_channels, target_size=output_size,
                                     block_size=block_size, device=device)
        if compile:
            self.module = torch.compile(self.module, mode="max-autotune")

    def forward(self, x):
        return self.module(x)


class Bandpass(nn.Module):
    """Reusable FFT bandpass filter with precomputed frequency mask."""
    def __init__(self, n_samples, sfreq=250.0, low=1.0, high=50.0, rolloff=2.0, device="cuda"):
        super().__init__()
        freqs = torch.fft.rfftfreq(n_samples, d=1.0 / sfreq, device=device)
        if rolloff > 0:
            low_rise = torch.clamp((freqs - (low - rolloff)) / rolloff, 0.0, 1.0)
            high_fall = torch.clamp(((high + rolloff) - freqs) / rolloff, 0.0, 1.0)
            mask = 0.5 * (1 - torch.cos(math.pi * low_rise)) * 0.5 * (1 - torch.cos(math.pi * high_fall))
        else:
            mask = ((freqs >= low) & (freqs <= high)).float()
        self.register_buffer("mask", mask)
        self.n_samples = n_samples

    def forward(self, x):
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * self.mask, n=self.n_samples, dim=-1)
