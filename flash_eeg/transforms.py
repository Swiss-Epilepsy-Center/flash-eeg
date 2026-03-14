"""Internal transform modules. Use flash_eeg.spectrogram/morlet/connectivity/reshape instead."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def _dpss(n, NW, num_tapers):
    """Compute DPSS tapers in pure torch (no scipy)."""
    diag = ((n - 1 - 2 * torch.arange(n, dtype=torch.float64)) / 2.0) ** 2 * math.cos(2 * math.pi * NW / n)
    off_diag = torch.arange(1, n, dtype=torch.float64) * torch.arange(n - 1, 0, -1, dtype=torch.float64) / 2.0
    T = torch.diag(diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    tapers = eigenvectors[:, -num_tapers:].flip(1).T
    for i in range(num_tapers):
        if tapers[i].sum() < 0:
            tapers[i] *= -1
    return tapers.float()


def _ensure_float32(x):
    """Upcast to float32 for FFT ops, return (x_f32, original_dtype)."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float(), x.dtype
    return x, None


def _restore_dtype(x, orig_dtype):
    """Cast back to original dtype if it was upcast."""
    return x.to(orig_dtype) if orig_dtype is not None else x


class SpectrogramSTFT(nn.Module):
    def __init__(self, sfreq=250.0, n_fft=1024, hop_length=128, freq_min=1.0, freq_max=100.0,
                 freq_breakpoint=30.0, n_freqs_out=224, target_size=224, device="cuda", dtype=torch.float32):
        super().__init__()
        self.n_fft, self.hop_length, self.n_freqs_out, self.target_size = n_fft, hop_length, n_freqs_out, target_size
        self.register_buffer("window", torch.hann_window(n_fft, device=device, dtype=torch.float32))

        freq_resolution = sfreq / n_fft
        self.bin_min = int((freq_min / freq_resolution) + 0.999)
        self.bin_max = int(freq_max / freq_resolution)
        self.n_freqs_orig = self.bin_max - self.bin_min + 1

        scaled_positions = torch.linspace(0, 2, n_freqs_out, device=device, dtype=torch.float32)
        freqs_target = torch.empty_like(scaled_positions)
        mask1 = scaled_positions < 1.0
        if mask1.any():
            log_ratio = torch.log10(torch.tensor(freq_breakpoint / freq_min, device=device, dtype=torch.float32))
            freqs_target[mask1] = freq_min * torch.pow(10.0, scaled_positions[mask1] * log_ratio)
        mask2 = ~mask1
        if mask2.any():
            freqs_target[mask2] = freq_breakpoint + (scaled_positions[mask2] - 1.0) * (freq_max - freq_breakpoint)

        indices = (freqs_target - freq_min) / (freq_max - freq_min) * (self.n_freqs_orig - 1)
        self.register_buffer("indices_floor", torch.clamp(torch.floor(indices).long(), 0, self.n_freqs_orig - 1))
        self.register_buffer("indices_ceil", torch.clamp(torch.ceil(indices).long(), 0, self.n_freqs_orig - 1))
        self.register_buffer("indices_frac", (indices - self.indices_floor.float()))

    def forward(self, x, output="image"):
        assert x.ndim == 3, f"Expected [B,C,T] input, got shape {x.shape}"
        assert output in ("image", "raw"), f"output must be 'image' or 'raw', got '{output}'"
        x, orig_dtype = _ensure_float32(x)
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)
        stft = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window, return_complex=True, center=True, normalized=False)
        spec_db = 10.0 * torch.log10(stft.real.pow(2) + stft.imag.pow(2) + 1e-10)
        spec_db_filtered = spec_db[:, self.bin_min:self.bin_max + 1, :]

        if output == "raw":
            return _restore_dtype(spec_db_filtered.reshape(B, C, self.n_freqs_orig, -1), orig_dtype)

        BC, _, T_frames = spec_db_filtered.shape
        spec_db_t = spec_db_filtered.permute(0, 2, 1)
        indices_floor_exp = self.indices_floor.view(1, 1, -1).expand(BC, T_frames, -1)
        indices_ceil_exp = self.indices_ceil.view(1, 1, -1).expand(BC, T_frames, -1)
        frac_exp = self.indices_frac.view(1, 1, -1).expand(BC, T_frames, -1)

        vals_low = torch.gather(spec_db_t, dim=2, index=indices_floor_exp)
        vals_high = torch.gather(spec_db_t, dim=2, index=indices_ceil_exp)
        spec_resampled = ((1 - frac_exp) * vals_low + frac_exp * vals_high).permute(0, 2, 1)
        spec_resampled = spec_resampled.reshape(B, C, self.n_freqs_out, T_frames)

        spec_flat = spec_resampled.reshape(B, C, -1)
        spec_min, spec_max = torch.aminmax(spec_flat, dim=2, keepdim=True)
        spec_min, spec_max = spec_min.unsqueeze(-1), spec_max.unsqueeze(-1)
        spec_norm = (spec_resampled - spec_min) / (spec_max - spec_min + 1e-8)
        result = F.interpolate(spec_norm, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return _restore_dtype(result, orig_dtype)


class SpectrogramMorlet(nn.Module):
    def __init__(self, sfreq=250.0, freq_min=1.0, freq_max=100.0, n_freqs=50,
                 n_cycles=None, n_samples=7500, decim=4, target_size=224, device="cuda", dtype=torch.float32):
        super().__init__()
        self.n_freqs, self.decim, self.target_size, self.n_samples = n_freqs, decim, target_size, n_samples

        freqs = torch.logspace(math.log10(freq_min), math.log10(freq_max), n_freqs, device=device, dtype=torch.float32)
        n_cycles = freqs / 2.0 if n_cycles is None else torch.full_like(freqs, float(n_cycles))
        # Build wavelets on rfft frequencies (positive only) for analytic signal
        rfft_freqs = torch.fft.rfftfreq(n_samples, d=1.0/sfreq, device=device)
        sigma_f = freqs / n_cycles
        f_diff = rfft_freqs.unsqueeze(0) - freqs.unsqueeze(1)
        wavelets_rfft = torch.exp(-0.5 * (f_diff / sigma_f.unsqueeze(1)) ** 2)
        wavelets_rfft = wavelets_rfft / wavelets_rfft.norm(dim=1, keepdim=True)
        self.register_buffer("wavelets_rfft", wavelets_rfft)

    def forward(self, x, output="image"):
        assert x.ndim == 3, f"Expected [B,C,T] input, got shape {x.shape}"
        assert output in ("image", "raw"), f"output must be 'image' or 'raw', got '{output}'"
        assert x.shape[-1] == self.n_samples, f"Expected T={self.n_samples}, got T={x.shape[-1]}"
        x, orig_dtype = _ensure_float32(x)
        B, C, T = x.shape
        # rfft -> multiply in half-spectrum -> construct analytic signal via irfft trick
        x_rfft = torch.fft.rfft(x, dim=-1)  # [B, C, T//2+1]
        conv_rfft = x_rfft.unsqueeze(2) * self.wavelets_rfft.unsqueeze(0).unsqueeze(0)  # [B, C, n_freqs, T//2+1]
        # Analytic signal: zero negative freqs, double positive freqs, ifft
        n_rfft = conv_rfft.shape[-1]
        analytic_fft = torch.zeros(*conv_rfft.shape[:-1], T, dtype=torch.cfloat, device=x.device)
        analytic_fft[..., :n_rfft] = conv_rfft
        analytic_fft[..., 1:n_rfft-1] *= 2  # double positive freqs (exclude DC and Nyquist)
        conv = torch.fft.ifft(analytic_fft, dim=-1)
        power = conv.real.pow(2) + conv.imag.pow(2)
        power = power[:, :, :, ::self.decim]
        power_db = 10.0 * torch.log10(power + 1e-10)

        if output == "raw":
            return _restore_dtype(power_db, orig_dtype)

        flat = power_db.reshape(B, C, -1)
        p_min, p_max = torch.aminmax(flat, dim=2, keepdim=True)
        p_min, p_max = p_min.unsqueeze(-1), p_max.unsqueeze(-1)
        power_norm = (power_db - p_min) / (p_max - p_min + 1e-8)
        result = F.interpolate(power_norm, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return _restore_dtype(result, orig_dtype)


class ConnectivityWPLI(nn.Module):
    def __init__(self, sfreq=250.0, n_channels=8, n_samples=7500, n_fft=512,
                 NW=2.0, num_tapers=3, target_size=224, device="cuda", dtype=torch.float32):
        super().__init__()
        self.target_size, self.n_channels, self.n_fft = target_size, n_channels, n_fft

        tapers = _dpss(n_samples, NW, num_tapers)
        self.register_buffer("tapers", tapers.to(device=device, dtype=torch.float32))

        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sfreq)
        bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 60), (60, 100), (1, 40), (1, 100)]
        for i, (fmin, fmax) in enumerate(bands):
            self.register_buffer(f"band_idx_{i}", torch.where((freqs >= fmin) & (freqs <= fmax))[0].to(device))
        self.register_buffer("diag_mask", torch.eye(n_channels, dtype=torch.bool, device=device))

    def _wpli_band(self, csd_imag, band_indices):
        csd_band = torch.index_select(csd_imag, -1, band_indices)
        num = csd_band.mean(dim=-1).abs()
        den = csd_band.abs().mean(dim=-1)
        wpli = num / (den + 1e-10)
        wpli_flat = wpli.reshape(wpli.shape[0], -1)
        wmin, wmax = torch.aminmax(wpli_flat, dim=1, keepdim=True)
        wmin, wmax = wmin.unsqueeze(-1), wmax.unsqueeze(-1)
        return (wpli - wmin) / (wmax - wmin + 1e-8)

    def _combine_bands(self, upper, lower):
        combined = torch.triu(upper.transpose(-2, -1), diagonal=0) + torch.tril(lower, diagonal=-1)
        return torch.where(self.diag_mask.unsqueeze(0), torch.ones_like(combined), combined)

    def forward(self, signal):
        assert signal.ndim == 3, f"Expected [B,C,T] input, got shape {signal.shape}"
        assert signal.shape[1] == self.n_channels, f"Expected C={self.n_channels}, got C={signal.shape[1]}"
        signal, orig_dtype = _ensure_float32(signal)
        B, C, T = signal.shape
        tapered = signal.unsqueeze(2) * self.tapers.unsqueeze(0).unsqueeze(0)
        fft_result = torch.fft.rfft(tapered, n=self.n_fft, dim=-1)
        csd = (fft_result.unsqueeze(2) * fft_result.unsqueeze(1).conj()).mean(dim=3)
        csd_imag = csd.imag

        w = [self._wpli_band(csd_imag, getattr(self, f"band_idx_{i}")) for i in range(8)]
        c0 = self._combine_bands(w[0], w[1])
        c1 = self._combine_bands(w[2], w[3])
        c2 = self._combine_bands(w[4], w[5])
        c3 = self._combine_bands(w[6], w[7])

        top = torch.cat([c0, c1], dim=2)
        bot = torch.cat([c2, c3], dim=2)
        stacked = torch.cat([top, bot], dim=1).unsqueeze(1)
        result = F.interpolate(stacked, size=(self.target_size, self.target_size), mode="nearest")
        return _restore_dtype(result, orig_dtype)


class ReshapeSquare(nn.Module):
    def __init__(self, n_samples=7500, n_channels=8, target_size=224, block_size=32, device="cuda"):
        super().__init__()
        self.square_size = int(n_samples ** 0.5)
        self.n_keep = self.square_size ** 2
        self.target_size, self.block_size = target_size, block_size
        self.n_blocks = target_size // block_size
        perm = torch.randperm(self.n_blocks, generator=torch.Generator().manual_seed(42))
        self.register_buffer("shuffle_perm", perm.to(device))

    def forward(self, x):
        assert x.ndim == 3, f"Expected [B,C,T] input, got shape {x.shape}"
        x, orig_dtype = _ensure_float32(x)
        B, C, T = x.shape
        x_sq = x[:, :, :self.n_keep].reshape(B, C, self.square_size, self.square_size)
        x_flat = x_sq.reshape(B, C, -1)
        x_min, x_max = torch.aminmax(x_flat, dim=2, keepdim=True)
        x_min, x_max = x_min.unsqueeze(-1), x_max.unsqueeze(-1)
        x_norm = (x_sq - x_min) / (x_max - x_min + 1e-8)
        x_resized = F.interpolate(x_norm, size=(self.target_size, self.target_size), mode="nearest")
        x_blocks = x_resized.reshape(B, C, self.n_blocks, self.block_size, self.target_size)
        x_shuffled = x_blocks[:, :, self.shuffle_perm, :, :]
        result = x_shuffled.reshape(B, C, self.target_size, self.target_size)
        return _restore_dtype(result, orig_dtype)
