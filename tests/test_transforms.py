"""Tests for flash-eeg transforms."""
import time
import pytest
import torch
import flash_eeg as feeg


@pytest.fixture
def sample_input():
    return torch.randn(2, 8, 7500)


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


class TestFunctionalAPI:
    def test_spectrogram(self, sample_input):
        out, dt = _timed(feeg.spectrogram, sample_input, compile=False)
        print(f"spectrogram: {dt*1000:.1f}ms")
        assert out.shape == (2, 8, 224, 224)

    def test_spectrogram_custom_size(self, sample_input):
        out, dt = _timed(feeg.spectrogram, sample_input, output_size=128, compile=False)
        print(f"spectrogram(128): {dt*1000:.1f}ms")
        assert out.shape == (2, 8, 128, 128)

    def test_spectrogram_raw(self, sample_input):
        out, dt = _timed(feeg.spectrogram, sample_input, output="raw", compile=False)
        print(f"spectrogram(raw): {dt*1000:.1f}ms, shape={out.shape}")
        assert out.shape[0] == 2 and out.shape[1] == 8
        assert out.ndim == 4

    def test_morlet(self, sample_input):
        out, dt = _timed(feeg.morlet, sample_input, compile=False)
        print(f"morlet: {dt*1000:.1f}ms")
        assert out.shape == (2, 8, 224, 224)

    def test_morlet_raw(self, sample_input):
        out, dt = _timed(feeg.morlet, sample_input, output="raw", compile=False)
        print(f"morlet(raw): {dt*1000:.1f}ms, shape={out.shape}")
        assert out.shape[0] == 2 and out.shape[1] == 8 and out.shape[2] == 50
        assert out.ndim == 4

    def test_connectivity(self, sample_input):
        out, dt = _timed(feeg.connectivity, sample_input, compile=False)
        print(f"connectivity: {dt*1000:.1f}ms")
        assert out.shape == (2, 1, 224, 224)

    def test_reshape(self, sample_input):
        out, dt = _timed(feeg.reshape, sample_input, compile=False)
        print(f"reshape: {dt*1000:.1f}ms")
        assert out.shape == (2, 8, 224, 224)

    def test_bandpass(self, sample_input):
        out, dt = _timed(feeg.bandpass, sample_input, sfreq=250.0, low=1.0, high=50.0)
        print(f"bandpass: {dt*1000:.1f}ms")
        assert out.shape == sample_input.shape


class TestClassAPI:
    def test_spectrogram(self, sample_input):
        spec = feeg.Spectrogram(device='cpu', compile=False)
        out, dt = _timed(spec, sample_input)
        print(f"Spectrogram class: {dt*1000:.1f}ms")
        assert out.shape == (2, 8, 224, 224)

    def test_spectrogram_raw(self, sample_input):
        spec = feeg.Spectrogram(device='cpu', compile=False)
        out = spec(sample_input, output="raw")
        assert out.ndim == 4 and out.shape[0] == 2

    def test_morlet(self, sample_input):
        morlet = feeg.Morlet(device='cpu', n_samples=7500, compile=False)
        out, dt = _timed(morlet, sample_input)
        print(f"Morlet class: {dt*1000:.1f}ms")
        assert out.shape == (2, 8, 224, 224)

    def test_connectivity(self, sample_input):
        conn = feeg.Connectivity(device='cpu', n_samples=7500, compile=False)
        out, dt = _timed(conn, sample_input)
        print(f"Connectivity class: {dt*1000:.1f}ms")
        assert out.shape == (2, 1, 224, 224)

    def test_reshape(self, sample_input):
        reshape = feeg.Reshape(device='cpu', compile=False)
        out, dt = _timed(reshape, sample_input)
        print(f"Reshape class: {dt*1000:.1f}ms")
        assert out.shape == (2, 8, 224, 224)

    def test_nn_module_interface(self, sample_input):
        spec = feeg.Spectrogram(device='cpu', compile=False)
        assert isinstance(spec, torch.nn.Module)
        assert hasattr(spec, 'parameters')


class TestOutputRange:
    def test_spectrogram_normalized(self, sample_input):
        out = feeg.spectrogram(sample_input, compile=False)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_morlet_normalized(self, sample_input):
        out = feeg.morlet(sample_input, compile=False)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_connectivity_normalized(self, sample_input):
        out = feeg.connectivity(sample_input, compile=False)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_raw_not_normalized(self, sample_input):
        out = feeg.spectrogram(sample_input, output="raw", compile=False)
        assert out.min() < 0.0  # dB values are negative for small signals


class TestReshapeFlexible:
    @pytest.mark.parametrize("output_size,block_size", [(224, 32), (256, 32), (128, 16)])
    def test_different_sizes(self, output_size, block_size):
        x = torch.randn(2, 8, 7500)
        out = feeg.reshape(x, output_size=output_size, block_size=block_size, compile=False)
        assert out.shape == (2, 8, output_size, output_size)


class TestBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_spectrogram_batches(self, batch_size):
        x = torch.randn(batch_size, 8, 7500)
        out, dt = _timed(feeg.spectrogram, x, compile=False)
        print(f"spectrogram B={batch_size}: {dt*1000:.1f}ms")
        assert out.shape == (batch_size, 8, 224, 224)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_morlet_batches(self, batch_size):
        x = torch.randn(batch_size, 8, 7500)
        out, dt = _timed(feeg.morlet, x, compile=False)
        print(f"morlet B={batch_size}: {dt*1000:.1f}ms")
        assert out.shape == (batch_size, 8, 224, 224)


class TestCorrectness:
    def test_stft_matches_torch(self):
        """Verify STFT spectrogram matches raw torch.stft output."""
        x = torch.randn(1, 1, 7500)
        raw = feeg.spectrogram(x, output="raw", compile=False)  # [1,1,n_freqs,T_frames]
        # Compare against direct torch.stft
        stft = torch.stft(x.squeeze(), n_fft=1024, hop_length=128,
                          window=torch.hann_window(1024), return_complex=True)
        ref_db = 10.0 * torch.log10(stft.real.pow(2) + stft.imag.pow(2) + 1e-10)
        # Check frequency range matches (bins 5-409 for 1-100Hz at 250Hz/1024fft)
        freq_res = 250.0 / 1024
        bin_min = int((1.0 / freq_res) + 0.999)
        bin_max = int(100.0 / freq_res)
        ref_slice = ref_db[bin_min:bin_max + 1, :]
        assert torch.allclose(raw.squeeze(), ref_slice, atol=1e-5)

    def test_morlet_power_is_smooth(self):
        """Verify Morlet power envelope is smooth (no 2f oscillations)."""
        sfreq, T = 250.0, 7500
        t = torch.arange(T) / sfreq
        # Pure 10 Hz sine
        x = torch.sin(2 * torch.pi * 10 * t).unsqueeze(0).unsqueeze(0)
        raw = feeg.morlet(x, sfreq=sfreq, freq_min=8.0, freq_max=12.0, n_freqs=1,
                          decim=1, output="raw", compile=False)
        power = raw[0, 0, 0, :]  # single freq band power over time
        # After initial transient, power should be nearly constant (smooth envelope)
        steady = power[500:-500]
        variation = (steady.max() - steady.min()) / steady.mean().abs()
        assert variation < 0.1, f"Power envelope too variable: {variation:.3f} (expect < 0.1)"

    def test_bandpass_removes_out_of_band(self):
        """Verify bandpass actually removes frequencies outside the band."""
        sfreq, T = 250.0, 7500
        t = torch.arange(T) / sfreq
        x = (torch.sin(2 * torch.pi * 10 * t) + torch.sin(2 * torch.pi * 80 * t)).unsqueeze(0).unsqueeze(0)
        filtered = feeg.bandpass(x, sfreq=sfreq, low=5.0, high=30.0)
        # Check 80Hz component is gone: FFT and look at 80Hz bin
        fft_out = torch.fft.rfft(filtered.squeeze())
        freqs = torch.fft.rfftfreq(T, d=1.0/sfreq)
        bin_80 = (freqs - 80.0).abs().argmin()
        bin_10 = (freqs - 10.0).abs().argmin()
        assert fft_out[bin_80].abs() < fft_out[bin_10].abs() * 0.01


class TestInputValidation:
    def test_wrong_ndim(self):
        x = torch.randn(8, 7500)  # missing batch dim
        with pytest.raises(AssertionError, match="Expected.*B,C,T"):
            feeg.spectrogram(x, compile=False)

    def test_invalid_output(self, sample_input):
        with pytest.raises(AssertionError, match="output must be"):
            feeg.spectrogram(sample_input, output="Image", compile=False)

    def test_morlet_wrong_samples(self):
        morlet = feeg.Morlet(device='cpu', n_samples=7500, compile=False)
        x = torch.randn(1, 8, 5000)  # wrong length
        with pytest.raises(AssertionError, match="Expected T=7500"):
            morlet(x)

    def test_connectivity_wrong_channels(self):
        conn = feeg.Connectivity(device='cpu', n_channels=8, n_samples=7500, compile=False)
        x = torch.randn(1, 4, 7500)  # wrong channels
        with pytest.raises(AssertionError, match="Expected C=8"):
            conn(x)


class TestFloat16:
    def test_spectrogram_float16(self):
        x = torch.randn(2, 8, 7500).half()
        out = feeg.spectrogram(x, compile=False)
        assert out.dtype == torch.float16
        assert out.shape == (2, 8, 224, 224)

    def test_morlet_float16(self):
        x = torch.randn(2, 8, 7500).half()
        out = feeg.morlet(x, compile=False)
        assert out.dtype == torch.float16
        assert out.shape == (2, 8, 224, 224)

    def test_bandpass_float16(self):
        x = torch.randn(2, 8, 7500).half()
        out = feeg.bandpass(x)
        assert out.dtype == torch.float16
        assert out.shape == x.shape

    def test_reshape_float16(self):
        x = torch.randn(2, 8, 7500).half()
        out = feeg.reshape(x, compile=False)
        assert out.dtype == torch.float16
        assert out.shape == (2, 8, 224, 224)

    def test_bfloat16(self):
        x = torch.randn(2, 8, 7500).bfloat16()
        out = feeg.spectrogram(x, compile=False)
        assert out.dtype == torch.bfloat16
