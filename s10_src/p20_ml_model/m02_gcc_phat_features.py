import torch
import torch.fft
import torch.nn as nn
import math


class GCCPHATFeatures(nn.Module):
    def __init__(self, n_mics: int, fs: int = 16000, n_samples_in_frame: int = 4096,
                 max_tau: float = 0.001, interp_factor: int = 4, eps: float = 1e-8):
        """
        Computes GCC-PHAT features for batches of multi-channel audio.

        Parameters:
        - n_mics: Number of microphones.
        - fs: Sampling rate in Hz.
        - n_samples_in_frame: Number of samples in each input audio frame.
        - max_tau: Maximum expected time delay in seconds (determines window size).
        - interp_factor: Upsampling factor for cross-correlation (L in the paper).
        - eps: Small epsilon for numerical stability in PHAT.
        """
        super().__init__()
        self.n_mics = n_mics
        self.fs = fs
        self.n_samples = n_samples_in_frame
        self.max_tau = max_tau
        self.interp_factor = interp_factor
        self.eps = eps

        # n_fft for the initial FFT of signals
        # The paper suggests "frames of 256 ms" which is 0.256 * 16000 = 4096 samples.
        # Using 2*n_samples for n_fft is a common practice for linear convolution via FFT.
        self.n_fft = 2 * self.n_samples  # Or a power of 2 like next_power_of_2(self.n_samples)

        # n_fft for IFFT, after upsampling in frequency domain
        self.n_fft_upsampled = self.n_fft * self.interp_factor

        # Calculate the number of samples to keep from the GCC function (N_c in paper)
        # This corresponds to the lags from -max_tau to +max_tau
        # max_shift_samples is half the window length in samples, at the upsampled rate
        self.max_shift_samples = math.ceil(self.max_tau * self.fs * self.interp_factor)
        self.gcc_feature_length = 2 * self.max_shift_samples + 1  # N_c

        # Precompute microphone pairs
        pairs = []
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                pairs.append((i, j))
        self.pairs = torch.tensor(pairs)  # Shape: [n_pairs, 2]
        self.n_pairs = self.pairs.shape[0]

        # Ensure pairs tensor is on the correct device when forward is called
        # self.register_buffer('pairs_buf', self.pairs) # Better way if no gradient needed

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        - signals: [batch_size, n_mics, n_samples] time-domain signals

        Returns:
        - gcc_features: [batch_size, n_pairs, gcc_feature_length]
        """
        if signals.shape[1] != self.n_mics or signals.shape[2] != self.n_samples:
            raise ValueError(
                f"Input signal shape mismatch. Expected [*, {self.n_mics}, {self.n_samples}], got {signals.shape}")

        device = signals.device
        self.pairs = self.pairs.to(device)  # Ensure pairs are on the same device

        batch_size, _, _ = signals.shape

        # FFTs of all signals in the batch
        # signals: [batch_size, n_mics, n_samples]
        # SIG: [batch_size, n_mics, n_fft_half] (if rfft) or [batch_size, n_mics, n_fft] (if fft)
        # Let's use fft for complex conjugate symmetry handling ease
        SIG = torch.fft.fft(signals, n=self.n_fft, dim=-1)  # [batch_size, n_mics, n_fft]

        # Select signals for pairs
        # SIG_i, SIG_j: [batch_size, n_pairs, n_fft]
        SIG_i = SIG[:, self.pairs[:, 0], :]
        SIG_j = SIG[:, self.pairs[:, 1], :]

        # Cross-power spectrum
        R = SIG_i * torch.conj(SIG_j)  # [batch_size, n_pairs, n_fft]

        # PHAT Transform
        R_phat = R / (torch.abs(R) + self.eps)  # [batch_size, n_pairs, n_fft]

        # Inverse FFT to get cross-correlation
        # Upsampling is achieved by specifying a larger n for ifft (zero-padding in freq)
        cc = torch.fft.ifft(R_phat, n=self.n_fft_upsampled, dim=-1).real  # [batch_size, n_pairs, n_fft_upsampled]

        # Shift zero-lag to center
        cc = torch.fft.fftshift(cc, dim=-1)  # [batch_size, n_pairs, n_fft_upsampled]

        # Windowing: Extract the central part of the GCC [-max_tau, max_tau]
        mid_point = cc.shape[-1] // 2

        # Ensure windowing indices are valid
        start_idx = mid_point - self.max_shift_samples
        end_idx = mid_point + self.max_shift_samples + 1  # Slicing is exclusive at end

        if start_idx < 0 or end_idx > cc.shape[-1]:
            # This can happen if max_tau is too large for the n_fft_upsampled
            # Or if n_samples is very small.
            # Fallback: take the whole thing if window is too large (should not happen with typical params)
            # Or, more robustly, clip to available range if Nc is larger than n_fft_upsampled
            # However, with typical parameters, self.gcc_feature_length <= self.n_fft_upsampled
            print(
                f"Warning: GCC window exceeds IFFT length. Clamping. Mid: {mid_point}, Max_shift: {self.max_shift_samples}, IFFT len: {cc.shape[-1]}")
            actual_gcc_len = cc.shape[-1]
            if actual_gcc_len < self.gcc_feature_length:
                # Pad if the actual computed GCC is shorter than expected (e.g. due to very small n_fft_upsampled)
                padding_needed = self.gcc_feature_length - actual_gcc_len
                pad_left = padding_needed // 2
                pad_right = padding_needed - pad_left
                cc_window = F.pad(cc, (pad_left, pad_right))
            else:  # Trim if it's somehow longer (shouldn't be due to fftshift)
                cc_window = cc[:, :, start_idx:end_idx]

        else:
            cc_window = cc[:, :, start_idx:end_idx]

        if cc_window.shape[-1] != self.gcc_feature_length:
            # This can happen if max_shift_samples makes the window slightly off due to rounding
            # Or if n_fft_upsampled is odd vs even in combination with window size
            # A common strategy is to ensure self.gcc_feature_length is what you get.
            # For example, by adjusting the window slightly or padding/trimming.
            # Given our +1, it should be correct. Let's assert.
            # Example: if mid_point=100, max_shift=10, window is 90 to 110 (21 samples)
            # start=90, end=111. cc[:,:,90:111] gives 21 samples. Correct.

            # If it's off by one due to n_fft_upsampled being odd/even and how fftshift works
            # with the mid_point calculation, we might need to adjust.
            # Let's check the shape and pad/trim if necessary to meet self.gcc_feature_length
            current_len = cc_window.shape[-1]
            if current_len < self.gcc_feature_length:
                diff = self.gcc_feature_length - current_len
                # Pad equally on both sides if possible
                pad_left = diff // 2
                pad_right = diff - pad_left
                cc_window = torch.nn.functional.pad(cc_window, (pad_left, pad_right))
            elif current_len > self.gcc_feature_length:
                # Trim from both sides if possible
                trim_left = (current_len - self.gcc_feature_length) // 2
                trim_right = (current_len - self.gcc_feature_length) - trim_left
                cc_window = cc_window[:, :, trim_left: -trim_right if trim_right > 0 else current_len]

        # Final check for the feature length
        if cc_window.shape[-1] != self.gcc_feature_length:
            raise ValueError(
                f"Final GCC feature length mismatch. Expected {self.gcc_feature_length}, got {cc_window.shape[-1]}")

        return cc_window  # [batch_size, n_pairs, gcc_feature_length]
