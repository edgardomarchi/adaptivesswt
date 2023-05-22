import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pywt


def frequency_with_energy_percentage(signal, energy_percentage, sample_rate):
    """
    Returns the frequency in Hz where the signal's spectrum accumulates the specified
    percentage of energy. The signal should be a numpy array.
    The energy percentage should be normalized, between 0 and 1.
    The sample rate should be specified in Hz.
    """
    # Compute the spectrum of the signal
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    # Compute the magnitude of the spectrum
    spectrum_magnitude = np.abs(spectrum)
    # Compute the total energy of the signal
    total_energy = np.sum(spectrum_magnitude)
    # Compute the cumulative energy up to each frequency
    cumulative_energy = np.cumsum(spectrum_magnitude)
    # Normalize the cumulative energy to be between 0 and 1
    normalized_cumulative_energy = cumulative_energy / total_energy
    # Find the frequency where the accumulated energy exceeds the desired threshold
    index = np.argmax(normalized_cumulative_energy >= energy_percentage)
    # Convert the index to a frequency in Hz
    frequency = (index * sample_rate / len(signal)) - sample_rate/2
    return frequency


if __name__ == '__main__':
    plt.close('all')
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 10}

    matplotlib.rc('font', **font)

    fs = 2000  #Hz

    wcf=1
    wbw=1

    wav = pywt.ContinuousWavelet(f'cmor{wbw}-{wcf}')
    wav.lower_bound = -16
    wav.upper_bound = 16

    precision = 8
    wavefun, x = wav.wavefun(level=precision)

    fig, axes = plt.subplots(2,1)
    fig.suptitle('Wavelet function')
    axes[0].plot(x, wavefun.real)
    axes[0].plot(x, wavefun.imag)

    epsilon_r = 0.2
    xi_1 = frequency_with_energy_percentage(wavefun, epsilon_r/2, fs)
    xi_2 = frequency_with_energy_percentage(wavefun, 1-(epsilon_r/2), fs)

    axes[1].set_title('Spectrum of wavelet function')
    f = np.linspace(-fs/2, fs/2, wavefun.shape[0])
    mw_fft = np.fft.fftshift(np.fft.fft(wavefun))
    axes[1].plot(f,abs(mw_fft))
    axes[1].axvline(xi_1, color='red')
    axes[1].axvline(xi_2, color='red')

    epsilon_array = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    print(f'Last e_r = {epsilon_array[-1]}')
    K = np.zeros_like(epsilon_array)
    delta_w = np.zeros_like(epsilon_array)
    w_array = np.linspace(fs/20, fs/10, 10)

    K_fig, K_ax = plt.subplots(1,1)

    data_fixed_w = {r'\epsilon_r':epsilon_array,
                    r'\Delta\omega_{min}':np.zeros_like(epsilon_array),
                    'K':np.zeros_like(epsilon_array)}

    for j, w in enumerate(w_array):
        for i, e_r in enumerate(epsilon_array):
            xi_1 = frequency_with_energy_percentage(wavefun, e_r/2, fs)
            xi_2 = frequency_with_energy_percentage(wavefun, 1-(e_r/2), fs)

            K[i] = 10*np.log10(2) / (np.log10(xi_2)-np.log10(xi_1))
            delta_w[i] = w*(xi_2/xi_1 - 1)
            if w==fs/20:
                data_fixed_w[r'\Delta\omega_{min}'][i]=delta_w[i]
                data_fixed_w['K'][i]=K[i]

        K_ax.semilogx(delta_w, K, label=f'$\omega$={w:.2f}')

    K_ax.legend()
    K_ax.set_xlabel(r'$\Delta \omega$')
    K_ax.set_ylabel('K')

    print(data_fixed_w)

    plt.show()
