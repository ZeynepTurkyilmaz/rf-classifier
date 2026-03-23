#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

// Bu fonksiyon Python'dan çağrılacak
// iq_data: numpy array, I ve Q değerleri ardışık [I0, Q0, I1, Q1, ...]
// n_fft: her FFT penceresinin boyutu
// hop_length: pencereler arası adım
py::array_t<double> compute_spectrogram(
    py::array_t<double> iq_data,
    int n_fft = 128,
    int hop_length = 32)
{
    auto buf = iq_data.request();
    double* ptr = static_cast<double*>(buf.ptr);
    int total_samples = buf.size / 2;  // I+Q çiftleri

    int n_frames = (total_samples - n_fft) / hop_length + 1;
    int n_bins = n_fft / 2;

    std::vector<double> result(n_frames * n_bins, 0.0);

    fftw_complex *in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n_fft);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n_fft);
    fftw_plan plan    = fftw_plan_dft_1d(n_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int frame = 0; frame < n_frames; frame++) {
        int offset = frame * hop_length;

        for (int i = 0; i < n_fft; i++) {
            in[i][0] = ptr[(offset + i) * 2];      // I kanalı
            in[i][1] = ptr[(offset + i) * 2 + 1];  // Q kanalı
        }

        fftw_execute(plan);

        for (int i = 0; i < n_bins; i++) {
            double mag = std::sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);
            result[frame * n_bins + i] = 20.0 * std::log10(mag + 1e-10); // dB cinsinden
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    // Python'a 2D numpy array olarak döndür: (n_frames, n_bins)
    return py::array_t<double>({n_frames, n_bins}, result.data());
}

PYBIND11_MODULE(fft_module, m) {
    m.doc() = "FFTW3 tabanlı RF sinyal spektrogram hesaplayıcı";
    m.def("compute_spectrogram", &compute_spectrogram,
          py::arg("iq_data"),
          py::arg("n_fft") = 128,
          py::arg("hop_length") = 32,
          "IQ verisinden spektrogram hesaplar, dB cinsinden döndürür");
}
