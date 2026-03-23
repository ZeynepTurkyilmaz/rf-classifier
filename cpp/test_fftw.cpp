#include <iostream>
#include <fftw3.h>
#include <cmath>

int main() {
    int N = 8;
    
    // FFTW için giriş ve çıkış dizileri
    fftw_complex *in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    // Plan oluştur: N noktalı, ileri yönde FFT
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // Test verisi: basit bir sinüs dalgası
    for (int i = 0; i < N; i++) {
        in[i][0] = std::cos(2.0 * M_PI * i / N);  // gerçel kısım (I)
        in[i][1] = 0.0;                              // sanal kısım (Q)
    }
    
    // FFT hesapla
    fftw_execute(plan);
    
    // Sonuçları yazdır
    std::cout << "FFT sonuçları (büyüklük):" << std::endl;
    for (int i = 0; i < N; i++) {
        double magnitude = std::sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);
        std::cout << "Frekans " << i << ": " << magnitude << std::endl;
    }
    
    // Belleği temizle
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    
    return 0;
}
