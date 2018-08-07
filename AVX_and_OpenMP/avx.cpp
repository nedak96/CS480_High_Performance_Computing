#include <immintrin.h>
#include <ctime>
#include <chrono>
#include <random>
#include <iostream>
#include <cmath>

using namespace std;

int main(int argc, char* argv[]){
	default_random_engine gen;
	uniform_real_distribution<float> u(-1.0, 1.0);
	const int n = 50000000;
	alignas(32) static float a[n], b[n], c[n], d[n], e[n], f[n], g[n], h[n];
	for(int i=0; i<n; i++){
		a[i] = u(gen);
		b[i] = u(gen);
		c[i] = u(gen);
		d[i] = u(gen);
		e[i] = u(gen);
		f[i] = u(gen);
		g[i] = u(gen);
		h[i] = u(gen);
	}
	static float l1[n];
	alignas(32) static float l2[n];
	auto start = chrono::high_resolution_clock::now();
	for(int i=0; i<n; i++){
		l1[n] = sqrt(pow(a[i]-b[i], 2)+pow(c[i]-d[i], 2)+pow(e[i]-f[i], 2)+pow(g[i]-h[i], 2));
	}
	auto end1 = chrono::high_resolution_clock::now();
	for(int i=0; i<n/8; i++){
		__m256 a1 = _mm256_load_ps(a+i*8);
		__m256 b1 = _mm256_load_ps(b+i*8);
		__m256 c1 = _mm256_load_ps(c+i*8);
		__m256 d1 = _mm256_load_ps(d+i*8);
		__m256 e1 = _mm256_load_ps(e+i*8);
		__m256 f1 = _mm256_load_ps(f+i*8);
		__m256 g1 = _mm256_load_ps(g+i*8);
		__m256 h1 = _mm256_load_ps(h+i*8);
		__m256 s1 = _mm256_sub_ps(a1, b1);
		__m256 s2 = _mm256_sub_ps(c1, d1);
		__m256 s3 = _mm256_sub_ps(e1, f1);
		__m256 s4 = _mm256_sub_ps(g1, h1);
		__m256 r = _mm256_sqrt_ps(_mm256_mul_ps(s1, s1)+_mm256_mul_ps(s2, s2)+_mm256_mul_ps(s3, s3)+_mm256_mul_ps(s4, s4));
		_mm256_store_ps(l2+8*i, r);
	}
	auto end2 = chrono::high_resolution_clock::now();
	cout<<"Sequential: "<<chrono::duration_cast<chrono::milliseconds>(end1-start).count()<<" ms"<<endl;
	cout<<"With AVX: "<<chrono::duration_cast<chrono::milliseconds>(end2-end1).count()<<" ms"<<endl;
	return 0;
}
