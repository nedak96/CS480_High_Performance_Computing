#include<random>
#include<iostream>
#include<cmath>
#include<ctime>
#include<chrono>
#include<omp.h>
#include<string.h>
#include<fstream>
using namespace std;

int main(int argc, char* argv[]){
	auto start = chrono::high_resolution_clock::now();
	vector<vector<float>> res(15, vector<float>());
	int n = 10000000;
	omp_set_num_threads(omp_get_max_threads());
	#pragma omp parallel for shared(res)
	for(int i=2; i<=16; i++){
    vector<float> t(100, 0.0);
    default_random_engine gen1;
    int n1 = 10000000;
    uniform_real_distribution<float> d1(0.0, 1.0);
    for(int j=0; j<n1; j++){
      t[static_cast<int>(pow(d1(gen1), 1.0/i)*100)] += 1.0/n1;
    }
    res[i-2] = t;
	}
	auto end = chrono::high_resolution_clock::now();
	ofstream file;
	file.open("output.txt", ofstream::out | ofstream::trunc);
	for(int i=0; i<15; i++){
    cout<<"Dimensions = "<<i+2<<":"<<endl;
    for(int j=0; j<100; j++){
      file<<res[i][j];
      if (j != 99) file<<" ";
      cout<<"\t"<<j*.01<<"-"<<(j+1)*.01<<": "<<res[i][99-j]<<endl;
    }
    file<<endl;
	}
	file.close();
	cout<<"Time: "<<chrono::duration_cast<chrono::seconds>(end-start).count()<<" s"<<endl;
	return 0;
}
