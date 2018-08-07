#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

//runs the neural network
__global__
void forward(float *inp, unsigned char* labels, float *weights, float *bias, float *correct){
	int ans;
	*correct = 0;
	float out[10];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int n = index; n < 10000; n += stride){
		ans = 0;
		for(int i=0; i<10; i++){
		 out[i] = 0;
		 for(int j=0; j<784; j++){
			 out[i] += weights[i*784+j]*inp[n*784+j];
		 }
		 out[i] += bias[i];
		 if(out[i] > out[ans]) ans = i;
		}
		if(ans == (int)labels[n]) atomicAdd(correct,1);
	}
}

//trains the neural network
__global__
void train(float *inp, unsigned char *labels, int *shuffled, float *wd, float *bd, float *weights, float *bias){
	int index = threadIdx.x;
  int stride = blockDim.x;
	for(int z=index; z<100; z+=stride){
		float t[10];
		float t1[10];
		float t2[10];
		float t3[10];
		//fully connected forward
	  for(int i=0; i<10; i++){
	 	 t[i] = 0;
	 	 for(int j=0; j<784; j++) t[i] += weights[i*784+j]*inp[j+shuffled[z]*784];
	 	 t[i] += bias[i];
	  }
	  //softmax forward
	  float m = 0;
	  for(int i=0; i<10; i++){
	 	 if(t[i]>m) m = t[i];
	  }
	  float sum = 0;
	  for(int i=0; i<10; i++){
	 	 t1[i] = exp(t[i]-m);
	 	 sum += t1[i];
	  }
	  for(int i=0; i<10; i++) t1[i] = t1[i]/sum;
	  //cross entropy
	  for(int i=0; i<10; i++) t2[i] = 0;
	  t2[(int)labels[shuffled[z]]] = -1/t1[(int)labels[shuffled[z]]];
	  //softmax backprop
	  for(int i=0; i<10; i++){
	 	 t3[i] = 0;
	 	 for(int j=0; j<10; j++){
	 		 if(i == j) t3[i] += t2[j]*t1[i]*(1-t1[j]);
	 		 else t3[i] += t2[j]*t1[j]*(-t1[i]);
	 	 }
	  }
	  //fully connected backprop
	  for(int i=0; i<10; i++){
	 	 for(int j=0; j<784; j++) atomicAdd(&wd[i*784+j], t3[i]*inp[j+shuffled[z]*784]/100);
	 	 atomicAdd(&bd[i], t3[i]/100);
	  }
	}
}

int main(){
 int temp;
 unsigned char temp1;
 //import training images
 ifstream fin("train-images.idx3-ubyte", ios::binary);
 fin.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 fin.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 fin.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 fin.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 float *training;
 cudaMallocManaged(&training, 47040000*sizeof(float));
 for(size_t i = 0; i<47040000; i++){
	 fin.read(reinterpret_cast<char*>(&temp1), sizeof(unsigned char));
	 training[i] = float(temp1)/127.5-1;
 }
 fin.close();
 //import training labels
 ifstream fin2("train-labels.idx1-ubyte", ios::binary);
 fin2.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 fin2.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 unsigned char *trainingl;
 cudaMallocManaged(&trainingl, 60000*sizeof(unsigned char));
 for(size_t i = 0; i<60000; i++){
	 fin2.read(reinterpret_cast<char*>(&trainingl[i]), sizeof(unsigned char));
 }
 fin2.close();
 //import testing images
 ifstream fin1("t10k-images.idx3-ubyte", ios::binary);
 fin1.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 fin1.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 fin1.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 fin1.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 float *testing;
 cudaMallocManaged(&testing, 7840000*sizeof(float));
 for(size_t i = 0; i<7840000; i++){
	 fin1.read(reinterpret_cast<char*>(&temp1), sizeof(unsigned char));
	 testing[i] = float(temp1)/127.5-1;
 }
 fin1.close();
 //import testing labels
 ifstream fin3("t10k-labels.idx1-ubyte", ios::binary);
 fin3.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 fin3.read(reinterpret_cast<char*>(&temp), sizeof(temp));
 unsigned char * testingl;
 cudaMallocManaged(&testingl, 10000*sizeof(unsigned char));
 for(size_t i = 0; i<10000; i++){
	 fin3.read(reinterpret_cast<char*>(&testingl[i]), sizeof(unsigned char));
 }
 fin3.close();
 float *weights;
 float *bias;
 float *wd;
 float *bd;
 cudaMallocManaged(&weights, 7840*sizeof(float));
 cudaMallocManaged(&bias, 10*sizeof(float));
 cudaMallocManaged(&wd, 7840*sizeof(float));
 cudaMallocManaged(&bd, 10*sizeof(float));
 std::normal_distribution<float> init;
 std::default_random_engine m_eng(100);
 //initialize weights to random numbers and bias to 0
 for(int i=0; i<7840; i++) weights[i] = init(m_eng)/28;
 for(int i=0; i<10; i++) bias[i] = 0;
 int *shuffled;
 cudaMallocManaged(&shuffled, 60000*sizeof(int));
 for(int i=0; i<60000; i++) shuffled[i] = i;
 float *temper;
 cudaMallocManaged(&temper, sizeof(float));
 forward<<<50,200>>>(testing, testingl, weights, bias, temper);
 cudaDeviceSynchronize();
 cout<<"Initial: "<<*temper/10000<<endl;
 for(int e=0; e<30; e++){
	 //shuffle training data
	 random_shuffle(&shuffled[0], &shuffled[59999]);
	 for(int i=0;i<600; i++){
		 //set weight and bias derivatives to 0
		 for(int j=0; j<7840; j++) wd[j] = 0;
 		 for(int j=0; j<10; j++) bd[j] = 0;
		 //train the weights and bias
		 train<<<1,100>>>(training, trainingl, shuffled+100*i, wd, bd, weights, bias);
		 cudaDeviceSynchronize();
		 //update weights and bias
		 for(int i=0; i<10; i++){
	 	 	 for(int j=0; j<784; j++) weights[i*784+j] -= .001*wd[i*784+j];
	 	 	 bias[i] -= .001*bd[i];
	 	  }
		}
		//run and print
	 forward<<<50,200>>>(testing, testingl, weights, bias, temper);
	 cudaDeviceSynchronize();
	 cout<<"Epoch "<<e+1<<": "<<*temper/10000<<endl;
 }
 cudaFree(shuffled);
 cudaFree(trainingl);
 cudaFree(testingl);
 cudaFree(training);
 cudaFree(testing);
 cudaFree(weights);
 cudaFree(bias);
 cudaFree(wd);
 cudaFree(bd);
 cudaFree(temper);
 return 0;
}
