#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <future>
#include <random>
#include <deque>
#include <ctime>
#include <chrono>

using namespace std;

class Node{
public:
	int axis;
  float n;
  Node* left;
  Node* right;
	vector<float> p;
	Node(){
		left = nullptr;
		right = nullptr;
	}
	~Node(){
		delete left;
		delete right;
	}
};

//Find distance of vector
float dist(vector<float> v1, vector<float> v2){
	float sum = 0;
	for(int i=0; i<v1.size(); i++) sum += pow((v1[i]-v2[i]),2);
	return sqrt(sum);
}

//Insert value into sorted array
void putitin(vector<float> p, float d, vector<vector<float>> &closest){
	p.push_back(d);
	auto it = lower_bound(closest.begin(), closest.end(), p, [](vector<float> v1, vector<float> v2){return v2[v2.size()-1]<v1[v1.size()-1];});
	closest.insert(it, p);
}

//Makes tree
void makeTree(Node* n, vector<vector<float>> &v, int c, uint64_t nd){
	if (v.size() == 1){
		n->axis = c;
		n->n = v[0][c];
		n->p = v[0];
		return;
	}
	if(v.size() == 2){
		n->axis = c;
		n->n = v[1][c];
		n->p = v[1];
		n->left = new Node();
		n->left->axis = (c+1)%nd;
		n->left->n = v[0][(c+1)%nd];
		n->left->p = v[0];
		return;
	}
  if (v.size()<1000){
    sort(v.begin(), v.end(), [c](vector<float> v1, vector<float> v2){return v1[c]<v2[c];});
		int ind = ceil((v.size()-1)/2);
    float piv = v[ind][c];
		n->left = new Node();
		n->right = new Node();
		n->axis = c;
		n->n = piv;
		n->p = v[ind];
    vector<vector<float>> lv(v.begin(), v.begin()+ind);
    vector<vector<float>> hv(v.begin()+ind+1, v.end());
    c = (c+1)%nd;
    makeTree(n->left, lv, c, nd);
    makeTree(n->right, hv, c, nd);
    return;
  }
  vector<float> temp(1000);
  for(int i=0; i<1000; i++){
    temp[i] = v[rand()%v.size()][c];
  }
  sort(temp.begin(), temp.end());
  float piv = temp[500];
  auto it = partition(v.begin(), v.end(), [&v,c,piv](vector<float> a){
		if (a[c] == piv) swap(v[v.size()-1], a);
		return a[c] < piv;
	});
	n->left = new Node();
	n->right = new Node();
	n->axis = c;
	n->n = piv;
	n->p = *(v.end()-1);
  vector<vector<float>> lv(v.begin(), it);
  vector<vector<float>> hv(it, v.end()-1);
  c = (c+1)%nd;
	makeTree(n->left, lv, c, nd);
	makeTree(n->right, hv, c, nd);
}

//Nearest-Neighbor Search
void nns(bool found, Node &n, vector<float> point, vector<vector<float>> &closest, uint64_t numPs){
	if(!found){
		if(n.left == nullptr && n.right==nullptr){
			closest.push_back(n.p);
			closest[0].push_back(dist(point, n.p));
			return;
		}
		if(n.right == nullptr){
			nns(false, *(n.left), point, closest, numPs);
			if (abs(n.n-point[n.axis]) < closest[0][closest[0].size()-1] || closest.size() < numPs){
				if(closest.size() < numPs){
					putitin(n.p, dist(point, n.p), closest);
				}
				else if(dist(point, n.p) < closest[0][closest[0].size()-1]){
					closest.erase(closest.begin());
					putitin(n.p, dist(point, n.p), closest);
				}
			}
			return;
		}
		if(point[n.axis] > n.n){
			nns(false, *(n.right), point, closest, numPs);
			if(point[n.axis]-n.n < closest[0][closest[0].size()-1] || closest.size() < numPs){
				if(closest.size() < numPs){
					putitin(n.p, dist(point, n.p), closest);
				}
				else if(dist(point, n.p) < closest[0][closest[0].size()-1]){
					closest.erase(closest.begin());
					putitin(n.p, dist(point, n.p), closest);
				}
				nns(true, *(n.left), point, closest, numPs);
			}
			return;
		}
		else{
			nns(false, *(n.left), point, closest, numPs);
			if (n.n-point[n.axis] < closest[0][closest[0].size()-1] || closest.size() < numPs){
				if(closest.size() < numPs){
					putitin(n.p, dist(point, n.p), closest);
				}
				else if(dist(point, n.p) < closest[0][closest[0].size()-1]){
					closest.erase(closest.begin());
					putitin(n.p, dist(point, n.p), closest);
				}
				nns(true, *(n.right), point, closest, numPs);
			}
			return;
		}
	}
	if (abs(n.n-point[n.axis]) < closest[0][closest[0].size()-1] || closest.size() < numPs){
		if(closest.size() < numPs){
			putitin(n.p, dist(point, n.p), closest);
		}
		else if(dist(point, n.p) < closest[0][closest[0].size()-1]){
			closest.erase(closest.begin());
			putitin(n.p, dist(point, n.p), closest);
		}
		if(n.left == nullptr && n.right==nullptr) return;
		else if(n.right == nullptr){
			nns(true, *(n.left), point, closest, numPs);
			return;
		}
		else{
			nns(true, *(n.right), point, closest, numPs);
			nns(true, *(n.left), point, closest, numPs);
		}
	}
}

//NNS helper function
void nns_helper(vector<vector<float>> &queries, vector<float> &retVect, Node &n, uint64_t numPs){
	for(int i=0; i<queries.size(); i++){
		vector<vector<float>> closest;
		nns(false, n, queries[i], closest, numPs);
		for(int j=closest.size()-1; j>=0; j--){
			for(int k=0; k<closest[j].size()-1; k++){
				retVect.push_back(closest[j][k]);
			}
		}
	}
}

int main(int argc, char* argv[]){
	int nThreads = std::thread::hardware_concurrency();
	uint64_t id, n, nd;
	char* type = new char[8];
  ifstream fin(argv[1], ios::binary);
	fin.read(type, sizeof(type));
  fin.read(reinterpret_cast<char*>(&id), sizeof(id));
  fin.read(reinterpret_cast<char*>(&n), sizeof(n));
  fin.read(reinterpret_cast<char*>(&nd), sizeof(nd));
  vector<vector<float>> points(n, vector<float>(nd));
  for(size_t i = 0; i<n; i++){
    for(size_t j=0; j<nd; j++){
      fin.read(reinterpret_cast<char*>(&points[i][j]), sizeof(float));
    }
  }
	fin.close();
	auto start1 = chrono::high_resolution_clock::now();
	Node root;
	deque<int> cs;
	deque<vector<vector<float>>> pointss;
	deque<Node*> nodes;
	cs.push_back(0);
	pointss.push_back(points);
	nodes.push_back(&root);
	while(cs.size() < nThreads || cs.size() == 0){
		int c = cs.front();
		cs.pop_front();
		vector<vector<float>> v = pointss.front();
		pointss.pop_front();
		Node* n = nodes.front();
		nodes.pop_front();
		if (v.size() == 1){
			n->axis = c;
			n->n = v[0][c];
			n->p = v[0];
		}
		else if(v.size() == 2){
			n->axis = c;
			n->n = v[1][c];
			n->p = v[1];
			n->left = new Node();
			n->left->axis = (c+1)%nd;
			n->left->n = v[0][(c+1)%nd];
			n->left->p = v[0];
		}
		else if (v.size()<1000){
	    sort(v.begin(), v.end(), [c](vector<float> v1, vector<float> v2){return v1[c]<v2[c];});
			int ind = ceil((v.size()-1)/2);
	    float piv = v[ind][c];
			n->left = new Node();
			n->right = new Node();
			n->axis = c;
			n->n = piv;
			n->p = v[ind];
	    vector<vector<float>> lv(v.begin(), v.begin()+ind);
	    vector<vector<float>> hv(v.begin()+ind+1, v.end());
	    c = (c+1)%nd;
			cs.push_back(c);
			cs.push_back(c);
			nodes.push_back(n->left);
			nodes.push_back(n->right);
			pointss.push_back(lv);
			pointss.push_back(hv);
		}
		else{
			vector<float> temp(1000);
		  for(int i=0; i<1000; i++){
		    temp[i] = v[rand()%v.size()][c];
		  }
		  sort(temp.begin(), temp.end());
		  float piv = temp[500];
		  auto it = partition(v.begin(), v.end(), [&v,c,piv](vector<float> a){
				if (a[c] == piv) swap(v[v.size()-1], a);
				return a[c] < piv;
			});
			n->left = new Node();
			n->right = new Node();
			n->axis = c;
			n->n = piv;
			n->p = *(v.end()-1);
		  vector<vector<float>> lv(v.begin(), it);
		  vector<vector<float>> hv(it, v.end()-1);
		  c = (c+1)%nd;
			cs.push_back(c);
			cs.push_back(c);
			nodes.push_back(n->left);
			nodes.push_back(n->right);
			pointss.push_back(lv);
			pointss.push_back(hv);
		}
	}
	vector<thread> threads1;
	for(int i=0; i<cs.size(); i++){
		threads1.push_back(thread(makeTree, nodes[i], ref(pointss[i]), cs[i], nd));
	}
	for(int i=0; i<threads1.size(); i++){
		threads1[i].join();
	}
	auto end1 = chrono::high_resolution_clock::now();
	uint64_t qid, qn, qnd, qnn;
	char* qtype = new char[8];
  ifstream qfin(argv[2], ios::binary);
	qfin.read(qtype, sizeof(qtype));
  qfin.read(reinterpret_cast<char*>(&qid), sizeof(qid));
  qfin.read(reinterpret_cast<char*>(&qn), sizeof(qn));
  qfin.read(reinterpret_cast<char*>(&qnd), sizeof(qnd));
	qfin.read(reinterpret_cast<char*>(&qnn), sizeof(qnn));
	vector<vector<float>> queries(qn, vector<float>(qnd));
	for(size_t i = 0; i<qn; i++){
    for(size_t j=0; j<qnd; j++){
      qfin.read(reinterpret_cast<char*>(&queries[i][j]), sizeof(float));
    }
  }
	qfin.close();
	char rtype[8] = "RESULT";
	uniform_int_distribution<int> d(10000000, 100000000);
	random_device rd("/dev/urandom");
	uint64_t rid = d(rd);
	ofstream fout(argv[3], ios::binary);
	fout.write(rtype, sizeof(rtype));
	fout.write(reinterpret_cast<const char*>(&id), sizeof(id));
	fout.write(reinterpret_cast<const char*>(&qid), sizeof(qid));
	fout.write(reinterpret_cast<const char*>(&rid), sizeof(rid));
	fout.write(reinterpret_cast<const char*>(&qn), sizeof(qn));
	fout.write(reinterpret_cast<const char*>(&nd), sizeof(nd));
	fout.write(reinterpret_cast<const char*>(&qnn), sizeof(qnn));
	auto start2 = chrono::high_resolution_clock::now();
	vector<vector<float>> returns(nThreads);
	vector<thread> threads(nThreads);
	vector<vector<vector<float>>> qdivs(nThreads);
	for(int i=0; i<nThreads; i++){
		vector<vector<float>> tvect(queries.begin()+(i/nThreads*qn), queries.begin()+((i+1)/nThreads*qn));
		qdivs[i] = tvect;
	}
	for(int i=0; i<nThreads; i++){
		threads[i] = thread(nns_helper, ref(qdivs[i]), ref(returns[i]), ref(root), qnn);
	}
	for(int i=0; i<nThreads; i++){
		threads[i].join();
	}
	auto end2 = chrono::high_resolution_clock::now();
	cout<<"Tree Time (ms): "<<chrono::duration_cast<chrono::milliseconds>(end1-start1).count()<<endl;
	cout<<"Queries Time (ms): "<<chrono::duration_cast<chrono::milliseconds>(end2-start2).count()<<endl;
	for(int i=0; i<nThreads; i++){
		for(int j=0; j<returns[i].size(); j++){
			fout.write(reinterpret_cast<const char*>(&returns[i][j]), sizeof(float));
		}
	}
	delete [] type;
	delete [] qtype;
	fout.close();
  return 0;
}
