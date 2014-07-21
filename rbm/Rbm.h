#ifndef RBM_H
#define RBM_H
class Rbm{
  public:
    Rbm();
    Rbm(float momentum, float alpha, int visibleSize, int hiddenSize);
    Rbm(const Rbm &rbm);
    float computeCostAndGradient(float* &data, int batchSize);//data batchSize*visibleSize
    void updateWeight();
    void train(float* &data, int iter, int batchSize);
    void sigm(float* &data, int batchSize);
    void sigmrnd(float* &data, int batchSize);
    virtual ~Rbm();
    
    int visibleSize;
    int hiddenSize;
    float momentum;
    float alpha;
    float* W;// hiddenSize * visibleSize
    float* b;//visibleSize
    float* c;//hiddenSize
    float* vW;
    float* vb;
    float* vc;
     
    float* h1;
    float* h2;
    float* v2;
    float* c1;
    float* c2;
    float* random;

};
#endif
