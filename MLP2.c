// MLP2 - TWO-LAYER MULTILAYER PERCEPTRON - SAMPLE IMPLEMENTATION (WITH THE XOR DATA)
// Version 2.1 ------------------------------------------ (c) R.Jaksa 2009,2020 GPLv3

#define Nin 2		// no. of inputs
#define Nh1 2		// no. of hidden units
#define Nou 1		// no. of outputs
#define Gamma 0.2	// learning rate
#define Epochs 40000	// no. of training epochs (cycles)

// ------------------------------------------------------------- END OF CONFIGURATION
#include <math.h>	// fabs, exp
#include <stdlib.h>	// rand, srand
#include <stdio.h>	// printf
#include <sys/timeb.h>	// ftime

#define Nx (1+Nin+Nh1+Nou) // no. of units
#define IN1 1		// 1st input
#define INn Nin		// last (n-th) input
#define H11 (Nin+1)	// 1st hidden
#define H1n (Nin+Nh1)	// last hidden
#define OU1 (H1n+1)	// 1st output
#define OUn (H1n+Nou)	// last output

typedef struct {	//
  double x[Nx];		// units inputs
  double y[Nx];		// units activations
  double delta[Nx];	// units delta signal
  double w[Nx][Nx];	// weights (single weights matrix)
  double dv[Nou];	// desired value on output
} ann_t;		//

#define w(i,j)	ann->w[i][j]
#define x(i)	ann->x[i]
#define y(i)	ann->y[i]
#define delta(i) ann->delta[i]
#define dv(i)	ann->dv[i-OU1]

// block of units: for function definitions
#define blk_t(ann) ann_t *ann, int i1, int in, int j1, int jn
// block of units: for function calls
#define blk(BLKi,BLKj) BLKi##1,BLKi##n,BLKj##1,BLKj##n
// cycle through the layer: OU or H1 or IN
#define forlayer(BLK,i) for(int i=BLK##1; i<=BLK##n; i++)

// --------------------------------------- ACTIVATION FUNCTION AND ITS 1st DERIVATION
#define af(X)	(1.0/(1.0+exp((-1.0)*X)))
#define df(X)	(exp((-1.0)*X)/((1.0+exp((-1.0)*X))*(1.0+exp((-1.0)*X))))

// -------------------------------------------------------------- RANDOM WEIGHTS INIT
void ann_rndinit(ann_t *ann, double min, double max) {	//
  y(0)=-1.0;						// the input for bias
  for(int i=0; i<Nx; i++) for(int j=0; j<Nx; j++)	//
    w(i,j) = rand() / (RAND_MAX/(max-min)) + min; }	//

// ----------------------------------------------------------------- SINGLE LAYER RUN
static void layer_run(blk_t(ann)) {			// output/input block from-to
  for(int i=i1; i<=in; i++) {				// for every output
    x(i) = w(i,0) * y(0);				// add bias contribution
    for(int j=j1; j<=jn; j++) x(i) += w(i,j) * y(j);	// add main inputs contrib.
    y(i) = af(x(i)); }}					// apply activation function

// ---------------------------------------------------------------------- NETWORK RUN
void MLP2_run(ann_t *ann) {				//
  layer_run(ann,blk(H1,IN));				// in -> h1
  layer_run(ann,blk(OU,H1)); }				// h1 -> ou

// ------------------------------------------------------ SINGLE LAYER WEIGHTS UPDATE
static void layer_update(blk_t(ann),double gamma) {	//
  for(int i=i1; i<=in; i++) {				//
    w(i,0) += gamma * delta(i) * y(0);			// bias (weight) update
    for(int j=j1; j<=jn; j++)				//
      w(i,j) += gamma * delta(i) * y(j); }}		// the weights update

// ------------------------------------------------- VANILLA BACKPROPAGATION LEARNING
void MLP2_vanilla_bp(ann_t *ann, double gamma) {	//
  MLP2_run(ann);					// 1st run the network
  forlayer(OU,i) delta(i) = (dv(i)-y(i)) * df(x(i));	// delta on output layer
  forlayer(H1,i) {					//
    double S=0.0; forlayer(OU,h) S += delta(h) * w(h,i);
    delta(i) = S * df(x(i)); }				// delta on hidden layer
  layer_update(ann,blk(OU,H1),gamma);			// h1 -> ou
  layer_update(ann,blk(H1,IN),gamma); }			// in -> h1

// ----------------------------------------------------------------------------- MAIN
int XOR[4][3] = {{0,0,0,},{0,1,1,},{1,0,1,},{1,1,0,}};	// XOR data
int main(void) {					//
  ann_t ann[1];						//
  struct timeb t; ftime(&t); srand(t.time);		// time-seed random generator 
  ann_rndinit(ann,-0.1,0.1);				// initialize the network
  printf("\nEpoch:  Output  Des.out. (Error)\n");	//
  printf("--------------------------------\n");		//
  ftime(&t); long t1=t.time*1000+t.millitm;		// start time in milliseconds
  for(int epoch=0; epoch<=Epochs; epoch++) {		// for every epoch
    for(int p=0; p<4; p++) {				// for every pattern
      y(IN1)  =XOR[p][0];				// input 1 (XOR data)
      y(IN1+1)=XOR[p][1];				// input 2 (XOR data)
      dv(OU1) =XOR[p][2];				// desired output (XOR data)
      MLP2_vanilla_bp(ann,Gamma);			// train
      if(epoch%5000==0) {				// every 5000 ep. print error
	if(p==0 && epoch!=0) printf("\n");		//
	MLP2_run(ann);					// run network
	double J=fabs(dv(OU1) - y(OU1));		// compute the error
	printf("%5d: %f %f (%.3f)\n",epoch,y(OU1),dv(OU1),J); }}}
  ftime(&t); long t2=t.time*1000+t.millitm;		// end time (in milliseconds)
  printf("--------------------------------\n");		//
  long con=((Nin+1)*Nh1+(Nh1+1)*Nou)*4*Epochs;		// no. of connect. updated
  int msec=t2-t1; if(!msec) msec=1;			// time in milliseconds
  printf("%ld kCUPS in %.3f sec\n\n",con/msec,(double)msec/1000.0);
  return(0); }						//

// ------------------------------------------------------------------------------ END
