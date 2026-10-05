// Brute-force check of opt.hpp: both formulations (benders, classic) must reach the
// exact optimum of the center-identity variant on small random instances.
// Build and run with: make test-opt   (needs Gurobi, see README)
#include "core.hpp"
#include "heuristics.hpp"
#include "fid.hpp"
#include "opt.hpp"
#include <random>
#include <functional>
#include <cstdio>
#include <cstring>
static void combos(int n,int k,int s,std::vector<int>& cur,const std::function<void(const std::vector<int>&)>& f){
  if((int)cur.size()==k){f(cur);return;} for(int j=s;j<n;++j){cur.push_back(j);combos(n,k,j+1,cur,f);cur.pop_back();}}
static double brute(const Points& P,int k,const Clustering* H,int budget){
  double best=std::numeric_limits<double>::infinity(); std::vector<int> cur;
  combos(P.n,k,0,cur,[&](const std::vector<int>& cs){ auto o=order_centers_by_slot(cs,H,k); std::vector<int> a;
    if(assign_with_budget(P,H,o,budget,a)){double c=Clustering(P,o,a).cost(); if(c<best) best=c;}});
  return best;}
static int run_bruteforce(int seed,int ntests,int nmin,int nmax,const char* forms){
  std::mt19937 rng(seed); int fails=0, total=0;
  for(int it=0;it<ntests;++it){
    int n=nmin+rng()%(nmax-nmin+1), k=2+rng()%3; Points P; P.n=n; P.d=2; P.data.resize(n*2);
    std::uniform_real_distribution<double> U(0,10);
    bool clustered = it%2; for(int i=0;i<n;++i){ double cx = clustered ? (i%3)*4.0 : 0; P.data[2*i]=cx+U(rng)*(clustered?0.4:1); P.data[2*i+1]=U(rng)*(clustered?0.4:1);}
    if(it%5==0) for(int i=1;i<n;i+=4){P.data[2*i]=P.data[2*i-2];P.data[2*i+1]=P.data[2*i-1];} // duplicates
    bool with_hist = it%4!=0;
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0); std::shuffle(idx.begin(),idx.end(),rng);
    std::vector<int> hc(idx.begin(),idx.begin()+k); Clustering hist(P,hc,assign_points(P,hc));
    int budget = with_hist ? (int)(rng()%(n/2+1)) : n; const Clustering* H = with_hist ? &hist : nullptr;
    double bf=brute(P,k,H,budget);
    std::mt19937 g(1); Clustering heur=kmedianppwpost(P,k,g);
    for(const char* f : {"benders","classic"}){
      if(!strstr(forms,f)) continue;
      OptInfo info; auto r=opt_lcc_median_center_identity(P,k,H,budget,{heur.centers},120,0,0,info,f);
      double c=r.clustering.cost(); int used=H?center_point_changes(hist,r.clustering):0; ++total;
      bool ok=std::fabs(c-bf)<=1e-6*std::max(1.0,bf)&&info.optimal&&used<=budget&&c>=info.bound-1e-6*std::max(1.0,c);
      if(!ok){++fails; std::printf("FAIL %s n=%d k=%d b=%d hist=%d opt=%.6f brute=%.6f bound=%.6f used=%d status=%d\n",f,n,k,budget,with_hist,c,bf,info.bound,used,info.status);}
      else if(it<3) std::printf("  ok %s n=%d k=%d b=%d hist=%d cost=%.4f lp_rounds=%d cuts=%ld lp_bound=%.4f\n",f,n,k,budget,with_hist,c,info.lp_rounds,info.cuts,info.lp_bound);
    }
  }
  std::printf("%d/%d solves matched brute force\n",total-fails,total); return fails?1:0;}

int main(int argc, char** argv) {
  const char* forms = argc > 1 ? argv[1] : "benders,classic";
  return run_bruteforce(7, 60, 8, 40, forms);
}