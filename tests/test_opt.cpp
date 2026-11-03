// Brute-force check of the exact algorithms in opt.hpp on small random instances:
// opt_kmedian, opt_fid (benders + classic) and opt_mc must reach the brute-force optimum,
// and opt_mc <= opt_fid, opt_mc <= FORM, opt_fid <= FID (when FID stays within its budget).
// Build and run with: make test-opt   (needs Gurobi, see README)
#include "core.hpp"
#include "heuristics.hpp"
#include "form.hpp"
#include "fid.hpp"
#include "opt.hpp"
#include <random>
#include <functional>
#include <cstdio>
#include <cstring>
static void combos(int n,int k,int s,std::vector<int>& cur,const std::function<void(const std::vector<int>&)>& f){
  if((int)cur.size()==k){f(cur);return;} for(int j=s;j<n;++j){cur.push_back(j);combos(n,k,j+1,cur,f);cur.pop_back();}}
static double brute_fid(const Points& P,int k,const Clustering* H,int budget){
  double best=std::numeric_limits<double>::infinity(); std::vector<int> cur;
  combos(P.n,k,0,cur,[&](const std::vector<int>& cs){ auto o=order_centers_by_slot(cs,H,k); std::vector<int> a;
    if(assign_with_budget(P,H,o,budget,a)){double c=Clustering(P,o,a).cost(); if(c<best) best=c;}});
  return best;}
static double brute_mc(const Points& P,int k,const Clustering& H,int budget){
  double best=std::numeric_limits<double>::infinity(); std::vector<int> cur;
  combos(P.n,k,0,cur,[&](const std::vector<int>& cs){ std::vector<int> a,m;
    double c=mc_assign_with_budget(P,H,cs,budget,a,m); if(c<best) best=c;});
  return best;}
static bool close(double a,double b){return std::fabs(a-b)<=1e-6*std::max(1.0,std::fabs(b));}
static int run_all(int seed,int ntests,int nmin,int nmax){
  std::mt19937 rng(seed); int fails=0,total=0,order_viol=0,fid_over=0;
  for(int it=0;it<ntests;++it){
    int n=nmin+rng()%(nmax-nmin+1), k=2+rng()%3; Points P; P.n=n; P.d=2; P.data.resize(n*2);
    std::uniform_real_distribution<double> U(0,10); bool cl=it%2;
    for(int i=0;i<n;++i){ double cx=cl?(i%3)*4.0:0; P.data[2*i]=cx+U(rng)*(cl?0.4:1); P.data[2*i+1]=U(rng)*(cl?0.4:1);}
    if(it%5==0) for(int i=1;i<n;i+=4){P.data[2*i]=P.data[2*i-2];P.data[2*i+1]=P.data[2*i-1];}
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0); std::shuffle(idx.begin(),idx.end(),rng);
    std::vector<int> hc(idx.begin(),idx.begin()+k); Clustering hist(P,hc,assign_points(P,hc));
    int budget=(int)(rng()%(n/2+1));
    std::mt19937 g(1); Clustering heur=kmedianppwpost(P,k,g);
    std::vector<std::vector<int>> starts={hist.centers,heur.centers};
    auto check=[&](const char* what,double got,double want,const OptInfo& info,bool budget_ok){ ++total;
      bool ok=close(got,want)&&info.optimal&&budget_ok&&got>=info.bound-1e-6*std::max(1.0,got);
      if(!ok){++fails;std::printf("FAIL %s n=%d k=%d b=%d got=%.6f want=%.6f bound=%.6f status=%d budget_ok=%d\n",what,n,k,budget,got,want,info.bound,info.status,budget_ok);}};
    if(it%4==0){ // k-median without history
      double bf=brute_fid(P,k,nullptr,n);
      for(const char* f:{"benders","classic"}){OptInfo info; auto r=opt_kmedian(P,k,{heur.centers},120,0,0,info,f); check(f,r.clustering.cost(),bf,info,true);}
      continue;}
    double bf=brute_fid(P,k,&hist,budget), bm=brute_mc(P,k,hist,budget);
    double cfid=0;
    for(const char* f:{"benders","classic"}){OptInfo info; auto r=opt_fid(P,k,hist,budget,starts,120,0,0,info,f);
      cfid=r.clustering.cost(); check(f,cfid,bf,info,center_point_changes(hist,r.clustering)<=budget);}
    OptInfo im; auto rm=opt_mc(P,k,hist,budget,starts,120,0,0,im);
    double cmc=rm.clustering.cost(); int used=cluster_distance(hist,rm.clustering,rm.mapping);
    check("opt_mc",cmc,bm,im,used<=budget && used==im.budget_used);
    // orderings: opt_mc <= opt_fid, opt_mc <= FORM, opt_fid <= FID (if FID within the variant budget)
    auto form=poly_lcc_median_fct(P,k,hist,budget,heur);
    auto fid=poly_lcc_median_f_id(P,k,hist,budget,heur);
    if(cmc>cfid+1e-6||cmc>form.clustering.cost()+1e-6){++order_viol;std::printf("ORDER mc=%.4f fid_opt=%.4f form=%.4f\n",cmc,cfid,form.clustering.cost());}
    if(center_point_changes(hist,fid.clustering)>budget) ++fid_over;
    else if(cfid>fid.clustering.cost()+1e-6){++order_viol;std::printf("ORDER fid_opt=%.4f > fid=%.4f\n",cfid,fid.clustering.cost());}
    if(it<4) std::printf("  n=%d k=%d b=%d: opt_mc=%.4f (lp %d rounds, %ld cuts) <= opt_fid=%.4f; FORM=%.4f FID=%.4f\n",
                         n,k,budget,cmc,im.lp_rounds,im.cuts,cfid,form.clustering.cost(),fid.clustering.cost());
  }
  std::printf("\n%d/%d solves matched brute force; ordering violations: %d; FID over variant budget (skipped): %d\n",total-fails,total,order_viol,fid_over);
  return fails||order_viol;}

int main() { return run_all(7, 60, 8, 30); }