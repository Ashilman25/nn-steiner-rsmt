from __future__ import annotations
import bisect
import hashlib
import json
import math
import random
from pathlib import Path


def validate_tree(tree):
    pts = [tuple(p) for p in tree['points']]
    edges = [tuple(e) for e in tree['edges']]
    nt = int(tree['num_terminals'])
    if not 1 < nt <= len(pts) or len(edges) != len(pts)-1:
        raise ValueError('Expected a tree with at least two terminals.')
    
    if any(len(p)!=2 or any(not math.isfinite(x) for x in p) for p in pts):
        raise ValueError('Invalid coordinates.')
    
    adj = [[] for _ in pts]
    for a,b in edges:
        if not (0 <= a < len(pts) and 0 <= b < len(pts)) or a==b:
            raise ValueError('Invalid edge.')
        if pts[a]==pts[b]:
            raise ValueError('Zero-length electrical edge.')
        adj[a].append(b); adj[b].append(a)
        
    seen={0}; todo=[0]
    for a in todo:
        for b in adj[a]:
            if b not in seen: seen.add(b);todo.append(b)

    if len(seen)!=len(pts):raise ValueError('Disconnected graph.')
    return pts,edges,adj


def segments(tree, bend='HV'):
    pts,edges,_ = validate_tree(tree)
    if bend not in ('HV','VH'):raise ValueError('bend must be HV or VH')
    out=[]
    
    for a,b in edges:
        x,y=pts[a];u,v=pts[b]
        corner=(u,y) if bend=='HV' else (x,v)
        for p,q in (((x,y),corner),(corner,(u,v))):
            if p!=q:out.append((p,q))

    return out


#Fixed synthetic background demand, not derived from NN predictions
def scenario(seed=42, bins=16, extent=10000.0):
    if bins<2 or extent<=0:raise ValueError('Invalid routing grid.')
    rng=random.Random(seed)
    cap=[[[4.0 for _ in range(bins)] for _ in range(bins)] for _ in range(2)]
    bg=[[[1.0+0.4*rng.random() for _ in range(bins)] for _ in range(bins)] for _ in range(2)]
    
    # A busy central corridor makes congestion-aware choices meaningful.
    for axis in range(2):
        for iy in range(bins):
            for ix in range(bins):
                if bins//3 <= ix < 2*bins//3 and bins//4 <= iy < 3*bins//4:
                    bg[axis][iy][ix]=3.7+0.4*rng.random()
                    
    return {
        'seed':seed,'bins':bins,'extent':extent,'capacity':cap,'background':bg,
        'r_per_unit':1e-4,'c_per_unit':1e-4,'driver_r':1.0,'sink_c':1.0,
        'driver_terminal':0,'units':'arbitrary; delay is an RC proxy, not seconds'
    }


#Directional segment length / bin width: track-equivalent density proxy
def wire_density(tree, sc, bend='HV'):
    n=sc['bins'];extent=sc['extent'];width=extent/n
    demand=[[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(2)]
    
    for (x,y),(u,v) in segments(tree,bend):
        if any(z<0 or z>extent for z in (x,y,u,v)):raise ValueError('Route outside scenario extent.')
        
        axis=0 if y==v else 1
        lo,hi=sorted((x,u) if axis==0 else (y,v))
        fixed=y if axis==0 else x
        row=min(n-1,int(fixed/width))
        
        for k in range(n):
            overlap=max(0.0,min(hi,(k+1)*width)-max(lo,k*width))
            if overlap:
                iy,ix=(row,k) if axis==0 else (k,row)
                demand[axis][iy][ix]+=overlap/width

    return demand


def congestion(tree,sc,bend='HV'):
    demand=wire_density(tree,sc,bend)
    incremental=0.0;peak=0.0;density=0.0
    for axis in range(2):
        for iy in range(sc['bins']):
            for ix in range(sc['bins']):
                cap=sc['capacity'][axis][iy][ix]
                bg=sc['background'][axis][iy][ix]
                d=demand[axis][iy][ix]
                if cap<=0 or bg<0:raise ValueError('Capacities must be positive and background nonnegative.')
                incremental+=max(0.0,bg+d-cap)**2-max(0.0,bg-cap)**2
                peak=max(peak,(bg+d)/cap)
                density+=d
                
    return {
        'congestion_proxy':incremental,
        'peak_utilization_proxy':peak,
        'density_sum':density
    }


#Worst-sink Elmore delay on the upstream tree, with uniform distributed RC.
def delay(tree,sc):
    pts,edges,adj=validate_tree(tree)
    nt=tree['num_terminals'];root=sc['driver_terminal']
    if not 0<=root<nt:raise ValueError('Driver must be a terminal index.')
    if any(sc[k]<0 for k in ('r_per_unit','c_per_unit','driver_r','sink_c')):
        raise ValueError('Negative electrical parameter.')
    
    order=[root];parent={root:None};length={};distance={root:0.0}
    for a in order:
        for b in adj[a]:
            if b==parent[a]:continue
            parent[b]=a;order.append(b)
            ell=abs(pts[a][0]-pts[b][0])+abs(pts[a][1]-pts[b][1])
            length[b]=ell;distance[b]=distance[a]+ell
            
    capacitance={i:(sc['sink_c'] if i<nt and i!=root else 0.0) for i in order}
    for b in reversed(order[1:]):
        capacitance[parent[b]]+=capacitance[b]+sc['c_per_unit']*length[b]
        
    arrival={root:sc['driver_r']*capacitance[root]}
    for b in order[1:]:
        ell=length[b]
        arrival[b]=arrival[parent[b]]+sc['r_per_unit']*ell*(capacitance[b]+0.5*sc['c_per_unit']*ell)
        
    sinks=[i for i in range(nt) if i!=root]
    return {
        'delay_proxy':max(arrival[i] for i in sinks),
        'worst_driver_sink_path':max(distance[i] for i in sinks)
    }


def metrics(tree,sc,bend='HV'):
    pts,edges,_=validate_tree(tree)
    wl=sum(abs(pts[a][0]-pts[b][0])+abs(pts[a][1]-pts[b][1]) for a,b in edges)
    return {
        'wirelength':wl,
        **congestion(tree,sc,bend),
        **delay(tree,sc)
    }


def scales(baseline,sc):
    # Fixed per-instance baseline units shared across all candidates and policies.
    return {
        'wirelength':max(baseline['wirelength'],1e-12),
        'congestion_proxy':max(baseline['density_sum'],1.0),
        'delay_proxy':max(baseline['delay_proxy'],1e-12)
    }


def objective(measurement,normalizers,weights):
    if len(weights)!=3 or any(not math.isfinite(w) or w<0 for w in weights) or sum(weights)<=0:
        raise ValueError('Three nonnegative weights, at least one positive, required.')
    
    keys=('wirelength','congestion_proxy','delay_proxy')
    return sum(w*measurement[k]/normalizers[k] for w,k in zip(weights,keys))


def choose(candidates,sc,weights,baseline_id='turn0-t0.9-HV',wirelength_budget=0.10):
    baseline=next(c for c in candidates if c['id']==baseline_id)
    normalizers=scales(baseline['metrics'],sc)
    limit=baseline['metrics']['wirelength']*(1+wirelength_budget)
    eligible=[c for c in candidates if c['metrics']['wirelength']<=limit+1e-8]
    
    # Deterministic ties; baseline remains eligible.
    best=min(eligible,key=lambda c:(objective(c['metrics'],normalizers,weights),c['id']))
    return best,objective(best['metrics'],normalizers,weights),normalizers


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        
    return h.hexdigest()


def turn(k,x,y,n=9999):
    return [(x,y),(n-y,x),(n-x,n-y),(y,n-x),
            (n-x,y),(x,n-y),(y,x),(n-y,n-x)][k]


def unturn(k,x,y,n=9999):
    return turn([0,3,2,1,4,5,6,7][k],x,y,n)


#Load released m15/kb4 checkpoint and generate NN proposals, weights frozen
def generate_candidates(repo,point_file,model_path,thresholds=(0.7,0.9,0.95),turns=(0,1,2,3),cache=None):
    import sys,time
    repo=Path(repo);upstream=repo/'third_party/NN-Steiner'
    if str(upstream) not in sys.path:sys.path.insert(0,str(upstream))
    import torch
    from omegaconf import OmegaConf
    from arora.models import NNArora
    from arora.data import get_transform
    from arora.flow.utils import get_quadtree
    from arora.solver import NNSteiner
    from arora.points.getPoint import read_points
    
    if 0 not in turns or 0.9 not in thresholds:raise ValueError('Include original baseline turn 0, threshold 0.9.')
    args=OmegaConf.load(upstream/'conf/config.yaml')
    args.quadtree.m=15;args.quadtree.kb=4
    args.quadtree.bbox.width=10000;args.quadtree.bbox.height=10000
    
    device='cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    network=NNArora(args.model,args.quadtree)
    state=torch.load(model_path,map_location='cpu',weights_only=True)
    
    network.load_state_dict({k.replace('module.',''):v for k,v in state.items()})
    network.eval()
    
    for p in network.parameters():p.requires_grad_(False)
    solver=NNSteiner(network,get_transform('mlp',torch.float32),device)
    points=read_points(str(point_file));candidates=[]
    
    for k in turns:
        rotated=[turn(k,x,y) for x,y in points]
        for threshold in thresholds:
            qt=get_quadtree(rotated,args.quadtree)
            _,_,stt=solver.solve(qt,float(threshold),10,1)
            stt.cleanup();stt.check()
            
            tree={
                'points':[unturn(k,p.x.to_float(),p.y.to_float()) for p in stt.points],
                'edges':[(e.src,e.tgt) for e in stt.edges], 
                'num_terminals':stt.num_terminals
            }
            
            validate_tree(tree)
            if len(points)!=tree['num_terminals'] or any(
                abs(tree['points'][i][0]-p[0])+abs(tree['points'][i][1]-p[1])>1e-5
                for i,p in enumerate(points)):
                raise ValueError('Original terminal correspondence was not preserved.')
            
            for bend in ('HV','VH'):
                candidates.append({'id':f'turn{k}-t{threshold}-{bend}','tree':tree,'bend':bend})
    return candidates
