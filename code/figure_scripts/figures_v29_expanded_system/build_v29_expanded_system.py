#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import ast, math, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'code/_rebuild_outputs/figures_v29_expanded_system'
PANELS = OUT / 'panels'; COMP = OUT / 'composites'; PREVIEWS = OUT / 'previews'; SRC = OUT / 'source_data'
for d in [PANELS, COMP, PREVIEWS, SRC]: d.mkdir(parents=True, exist_ok=True)

INK = '#252321'; TEXT = '#4f4a44'; MUTED = '#827b72'; GRID = '#eee9de'
LAND = '#fbfaf7'; SAND = '#f2d890'; GOLD = '#e8c461'; RUST = '#d8836d'; BRICK = '#b85f55'
GREEN = '#5f967a'; BLUE = '#35647f'; TEAL = '#8fb8b4'; GREY = '#c8c1b7'
PALETTE = [RUST, GOLD, GREEN, BLUE, TEAL, '#b9a37b', '#9b8f84', '#d9c6a0']
CMAP_WARM = LinearSegmentedColormap.from_list('warm_dense', ['#fffaf0','#f2d890','#dda064','#b85f55'])
DISTRICT_EN = {'福田区':'Futian','罗湖区':'Luohu','南山区':'Nanshan','宝安区':'Baoan','龙岗区':'Longgang','龙华区':'Longhua','坪山区':'Pingshan','光明区':'Guangming','盐田区':'Yantian'}
TYPE_EN = {'shopping_mall':'Malls','restaurant':'Restaurants','hotel':'Hotels','park_or_recreation':'Parks','residential_property':'Residential'}

mpl.rcParams.update({
 'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans','sans-serif'],
 'svg.fonttype':'none','pdf.fonttype':42,'font.size':5.2,'axes.titlesize':6,'axes.labelsize':5,
 'xtick.labelsize':4.2,'ytick.labelsize':4.2,'legend.fontsize':4.0,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','axes.edgecolor':INK,'axes.linewidth':0.42
})

def save(fig, stem: Path, dpi=600, tight=False):
    kwargs={'bbox_inches':'tight','pad_inches':0.025} if tight else {}
    fig.savefig(stem.with_suffix('.png'), dpi=dpi, **kwargs)
    fig.savefig(stem.with_suffix('.pdf'), **kwargs)
    fig.savefig(stem.with_suffix('.svg'), **kwargs)
    plt.close(fig)

def label(ax, letter, title, x=0, y=1.02):
    ax.text(x,y,letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5, fontweight='bold', color=INK)
    ax.text(x+0.065,y+0.004,title, transform=ax.transAxes, ha='left', va='bottom', fontsize=5.9, fontweight='bold', color=INK)

def clean(ax, grid=True):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(length=2, width=.35, color=MUTED, labelcolor=TEXT)
    if grid: ax.grid(axis='x', color=GRID, lw=.35)

def parse_dict(s):
    if pd.isna(s): return {}
    return ast.literal_eval(str(s))

# ---------- Fig 7: threshold diffusion ----------
def load_diffusion():
    scenarios = pd.read_csv(ROOT/'data_processed/platform/schelling_full_rule_adoption_scenarios_2025_v2.csv')
    agents = pd.read_csv(ROOT/'data_processed/platform/schelling_full_rule_adoption_final_agents_2025_v2.csv')
    return scenarios, agents

def panel7a(ax, scenarios):
    rows=[]
    for _,r in scenarios.iterrows():
        d=parse_dict(r['adopted_by_step'])
        cum=0
        for step in sorted(d):
            cum += d[step]
            rows.append({'scenario':r['scenario'],'step':int(step),'new':int(d[step]),'cumulative':cum,'share':cum/r['silent_agents']})
    df=pd.DataFrame(rows)
    names={'conservative_same_type':'conservative','balanced_threshold':'balanced','permissive_tipping':'permissive'}
    colors={'conservative_same_type':BLUE,'balanced_threshold':RUST,'permissive_tipping':GREEN}
    for sc,g in df.groupby('scenario'):
        ax.plot(g['step'].to_numpy(), (g['share']*100).to_numpy(), lw=1.45, color=colors.get(sc,RUST), marker='o', ms=3.4, mec='white', mew=.45, label=names.get(sc,sc))
        last=g.iloc[-1]
        ax.text(last['step']+.08, last['share']*100, f"{last['share']*100:.1f}%", fontsize=4.2, color=colors.get(sc,RUST), va='center')
    ax.set_xlabel('threshold step'); ax.set_ylabel('cumulative simulated adoption (%)')
    ax.set_xlim(-0.15, 6.55); ax.set_ylim(0, 84)
    clean(ax); label(ax,'a','threshold scenarios create different adoption regimes')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.02,.98), ncol=1)
    ax.text(.98,.04,'n=154,188 silent-rule agents; scenario output is diagnostic, not observed adoption', transform=ax.transAxes, ha='right', va='bottom', fontsize=3.6, color=MUTED)
    df.to_csv(SRC/'fig7_panel_a_threshold_cumulative_source.csv', index=False)
    return df

def panel7b(ax, scenarios):
    # Alternative scenarios must not be pooled as one population. Use equal-width
    # composition bars so scenario totals stay separate and readable.
    flow=[]
    for _,r in scenarios.iterrows():
        for typ,n in parse_dict(r['adopters_by_type']).items(): flow.append({'scenario':r['scenario'],'type':typ,'n':int(n)})
    df=pd.DataFrame(flow)
    scen_order=['conservative_same_type','balanced_threshold','permissive_tipping']
    type_order=['shopping_mall','park_or_recreation','restaurant','hotel','residential_property']
    color_map={'shopping_mall':RUST,'restaurant':GOLD,'hotel':BLUE,'park_or_recreation':GREEN,'residential_property':GREY}
    mat=df.pivot_table(index='scenario', columns='type', values='n', aggfunc='sum', fill_value=0).reindex(index=scen_order, columns=type_order, fill_value=0)
    totals=mat.sum(axis=1)
    shares=mat.div(totals.replace(0,np.nan), axis=0).fillna(0)
    y=np.arange(len(mat))
    left=np.zeros(len(mat))
    for typ in type_order:
        vals=shares[typ].values
        ax.barh(y, vals, left=left, height=.62, color=color_map.get(typ,MUTED), edgecolor='white', lw=.35, label=TYPE_EN.get(typ,typ), alpha=.82)
        for i,v in enumerate(vals):
            n=mat.iloc[i][typ]
            if v>.07:
                ax.text(left[i]+v/2, i, f"{int(n):,}", ha='center', va='center', fontsize=3.25, color='white' if typ in ['hotel','shopping_mall'] else INK)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace('_','\n') + f"\nn={int(totals.loc[s]):,}" for s in mat.index])
    ax.set_xlim(0,1)
    ax.set_xticks([0,.25,.5,.75,1.0])
    ax.set_xticklabels(['0','25','50','75','100'])
    ax.set_xlabel('within-scenario adopter composition (%)')
    ax.invert_yaxis()
    clean(ax)
    label(ax,'b','alternative scenarios shift which host types tip')
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5,-.38), fontsize=3.1, handlelength=1)
    ax.text(.98,-.19,'each row is one alternative scenario; counts are not pooled across scenarios', transform=ax.transAxes, ha='right', va='top', fontsize=3.35, color=MUTED)
    df.to_csv(SRC/'fig7_panel_b_scenario_type_flow_source.csv', index=False)
    return df

def panel7c(ax, agents):
    adop=agents[agents['schelling_state'].eq('simulated_adopter')].copy()
    mat=adop.pivot_table(index='district_name', columns='adopted_step', values='id', aggfunc='count', fill_value=0)
    mat=mat.reindex(index=['宝安区','龙岗区','龙华区','南山区','福田区','光明区','罗湖区','坪山区','盐田区']).fillna(0)
    mat=mat[[c for c in sorted(mat.columns) if c>=0]]
    share=mat.div(mat.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    ax.imshow(share.values, cmap=CMAP_WARM, aspect='auto', vmin=0, vmax=max(.01,share.values.max()))
    ax.set_yticks(range(len(mat))); ax.set_yticklabels([DISTRICT_EN.get(i,i) for i in mat.index])
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels([str(c) for c in mat.columns])
    ax.tick_params(length=0)
    for i in range(len(mat)):
        for j,c in enumerate(mat.columns):
            v=int(mat.iloc[i,j]); pct=share.iloc[i,j]*100
            txt=f'{v//1000}k\n{pct:.0f}%' if v>=1000 else (f'{v}\n{pct:.0f}%' if v>0 else '·')
            ax.text(j,i,txt,ha='center',va='center',fontsize=2.95,color='white' if share.iloc[i,j]>.28 else INK, linespacing=.82)
    ax.set_xlabel('adoption step'); label(ax,'c','district timing fingerprints expose early and late tipping')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks(np.arange(-.5,len(mat.columns),1),minor=True); ax.set_yticks(np.arange(-.5,len(mat),1),minor=True); ax.grid(which='minor',color='white',lw=.55)
    mat.reset_index().to_csv(SRC/'fig7_panel_c_district_timing_source.csv', index=False)
    return mat

def panel7d(ax, agents):
    df=agents.copy()
    df['margin']=df['adoption_propensity']-df['threshold']
    df['type']=df['primary_venue_type'].map(TYPE_EN).fillna(df['primary_venue_type'])
    order=['Restaurants','Malls','Hotels','Parks','Residential']
    order=[o for o in order if o in set(df['type'])]
    # sample for raw marks, keep full distribution for quantiles
    rng=np.random.default_rng(42)
    for i,t in enumerate(order):
        vals=df.loc[df['type'].eq(t),'margin'].dropna().clip(-.65,.65).to_numpy()
        if len(vals)==0: continue
        parts=ax.violinplot([vals],positions=[i],widths=.78,showmeans=False,showmedians=False,showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor({'Malls':RUST,'Hotels':BLUE,'Parks':GREEN,'Restaurants':GOLD,'Residential':GREY}.get(t,GREY)); pc.set_alpha(.42); pc.set_edgecolor('none')
        q=np.quantile(vals,[.10,.25,.50,.75,.90])
        ax.vlines(i,q[1],q[3],color=INK,lw=.85); ax.scatter([i],[q[2]],s=15,color=INK,zorder=5)
        sample=vals if len(vals)<=900 else rng.choice(vals,900,replace=False)
        ax.scatter(np.full(len(sample),i)+rng.normal(0,.055,len(sample)),sample,s=1.4,color=TEXT,alpha=.18,lw=0,rasterized=True)
        near=((vals>-0.04)&(vals<0.04)).mean()*100
        ax.text(i,.60,f'n={len(vals):,}\nnear {near:.0f}%',ha='center',va='top',fontsize=3.2,color=MUTED,linespacing=.88)
    ax.axhline(0,color=INK,lw=.65,alpha=.72)
    ax.fill_between([-0.6,len(order)-.4],[-.04,-.04],[.04,.04],color=GOLD,alpha=.16,zorder=0)
    ax.set_xlim(-.55,len(order)-.45); ax.set_ylim(-.52,.64)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=25,ha='right')
    ax.set_ylabel('propensity minus threshold')
    clean(ax); label(ax,'d','threshold-margin distributions show which hosts sit near tipping')
    ax.text(.98,.04,'gold band marks near-threshold agents; full agent distributions are summarized by violins and quantiles', transform=ax.transAxes, ha='right', va='bottom', fontsize=3.35, color=MUTED)
    df.groupby('type')['margin'].describe().to_csv(SRC/'fig7_panel_d_threshold_margin_source.csv')

def render_fig7():
    scenarios, agents=load_diffusion()
    funcs=[panel7a,panel7b,panel7c,panel7d]
    for letter,func in zip('abcd',funcs):
        fig,ax=plt.subplots(figsize=(3.45,2.75)); fig.subplots_adjust(left=.15,right=.965,bottom=.17,top=.82)
        if letter in 'ab': func(ax, scenarios)
        else: func(ax, agents)
        save(fig, PANELS/f'fig7_threshold_diffusion_v29_panel_{letter}', tight=False)
    fig=plt.figure(figsize=(7.2,5.9)); gs=GridSpec(2,2,figure=fig,wspace=.22,hspace=.42,left=.08,right=.985,bottom=.08,top=.90)
    axs=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    panel7a(axs[0], scenarios); panel7b(axs[1], scenarios); panel7c(axs[2], agents); panel7d(axs[3], agents)
    fig.text(.02,.975,'Fig. 7 | Threshold diffusion shows how latent ecology can tip into rule visibility',ha='left',va='top',fontsize=8.2,fontweight='bold',color=INK)
    fig.text(.02,.947,'The scenario model is used as a mechanism diagnostic: it shows where rule publicisation could propagate, not observed pet entry.',ha='left',va='top',fontsize=5,color=TEXT)
    save(fig, COMP/'fig7_threshold_diffusion_v29_composite', dpi=520, tight=False)

# ---------- Fig 8: validation and robustness ----------
def load_validation():
    queue=pd.read_csv(ROOT/'data_processed/platform/rule_liminal_verification_queue_2025_v4.csv')
    ledger=pd.read_csv(ROOT/'data_processed/platform/rule_source_ledger_v1.csv')
    sens=pd.read_csv(ROOT/'data_processed/model/primary_source_sensitivity_district_v1.csv')
    loo=pd.read_csv(ROOT/'data_processed/platform/rule_adoption_leave_one_out_validation_v1.csv')
    coef=pd.read_csv(ROOT/'data_processed/model/controlled_grid_model_coefficients_v61_sparse_repaired.csv')
    metrics=pd.read_csv(ROOT/'data_processed/model/controlled_grid_model_metrics_v61_sparse_repaired.csv')
    return queue, ledger, sens, loo, coef, metrics

def panel8a(ax, queue):
    df=queue.copy(); df['district']=df['district_name'].map(DISTRICT_EN); df['type']=df['primary_venue_type'].map(TYPE_EN)
    top=df.groupby(['district','type']).agg(n=('host_key','count'), score=('verification_priority_score','mean')).reset_index()
    pivot=top.pivot(index='district',columns='type',values='n').fillna(0)
    order=['Baoan','Longgang','Longhua','Nanshan','Futian','Guangming','Luohu','Pingshan','Yantian']
    pivot=pivot.reindex(order).fillna(0)
    types=[c for c in ['Malls','Hotels','Residential','Parks'] if c in pivot.columns]
    y=np.arange(len(pivot)); left=np.zeros(len(pivot))
    colors={'Malls':RUST,'Hotels':BLUE,'Residential':GREY,'Parks':GREEN}
    for t in types:
        ax.barh(y,pivot[t],left=left,height=.62,color=colors[t],edgecolor='white',lw=.25,label=t)
        for i,v in enumerate(pivot[t]):
            if v>60: ax.text(left[i]+v/2,i,f'{int(v)}',ha='center',va='center',fontsize=3.45,color='white' if t in ['Malls','Hotels'] else INK)
        left+=pivot[t].values
    ax.set_yticks(y); ax.set_yticklabels(pivot.index); ax.invert_yaxis(); ax.set_xlabel('manual-verification candidates')
    clean(ax); label(ax,'a','2,573-candidate verification queue is stratified, not anecdotal')
    ax.legend(frameon=False,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.35),handlelength=1,columnspacing=.8)
    top.to_csv(SRC/'fig8_panel_a_validation_queue_source.csv', index=False)

def panel8b(ax, ledger):
    # Mosaic evidence ledger: source class width x publication-use height.
    status_order=['main_model','main_model_with_caution','supplement_or_sensitivity','discovery_only_until_verified','candidate_queue_only_no_rule_claim','context_only_exclude_from_shenzhen_model']
    class_order=['A_official_primary','B_operator_primary','C_reported_operator_detail','D_secondary_industry_media','E_discovery_or_context','G_venue_identity_only','F_external_context']
    mat=ledger.pivot_table(index='source_class',columns='publication_use_status',values='source_id',aggfunc='count',fill_value=0).reindex(index=class_order,columns=status_order,fill_value=0)
    class_tot=mat.sum(axis=1); total=class_tot.sum()
    colors={'main_model':RUST,'main_model_with_caution':GOLD,'supplement_or_sensitivity':BLUE,'discovery_only_until_verified':GREEN,'candidate_queue_only_no_rule_claim':GREY,'context_only_exclude_from_shenzhen_model':'#efe8db'}
    x0=0
    rows=[]
    for cls in class_order:
        w=class_tot.loc[cls]/total if total else 0
        if w<=0: continue
        y0=0
        for st in status_order:
            n=mat.loc[cls,st]; h=n/class_tot.loc[cls] if class_tot.loc[cls] else 0
            if n>0:
                ax.add_patch(Rectangle((x0,y0),w,h,facecolor=colors[st],alpha=.82,edgecolor='white',lw=.5))
                if w*h>.018:
                    ax.text(x0+w/2,y0+h/2,f'{int(n)}',ha='center',va='center',fontsize=4,color='white' if st in ['main_model','supplement_or_sensitivity'] else INK)
            rows.append({'source_class':cls,'publication_use_status':st,'n':int(n),'x':x0,'width':w,'y':y0,'height':h})
            y0+=h
        short={'A_official_primary':'A official','B_operator_primary':'B operator','C_reported_operator_detail':'C reported','D_secondary_industry_media':'D secondary','E_discovery_or_context':'E discovery','G_venue_identity_only':'G identity','F_external_context':'F external'}.get(cls,cls)
        if w > .045:
            ax.text(x0+w/2,-.035,short.replace(' ','\n'),ha='center',va='top',fontsize=2.75,color=TEXT)
        else:
            ax.text(x0+w/2,-.035,short.split()[0],ha='center',va='top',fontsize=2.65,color=TEXT)
        ax.text(x0+w/2,1.02,f'n={int(class_tot.loc[cls])}',ha='center',va='bottom',fontsize=3.0,color=MUTED)
        x0+=w+.006
    # compact legend in reserved top-left whitespace
    lx=.02; ly=.88
    for i,st in enumerate(status_order[:5]):
        ax.add_patch(Rectangle((lx,ly-i*.055),.022,.025,transform=ax.transAxes,facecolor=colors[st],edgecolor='none',alpha=.82))
        ax.text(lx+.03,ly-i*.055+.012,st.replace('_',' '),transform=ax.transAxes,ha='left',va='center',fontsize=3.0,color=MUTED)
    ax.set_xlim(0,1.04); ax.set_ylim(-.16,1.10); ax.axis('off')
    label(ax,'b','source hierarchy and claim boundary')
    pd.DataFrame(rows).to_csv(SRC/'fig8_panel_b_source_ledger_mosaic_source.csv', index=False)

def panel8c(ax, sens):
    df=sens.copy(); df['district']=df['district_name'].map(DISTRICT_EN)
    df=df.sort_values('mean_primary_sensitive_suppression_index')
    y=np.arange(len(df))
    ax.hlines(y,df['mean_v5_suppression_index'].to_numpy(),df['mean_primary_sensitive_suppression_index'].to_numpy(),color='#d8cfc4',lw=2.8)
    ax.scatter(df['mean_v5_suppression_index'].to_numpy(),y,s=18,color=GREY,edgecolor='white',lw=.4,label='all sources')
    ax.scatter(df['mean_primary_sensitive_suppression_index'].to_numpy(),y,s=22,color=RUST,edgecolor='white',lw=.4,label='primary-source sensitive')
    for _,r in df.tail(3).iterrows():
        ax.text(r['mean_primary_sensitive_suppression_index']+.004, list(df.index).index(r.name), r['district'], fontsize=3.8, color=TEXT, va='center')
    ax.set_yticks(y); ax.set_yticklabels(df['district']); ax.set_xlabel('mean suppression index')
    clean(ax); label(ax,'c','primary-source sensitivity changes level, not the spatial story')
    ax.legend(frameon=False,loc='lower right')
    df.to_csv(SRC/'fig8_panel_c_primary_sensitivity_source.csv', index=False)

def panel8d(ax, loo):
    df=loo.copy(); df['type']=df['primary_venue_type'].map(TYPE_EN).fillna(df['primary_venue_type'])
    order=df.groupby('type')['rank_percentile'].median().sort_values().index
    data=[df[df['type'].eq(t)]['rank_percentile'].dropna().values for t in order]
    parts=ax.violinplot(data,positions=np.arange(len(order)),widths=.72,showmeans=False,showmedians=False,showextrema=False)
    for pc,t in zip(parts['bodies'],order):
        pc.set_facecolor({'Malls':RUST,'Hotels':BLUE,'Parks':GREEN,'Restaurants':GOLD}.get(t,GREY)); pc.set_alpha(.48); pc.set_edgecolor('none')
    for i,vals in enumerate(data):
        q=np.quantile(vals,[.25,.5,.75])
        ax.vlines(i,q[0],q[2],color=INK,lw=.8); ax.scatter([i],[q[1]],s=13,color=INK,zorder=5)
        jitter=np.random.default_rng(10+i).normal(0,.055,len(vals))
        ax.scatter(np.full(len(vals),i)+jitter,vals,s=5,color=TEXT,alpha=.42,lw=0)
        ax.text(i,.02,f'n={len(vals)}',ha='center',va='bottom',fontsize=3.3,color=MUTED)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order,rotation=25,ha='right')
    ax.set_ylabel('leave-one-out adoption rank percentile'); ax.set_ylim(0,1)
    clean(ax, grid=True); label(ax,'d','leave-one-out validation checks whether known rules rank as plausible adopters')
    df.to_csv(SRC/'fig8_panel_d_leave_one_out_source.csv', index=False)

def panel8e(ax, coef):
    df=coef[(coef['target'].eq('is_suppressed_frontier')) & (coef['spec'].eq('core_plus_controls'))].copy()
    keep=~df['feature'].str.startswith('district_name')
    df=df[keep].sort_values('abs_coefficient',ascending=False).head(14).iloc[::-1]
    y=np.arange(len(df)); colors=[RUST if v>0 else BLUE for v in df['coefficient']]
    ax.axvline(0,color=INK,lw=.55,alpha=.6)
    ax.hlines(y,0,df['coefficient'].to_numpy(),color='#d5cec2',lw=1.5)
    ax.scatter(df['coefficient'].to_numpy(),y,s=22,color=colors,edgecolor='white',lw=.35,zorder=4)
    labels=[f.replace('_norm_v51','').replace('_norm','').replace('_',' ') for f in df['feature']]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('standardized diagnostic coefficient')
    clean(ax); label(ax,'e','suppressed frontier is carried by ecology-rule imbalance, not controls alone')
    ax.text(.98,.04,'coefficients are diagnostic model weights, not causal effects', transform=ax.transAxes, ha='right', va='bottom', fontsize=3.5, color=MUTED)
    df.to_csv(SRC/'fig8_panel_e_coefficient_source.csv', index=False)

def panel8f(ax, metrics):
    df=metrics[metrics['target'].eq('is_suppressed_frontier') & metrics['model'].str.contains('logistic',na=False)].copy()
    order=['controls_only','core_only','core_plus_controls']
    df=df[df['spec'].isin(order)].groupby('spec', as_index=False).agg(
        roc_auc=('roc_auc','max'),
        average_precision=('average_precision','max'),
        n=('n','max'),
        positive=('positive','max')
    ).set_index('spec').reindex(order).reset_index()
    y=np.arange(len(df))
    ax.hlines(y,df['average_precision'],df['roc_auc'],color='#d7cfc2',lw=3.0,zorder=1)
    ax.scatter(df['average_precision'],y,s=30,color=RUST,edgecolor='white',lw=.45,label='average precision',zorder=3)
    ax.scatter(df['roc_auc'],y,s=30,color=BLUE,edgecolor='white',lw=.45,label='ROC-AUC',zorder=3)
    for i,r in df.iterrows():
        ax.text(r['average_precision']-.022,i-.18,f"{r['average_precision']:.2f}",ha='right',va='center',fontsize=3.6,color=RUST)
        ax.text(min(r['roc_auc']+.010,1.016),i+.18,f"{r['roc_auc']:.2f}",ha='left',va='center',fontsize=3.6,color=BLUE)
        delta=r['average_precision']-df.loc[df['spec'].eq('controls_only'),'average_precision'].iloc[0]
        ax.text(.575,i,f"ΔAP {delta:+.2f}",ha='left',va='center',fontsize=3.15,color=MUTED)
    label_map={'controls_only':'controls only','core_only':'core only','core_plus_controls':'core + controls'}
    ax.set_yticks(y); ax.set_yticklabels(df['spec'].map(label_map).fillna(df['spec']))
    ax.set_xlim(.45,1.05); ax.set_xlabel('diagnostic score')
    clean(ax); label(ax,'f','diagnostic separability, not causal validation')
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(.02,.98))
    ax.text(.98,-.27,'near-perfect scores reflect constructed diagnostic targets; use as separability check only', transform=ax.transAxes, ha='right', va='top', fontsize=3.25, color=MUTED)
    df.to_csv(SRC/'fig8_panel_f_model_score_strip_source.csv', index=False)

def render_fig8():
    queue,ledger,sens,loo,coef,metrics=load_validation()
    calls=[(panel8a,queue),(panel8b,ledger),(panel8c,sens),(panel8d,loo),(panel8e,coef),(panel8f,metrics)]
    for letter,(func,data) in zip('abcdef',calls):
        fig,ax=plt.subplots(figsize=(3.45,2.75)); fig.subplots_adjust(left=.20,right=.965,bottom=.22,top=.80)
        if letter=='b': fig.subplots_adjust(left=.25,right=.985,bottom=.25,top=.78)
        if letter=='e': fig.subplots_adjust(left=.34,right=.965,bottom=.17,top=.80)
        if letter=='f': fig.subplots_adjust(left=.28,right=.965,bottom=.28,top=.80)
        func(ax,data); save(fig,PANELS/f'fig8_validation_robustness_v29_panel_{letter}',tight=False)
    fig=plt.figure(figsize=(7.25,8.4)); gs=GridSpec(3,2,figure=fig,wspace=.32,hspace=.58,left=.09,right=.985,bottom=.07,top=.90)
    axs=[fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]
    for ax,(func,data) in zip(axs,calls): func(ax,data)
    fig.text(.02,.975,'Fig. 8 | Validation queue and robustness diagnostics define what can be claimed',ha='left',va='top',fontsize=8.2,fontweight='bold',color=INK)
    fig.text(.02,.948,'The figure separates source-grounded rules, verification candidates, primary-source sensitivity and diagnostic model strength.',ha='left',va='top',fontsize=5,color=TEXT)
    save(fig,COMP/'fig8_validation_robustness_v29_composite',dpi=520,tight=False)

# ---------- Fig 9: morphology and spatial heterogeneity ----------
def load_morphology():
    cluster=pd.read_csv(ROOT/'data_processed/model/advanced_cluster_spatial_diagnostics_500m.csv')
    model=pd.read_csv(ROOT/'data_processed/model/controlled_grid_model_500m_2025_v61_sparse_repaired.csv')
    grid=gpd.read_file(ROOT/'data_processed/model/grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson')
    boundary=gpd.read_file(ROOT/'data_processed/geo/shenzhen_boundary_verified.geojson')
    return cluster, model, grid, boundary

def panel9a(ax, cluster):
    df=cluster.copy()
    x=df['service_score'].clip(0,0.18); y=df['affordance_morphology_score'].clip(0,1)
    hb=ax.hexbin(x,y,gridsize=42,cmap=CMAP_WARM,mincnt=1,bins='log',linewidths=0,alpha=.92)
    colors={'high_service_high_morph_affordance':GREEN,'high_service_low_morph_affordance':RUST,'low_service_high_morph_affordance':BLUE,'low_service_low_morph_affordance':GREY}
    for typ,g in df[df['baseline_mismatch_type'].str.startswith('high_service')].groupby('baseline_mismatch_type'):
        sample=g.sample(min(len(g),320), random_state=4)
        ax.scatter(sample['service_score'].clip(0,0.18),sample['affordance_morphology_score'],s=5,color=colors.get(typ,RUST),alpha=.55,lw=0,label=typ.replace('_',' '))
    ax.set_xlabel('pet-service score'); ax.set_ylabel('morphological affordance score')
    ax.set_xlim(0,.18); ax.set_ylim(0,1.02); clean(ax,grid=False)
    label(ax,'a','service-morphology phase space reveals mismatch')
    ax.text(.98,.04,'hexagons summarize 9,049 grid cells; highlighted points are high-service mismatch cells',transform=ax.transAxes,ha='right',va='bottom',fontsize=3.35,color=MUTED)
    df[['grid_id','district_name','baseline_mismatch_type','service_score','affordance_morphology_score']].to_csv(SRC/'fig9_panel_a_phase_space_source.csv',index=False)

def panel9b(ax, cluster):
    order=['low_service_low_morph_affordance','low_service_high_morph_affordance','high_service_low_morph_affordance','high_service_high_morph_affordance']
    labels=['low service\nlow morph','low service\nhigh morph','high service\nlow morph','high service\nhigh morph']
    mat=cluster.pivot_table(index='district_name',columns='baseline_mismatch_type',values='grid_id',aggfunc='count',fill_value=0)
    district_order=['宝安区','龙岗区','龙华区','南山区','福田区','光明区','罗湖区','坪山区','盐田区']
    mat=mat.reindex(index=district_order,columns=order,fill_value=0)
    share=mat.div(mat.sum(axis=1),axis=0).fillna(0)
    y=np.arange(len(mat)); left=np.zeros(len(mat)); colors=[GREY,BLUE,RUST,GREEN]
    for col,lab,c in zip(order,labels,colors):
        vals=share[col].values
        ax.barh(y,vals,left=left,height=.65,color=c,edgecolor='white',lw=.25,label=lab)
        for i,v in enumerate(vals):
            n=mat.iloc[i][col]
            if v>.12: ax.text(left[i]+v/2,i,f'{int(n)}',ha='center',va='center',fontsize=3.25,color='white' if c in [BLUE,RUST,GREEN] else INK)
        left+=vals
    ax.set_yticks(y); ax.set_yticklabels([DISTRICT_EN.get(i,i) for i in mat.index]); ax.invert_yaxis()
    ax.set_xlim(0,1); ax.set_xlabel('share of grids')
    clean(ax); label(ax,'b','district mosaic shows morphology can absorb or expose pet-service pressure')
    ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.38),fontsize=3.25)
    mat.reset_index().to_csv(SRC/'fig9_panel_b_mismatch_mosaic_source.csv',index=False)

def panel9c(ax, model):
    vars=[('road_density_m_per_km2','road density'),('ped_path_density_m_per_km2','pedestrian-path density'),('public_space_share','capped public-space share'),('building_coverage_share','building coverage')]
    groups=[('suppressed frontier',model[model['is_suppressed_frontier'].eq(1)]),('other grids',model[model['is_suppressed_frontier'].eq(0)])]
    rows=[]
    for vi,(col,labeltxt) in enumerate(vars):
        for gi,(gname,gdf) in enumerate(groups):
            vals=gdf[col].replace([np.inf,-np.inf],np.nan).dropna().clip(lower=0)
            if 'density' in col: vals=np.log1p(vals)
            vals=vals.to_numpy()
            parts=ax.violinplot([vals],positions=[vi*3+gi*.85],widths=.62,showmeans=False,showmedians=False,showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(RUST if gi==0 else BLUE); pc.set_alpha(.46 if gi==0 else .30); pc.set_edgecolor('none')
            q=np.quantile(vals,[.25,.5,.75]); ax.vlines(vi*3+gi*.85,q[0],q[2],color=INK,lw=.7); ax.scatter([vi*3+gi*.85],[q[1]],s=12,color=INK,zorder=4)
            rows.append({'variable':col,'group':gname,'n':len(vals),'q25':q[0],'median':q[1],'q75':q[2]})
    ax.set_xticks([i*3+.42 for i in range(len(vars))]); ax.set_xticklabels([v[1] for v in vars],rotation=20,ha='right')
    ax.set_ylabel('transformed value; densities use log1p')
    clean(ax); label(ax,'c','control distributions bound the frontier claim')
    ax.text(.99,-.25,'share variables are capped 0-1 in this controlled-model layer', transform=ax.transAxes, ha='right', va='top', fontsize=3.35, color=MUTED)
    ax.scatter([],[],color=RUST,label='suppressed frontier'); ax.scatter([],[],color=BLUE,label='other grids'); ax.legend(frameon=False,loc='upper right')
    pd.DataFrame(rows).to_csv(SRC/'fig9_panel_c_control_distribution_source.csv',index=False)

def panel9d(ax, cluster):
    df=cluster.copy()
    counts=pd.crosstab(df['local_service_cluster_sig'],df['local_morph_cluster_sig']).reindex(index=['sig','not_sig'],columns=['sig','not_sig'],fill_value=0)
    vals=counts.values; scaled=vals/vals.sum()
    ax.imshow(scaled,cmap=CMAP_WARM,aspect='equal')
    ax.set_xticks([0,1]); ax.set_xticklabels(['morphology\nclustered','morphology\nnot clustered'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['service\nclustered','service\nnot clustered'])
    ax.tick_params(length=0)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{int(vals[i,j]):,}\n{scaled[i,j]*100:.0f}%",ha='center',va='center',fontsize=5,color='white' if scaled[i,j]>.35 else INK)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks(np.arange(-.5,2,1),minor=True); ax.set_yticks(np.arange(-.5,2,1),minor=True); ax.grid(which='minor',color='white',lw=.7)
    label(ax,'d','local Moran overlap separates service clustering from urban-form clustering')
    counts.to_csv(SRC/'fig9_panel_d_local_moran_overlap_source.csv')

def panel9e(ax, grid, boundary):
    g=grid.copy()
    if g.crs and str(g.crs).upper()!='EPSG:32649':
        g=g.to_crs('EPSG:32649'); boundary=boundary.to_crs('EPSG:32649')
    color_map={'suppressed_emergence_frontier':RUST,'emergent_capability_core':GREEN,'mixed_or_transitional_zone':GOLD,'rule_first_demonstration_zone':BLUE,'low_pet_city_signal':'#efebe3'}
    g.plot(ax=ax,color=g['grid_emergence_type_v51'].map(color_map).fillna('#efebe3'),edgecolor='white',linewidth=.01,alpha=.82,rasterized=True)
    boundary.boundary.plot(ax=ax,color='#716b63',linewidth=.35,alpha=.78)
    ax.set_axis_off(); ax.set_aspect('equal')
    # inset legend
    items=[('suppressed frontier',RUST),('capability core',GREEN),('mixed/transitional',GOLD),('rule-first zone',BLUE)]
    for i,(lab,c) in enumerate(items):
        ax.add_patch(Rectangle((.02,.12+i*.055),.025,.025,transform=ax.transAxes,facecolor=c,edgecolor='white',lw=.2))
        ax.text(.052,.132+i*.055,lab,transform=ax.transAxes,ha='left',va='center',fontsize=3.55,color=MUTED)
    label(ax,'e','spatial typology links morphology, service ecology and rule silence')
    g[['grid_id','district_name','grid_emergence_type_v51','suppression_index_v51','emergence_index_v51']].to_csv(SRC/'fig9_panel_e_spatial_typology_source.csv',index=False)

def render_fig9():
    cluster, model, grid, boundary=load_morphology()
    calls=[(panel9a,cluster),(panel9b,cluster),(panel9c,model),(panel9d,cluster),(panel9e,(grid,boundary))]
    for letter,(func,data) in zip('abcde',calls):
        fig,ax=plt.subplots(figsize=(3.45,2.75)); fig.subplots_adjust(left=.17,right=.985,bottom=.22,top=.78)
        if letter=='e': fig.subplots_adjust(left=.02,right=.99,bottom=.04,top=.86)
        if isinstance(data,tuple): func(ax,*data)
        else: func(ax,data)
        save(fig,PANELS/f'fig9_morphology_heterogeneity_v29_panel_{letter}',tight=False)
    fig=plt.figure(figsize=(7.25,8.0)); gs=GridSpec(3,2,figure=fig,wspace=.28,hspace=.55,left=.08,right=.985,bottom=.07,top=.90)
    axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[2,:])]
    panel9a(axs[0],cluster); panel9b(axs[1],cluster); panel9c(axs[2],model); panel9d(axs[3],cluster); panel9e(axs[4],grid,boundary)
    fig.text(.02,.975,'Fig. 9 | Urban morphology conditions where pet-service ecology becomes urban capability',ha='left',va='top',fontsize=8.2,fontweight='bold',color=INK)
    fig.text(.02,.948,'The spatial mechanism is not a POI count alone: service concentration, morphology and local clustering jointly shape rule-liminal capability.',ha='left',va='top',fontsize=5,color=TEXT)
    save(fig,COMP/'fig9_morphology_heterogeneity_v29_composite',dpi=520,tight=False)

def contact_sheet():
    files=sorted(PANELS.glob('fig7_*.png'))+sorted(PANELS.glob('fig8_*.png'))+sorted(PANELS.glob('fig9_*.png'))+sorted(COMP.glob('fig7_*.png'))+sorted(COMP.glob('fig8_*.png'))+sorted(COMP.glob('fig9_*.png'))
    thumbs=[]
    for f in files:
        im=Image.open(f).convert('RGB'); im.thumbnail((420,320),Image.LANCZOS)
        can=Image.new('RGB',(450,370),'white'); can.paste(im,((450-im.width)//2,12))
        ImageDraw.Draw(can).text((10,346),f.name,fill=(35,35,35)); thumbs.append(can)
    cols=3; rows=math.ceil(len(thumbs)/cols); sheet=Image.new('RGB',(cols*450,rows*370),'white')
    for i,im in enumerate(thumbs): sheet.paste(im,((i%cols)*450,(i//cols)*370))
    sheet.save(PREVIEWS/'v29_expanded_system_contact_sheet.png',dpi=(180,180))
    manifest=[]
    for f in files: manifest.append({'asset':f.relative_to(ROOT).as_posix(),'role':'panel' if f.parent==PANELS else 'composite'})
    pd.DataFrame(manifest).to_csv(SRC/'v29_expanded_system_export_manifest.csv',index=False)

if __name__=='__main__':
    render_fig7(); render_fig8(); render_fig9(); contact_sheet(); print('rendered v29 expanded system')
