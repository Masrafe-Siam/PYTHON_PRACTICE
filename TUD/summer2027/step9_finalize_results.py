"""TU Dortmund M.Sc. Data Science Summer 2027 — Step 9.
Regenerate final verification tables from the original CSV without modifying observations.
Usage: python step9_finalize_results.py --csv waterquality.csv --out results/step_9
"""
import argparse
import hashlib
import platform
from pathlib import Path
from itertools import combinations
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import t, studentized_range, shapiro, levene
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

SEASONS = ['spring', 'summer', 'autumn', 'winter']
SECTIONS = ['A1_headwaters', 'A2_upper_course', 'A3_middle_course', 'A4_lower_course', 'A5_mouth']
OUTCOMES = ['oxygen_mgL', 'nitrate_mgL']


def model_design(model, data):
    return np.asarray(build_design_matrices([model.model.data.design_info], data)[0])


def spring_autumn_contrast(model):
    frame = pd.DataFrame({'saison': ['spring']*5+['autumn']*5, 'river_section': SECTIONS*2})
    matrix = model_design(model, frame)
    return matrix[:5].mean(axis=0) - matrix[5:].mean(axis=0)


def summary_by_group(df, grouping):
    groups = [grouping] if isinstance(grouping,str) else list(grouping)
    long = df.melt(id_vars=groups, value_vars=OUTCOMES, var_name='variable', value_name='concentration')
    gb = long.groupby(groups+['variable'],observed=True)['concentration']
    result = gb.agg(n='count',mean_mgL='mean',sd_mgL='std',median_mgL='median',
                    min_mgL='min',max_mgL='max').reset_index()
    q = gb.quantile([.25,.75]).unstack(level=-1).reset_index()
    q.columns = groups+['variable','Q1_mgL','Q3_mgL']
    result=result.merge(q,on=groups+['variable'])
    result['IQR_mgL']=result['Q3_mgL']-result['Q1_mgL']
    return result


def run(csv_path, out_path):
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True,exist_ok=True)
    df = pd.read_csv(csv_path, sep=';', decimal='.')
    needed = ['ID','river_section','saison',*OUTCOMES]
    assert set(df.columns)==set(needed) and len(df.columns)==len(needed), f'Unexpected CSV header: {df.columns.tolist()}'
    assert len(df)==900 and df['ID'].is_unique
    assert set(df.saison.unique())==set(SEASONS)
    assert set(df.river_section.unique())==set(SECTIONS)
    assert not df[needed].isnull().any().any()
    assert all(pd.api.types.is_numeric_dtype(df[v]) for v in OUTCOMES)
    assert np.isfinite(df[OUTCOMES].to_numpy()).all()
    assert (df[OUTCOMES]>=0).all().all()
    assert (df.groupby(['saison','river_section']).size()==45).all()

    # Full reproducible descriptive tables
    season = summary_by_group(df, 'saison')
    section = summary_by_group(df, 'river_section')
    cell = summary_by_group(df, ['river_section','saison'])
    season['saison'] = pd.Categorical(season['saison'], SEASONS, ordered=True)
    section['river_section'] = pd.Categorical(section['river_section'], SECTIONS, ordered=True)
    season.sort_values(['saison','variable']).to_csv(out_path/'table_1a_seasons.csv',index=False)
    section.sort_values(['river_section','variable']).to_csv(out_path/'table_1b_sections.csv',index=False)
    cell.to_csv(out_path/'detailed_20_cell_descriptives.csv',index=False)

    # Compact means and sd for main report table
    combined = pd.concat([
        season.assign(group_type='Season', group=season['saison'].astype(str)),
        section.assign(group_type='River section', group=section['river_section'].astype(str))
    ],ignore_index=True)
    displaytable = combined.pivot(index=['group_type','group'], columns='variable', values=['n','mean_mgL','sd_mgL'])
    displaytable.columns=['_'.join([a,b]) for a,b in displaytable.columns]
    displaytable.reset_index().to_csv(out_path/'report_table_1_descriptive.csv',index=False)

    models = {}
    contrast_rows = []
    anova_rows = []
    diagnostics_rows = []
    sensitivity_rows = []
    oxygen_rows = []
    nitrate_rows = []

    for v in OUTCOMES:
        model = smf.ols(f'{v} ~ C(saison,Sum) * C(river_section,Sum)',data=df).fit()
        models[v] = model
        robust = model.get_robustcov_results(cov_type='HC3')
        a = sm.stats.anova_lm(model, typ=3)
        mse = model.mse_resid
        res = model.resid
        groups = [g[v].values for _,g in df.groupby(['saison','river_section'])]
        s_w,s_p=shapiro(res)
        l_f,l_p=levene(*groups,center='median')
        diagnostics_rows.append(dict(variable=v,shapiro_W=s_w,shapiro_p=s_p,brown_forsythe_F=l_f,brown_forsythe_p=l_p))

        contrast_vec = spring_autumn_contrast(model)
        tt=model.t_test(contrast_vec)
        tr=robust.t_test(contrast_vec)
        ci=tt.conf_int()[0]
        cri=tr.conf_int()[0]
        contrast_rows.append(dict(
            variable=v,spring_mean=df.loc[df.saison=='spring',v].mean(),
            autumn_mean=df.loc[df.saison=='autumn',v].mean(),
            mean_diff_mgL=np.asarray(tt.effect).item(),
            t=np.asarray(tt.tvalue).item(), df=int(model.df_resid),p=np.asarray(tt.pvalue).item(),
            ci_lower_mgL=ci[0],ci_upper_mgL=ci[1],
            pooled_residual_standardized_d=np.asarray(tt.effect).item()/np.sqrt(mse),
            hc3_p=np.asarray(tr.pvalue).item(),hc3_ci_lower_mgL=cri[0],hc3_ci_upper_mgL=cri[1]
        ))

        for term,label in [('C(saison, Sum)','Season'),('C(river_section, Sum)','River section'),('C(saison, Sum):C(river_section, Sum)','Interaction')]:
            ss=a.loc[term,'sum_sq'];ssr=a.loc['Residual','sum_sq']
            anova_rows.append(dict(variable=v,effect=label,df1=int(a.loc[term,'df']),df2=int(model.df_resid),F=a.loc[term,'F'],p=a.loc[term,'PR(>F)'],partial_eta_squared=ss/(ss+ssr)))

        # Outliers are flagged per season by section and ONLY excluded for secondary sensitivity check
        flagged=pd.Series(False,index=df.index)
        for _,g in df.groupby(['saison','river_section']):
            Q1,Q3=g[v].quantile([.25,.75]);iqr=Q3-Q1
            flagged.loc[g.index]=(g[v]<Q1-1.5*iqr)|(g[v]>Q3+1.5*iqr)
        mm=smf.ols(f'{v} ~ C(saison,Sum) * C(river_section,Sum)',data=df.loc[~flagged]).fit()
        t2=mm.t_test(spring_autumn_contrast(mm))
        sensitivity_rows.append(dict(variable=v,n_flagged=int(flagged.sum()),n_retained=int((~flagged).sum()),spring_autumn_difference_after_flagged_exclusion=np.asarray(t2.effect).item(),p_after_exclusion=np.asarray(t2.pvalue).item()))

        if v=='oxygen_mgL':
            means=df.groupby(['saison','river_section'])[v].mean()
            std_error=np.sqrt(2*mse/45)
            for s in SEASONS:
                for fst,snd in combinations(SECTIONS,2):
                    d=means.loc[s,fst]-means.loc[s,snd]
                    p=2*t.sf(abs(d/std_error),df=model.df_resid)
                    oxygen_rows.append(dict(season=s,pair=f'{fst[:2]} - {snd[:2]}',difference_mgL=d,t=d/std_error,p_raw=p))
        else:
            means=df.groupby('river_section')[v].mean()
            qse=np.sqrt(mse/180)
            crit=studentized_range.ppf(.95,5,model.df_resid)
            for fst,snd in combinations(SECTIONS,2):
                d=means.loc[fst]-means.loc[snd]
                p=studentized_range.sf(abs(d)/qse,5,model.df_resid)
                nitrate_rows.append(dict(pair=f'{fst[:2]} - {snd[:2]}',difference_mgL=d,p_tukey=p,ci_lower_mgL=d-crit*qse,ci_upper_mgL=d+crit*qse))

    contrasts=pd.DataFrame(contrast_rows)
    contrasts['p_holm']=multipletests(contrasts.p,method='holm')[1]
    contrasts['hc3_p_holm']=multipletests(contrasts.hc3_p,method='holm')[1]
    contrasts.to_csv(out_path/'report_table_2_spring_autumn.csv',index=False)
    anova=pd.DataFrame(anova_rows)
    # Group corrections only for section effects and interactions across two outcomes
    anova['holm_p']=np.nan
    for label in ['River section','Interaction']:
        ix=anova.effect==label
        anova.loc[ix,'holm_p']=multipletests(anova.loc[ix,'p'],method='holm')[1]
    anova.to_csv(out_path/'report_table_3_anova.csv',index=False)
    pd.DataFrame(diagnostics_rows).to_csv(out_path/'assumptions.csv',index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out_path/'outlier_sensitivity.csv',index=False)

    oxygen=pd.DataFrame(oxygen_rows)
    oxygen['p_holm']=multipletests(oxygen.p_raw,method='holm')[1]
    oxygen['significant']=oxygen.p_holm<.05
    oxygen.to_csv(out_path/'full_oxygen_pairwise_40.csv',index=False)
    oxygen.loc[oxygen.significant].to_csv(out_path/'report_table_4_oxygen_significant_pairs.csv',index=False)
    nitrate=pd.DataFrame(nitrate_rows)
    nitrate['significant']=nitrate.p_tukey<.05
    nitrate.to_csv(out_path/'report_table_5_nitrate_all_pairs.csv',index=False)

    assert int(oxygen.significant.sum())==6
    assert int(nitrate.significant.sum())==10
    assert np.isclose(contrasts.loc[contrasts.variable=='oxygen_mgL','mean_diff_mgL'].iloc[0],.15193777777777767)
    assert np.isclose(contrasts.loc[contrasts.variable=='nitrate_mgL','mean_diff_mgL'].iloc[0],.8706622222222222)

    with open(out_path/'software_and_input_provenance.txt','w',encoding='utf-8') as fh:
        fh.write(f'Source data: {csv_path.name}\n')
        fh.write(f'SHA256: {hashlib.sha256(csv_path.read_bytes()).hexdigest()}\n')
        fh.write(f'Python: {platform.python_version()}\n')
        for lib in ['numpy','pandas','scipy','statsmodels','matplotlib','patsy']:
            try: val=version(lib)
            except PackageNotFoundError: val='not installed'
            fh.write(f'{lib}: {val}\n')
        fh.write('\nCheck versions again when final notebook is executed on the application computer.\n')

    print('Step 9 complete — all main numerical checks passed.')
    print('Spring–autumn contrasts:\n',contrasts[['variable','mean_diff_mgL','p_holm']].to_string(index=False))
    print('Significant pairs: oxygen',oxygen.significant.sum(),'/ 40; nitrate',nitrate.significant.sum(),'/ 10')
    print('Results saved to:',out_path.resolve())

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default='waterquality.csv')
    parser.add_argument('--out',default='results/step_9')
    args=parser.parse_args()
    run(args.csv,args.out)
