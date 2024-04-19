import os
import numpy as np
import pandas as pd
from roofline_V100 import roofline


path = './csv_V100'
filenames=os.listdir(path)
files = [os.path.join(path,filename) for filename in filenames]
print(files)
# files = ['/home/jiangjz/deep-rec/corec/profile/csv_out_new/metrics_amazon_dien_new.csv']

dfs={}
for file in files:
    tag, ext = os.path.splitext(os.path.basename(file))
    dfs[tag]=pd.DataFrame()
    with open(file,'r') as f:
        df = pd.read_csv(file, thousands=",")
        dft= df.groupby(['Kernel Name','Metric Name']).sum(numeric_only=False)
        # print(dft.columns.tolist())
        dfmetric=pd.pivot_table(dft, index='Kernel Name', columns='Metric Name', values='Metric Value')
        dfmetric['Count']=df.groupby(['Kernel Name']).count()['ID'].div(dfmetric.shape[1])
        # "smsp__cycles_elapsed.avg,smsp__cycles_elapsed.avg.per_second,"
        dfmetric['Time']=dfmetric['smsp__cycles_elapsed.avg']/(dfmetric['smsp__cycles_elapsed.avg.per_second']/dfmetric['Count'] )          
        dfmetric['CC FLOPs']= 2 * dfmetric['smsp__sass_thread_inst_executed_op_dfma_pred_on.sum'] \
                            + dfmetric['smsp__sass_thread_inst_executed_op_dmul_pred_on.sum'] \
                            + dfmetric['smsp__sass_thread_inst_executed_op_dadd_pred_on.sum'] \
                            + 2 * dfmetric['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum'] \
                            + dfmetric['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum'] \
                            + dfmetric['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum'] \
                            + 2 * dfmetric['smsp__sass_thread_inst_executed_op_hfma_pred_on.sum'] \
                            + dfmetric['smsp__sass_thread_inst_executed_op_hmul_pred_on.sum'] \
                            + dfmetric['smsp__sass_thread_inst_executed_op_hadd_pred_on.sum'] 

        dfmetric['TC FLOPs']= 512 * dfmetric['smsp__inst_executed_pipe_tensor.sum']
        dfmetric['all FLOPs']= dfmetric['CC FLOPs'] + dfmetric['TC FLOPs']
        dfmetric['AI HBM'] = dfmetric['all FLOPs'].div(dfmetric['dram__bytes.sum'])
        dfmetric['AI L2'] = dfmetric['all FLOPs'].div(dfmetric['lts__t_bytes.sum'])
        dfmetric['AI L1'] = dfmetric['all FLOPs'].div(dfmetric['l1tex__t_bytes.sum'])

        dfmetric['GFLOP/s'] = dfmetric['all FLOPs']/ dfmetric['Time'] /1024/1024/1024
        dfmetric['TC GFLOP/s'] = dfmetric['TC FLOPs']/ dfmetric['Time'] /1024/1024/1024
        dfs[tag]=dfmetric

LABELS = []
FLOPS  = []
AIHBM  = []
AIL2   = []
AIL1   = []

tags=dfs.keys()
flags=['all'] #'HBM','L2','L1' or 'all'
for tag in tags:
    for flag in flags:
        dfm=dfs[tag]
        LABELS += [tag]
        FLOPS  += [dfm['all FLOPs'].sum() / dfm['Time'].sum() / 1024 / 1024 / 1024]
        AIHBM  += [dfm['all FLOPs'].sum() / dfm['dram__bytes.sum'].sum()]
        AIL2   += [dfm['all FLOPs'].sum() / dfm['lts__t_bytes.sum'].sum()]
        AIL1   += [dfm['all FLOPs'].sum() / dfm['l1tex__t_bytes.sum'].sum()]
        print (tag,": ",FLOPS[-1], " Gflops, ",AIHBM[-1]," ,", AIL2[-1],",", AIL1[-1],",", dfm['all FLOPs'].sum(),",", dfm['dram__bytes.sum'].sum(),",",dfm['Time'].sum())
        # print (tag,",",AIHBM,",", dfm['all FLOPs'].sum(),",", dfm['dram__bytes.sum'].sum(),",",dfm['Time'].sum())
        # print ("\"",tag,"\": ", AIHBM,",")
        
        
roofline("DGX", FLOPS, AIHBM, AIL2, AIL1, LABELS, flag)