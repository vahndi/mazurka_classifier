import re

reCRPfilename = re.compile(r'CRP_pcId_(.+)_pfId_(.+)_mthd_(.+)_dimn_(.+)_tdly_(.+)_nsz_(.+)_dsf_(.+)_sqln_(.+)\.npy')
reNCDfilename = re.compile(r'NCD_pc1Id_(.+)_pc1pfId_(.+)_pc2Id_(.+)_pc2pfId_(.+)_mthd_(.+)_dimn_(.+)_tdly_(.+)_nsz_(.+)_dsf_(.+)_feat_(.+)_sqln_(.+)\.pkl')
reNCDfnRoot = re.compile(r'NCD_pc1Id_(.+)_pc1pfId_(.+)_pc2Id_(.+)_pc2pfId_(.+)_mthd_(.+)_dimn_(.+)_tdly_(.+)_nsz_(.+)_dsf_(.+)_feat_(.+)_sqln_(.+)')