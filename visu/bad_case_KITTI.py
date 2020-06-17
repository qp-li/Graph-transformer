import numpy as np
import os
from tqdm import tqdm

sequence = "08"
dist_thresh = 10
prob = 0.1
list_dir = "/media/work/data/Ford/IJRR-Dataset-"+str(int(sequence)) +"/SG/"+str(dist_thresh)+"_20/"
list_path = "../eval/gt/"+ str(dist_thresh)+"_20/draw_curve_" +sequence+ ".npy"
output_path = "/media/work/3D/SimGNN/kx/SG_LC/visu/case_study/KITTI/" + sequence +"/" + str(dist_thresh) + "_20"
if not os.path.exists(output_path):
    os.makedirs(output_path)
pairs_files = np.load(list_path).tolist()

M2DP_dir = "/media/work/3D/SimGNN/kx/SG_LC/eval/compare/M2DP/eval/" + str(dist_thresh) + "_20/drop0"
SC_dir = "/media/work/3D/SimGNN/kx/SG_LC/eval/compare/SC/eval/" + str(dist_thresh) + "_20/drop0"
PV_PRE_dir = "/media/work/3D/SimGNN/kx/SG_LC/eval/compare/PV_PRE/eval/" + str(dist_thresh) + "_20/drop0"
PV_KITTI_dir = "/media/work/3D/SimGNN/kx/SG_LC/eval/compare/PV_KITTI/eval/" + str(dist_thresh) + "_20/drop0"
Ours_dir = "/media/work/3D/SimGNN/kx/SG_LC/eval/" + str(dist_thresh) + "_20/Ours_RN/drop0"

M2DP_gt = np.load(os.path.join(M2DP_dir,sequence,sequence+"_gt_db.npy"))
SC_gt = np.load(os.path.join(SC_dir,sequence,sequence+"_gt_db.npy"))
PV_PRE_gt = np.load(os.path.join(PV_PRE_dir,sequence,sequence+"_gt_db.npy"))
PV_KITTI_gt = np.load(os.path.join(PV_KITTI_dir,sequence,sequence+"_gt_db.npy"))
Ours_gt = np.load(os.path.join(Ours_dir,sequence, sequence+"_gt_db.npy"))

assert len(M2DP_gt) == len(SC_gt) and len(SC_gt) == len(Ours_gt) and len(Ours_gt) == len(PV_KITTI_gt) and\
       len(PV_KITTI_gt) == len(PV_PRE_gt) and len(PV_KITTI_gt) == len(pairs_files)

M2DP_db = np.load(os.path.join(M2DP_dir,sequence,sequence+"_M2DP_db.npy"))
SC_db = np.load(os.path.join(SC_dir,sequence,sequence+"_SC_db.npy"))
PV_PRE_db = np.load(os.path.join(PV_PRE_dir,sequence,sequence+"_PV_db.npy"))
PV_KITTI_db = np.load(os.path.join(PV_KITTI_dir,sequence,sequence+"_PV_db.npy"))
Ours_db = np.load(os.path.join(Ours_dir,sequence,sequence+"_DL_db.npy"))

pairs_num = M2DP_db.shape[0]

all_wrong = []
ours_right = []
SC_right = []
PV_right = []
M2DP_right = []
ours_wrong = []
# print(max(M2DP_db))
# print(min(M2DP_db))
# print(max(SC_db))
# print(min(SC_db))
# print(max(Ours_db))
# print(min(Ours_db))
# print(max(PV_PRE_db))
# print(min(PV_PRE_db))

M2DP_db = (M2DP_db - min(M2DP_db)) / max(M2DP_db)
SC_db = (SC_db - min(SC_db)) / max(SC_db)
Ours_db = (Ours_db - min(Ours_db)) / max(Ours_db)
PV_PRE_db = (PV_PRE_db - min(PV_PRE_db)) / max(PV_PRE_db)
PV_KITTI_db = (PV_KITTI_db - min(PV_KITTI_db)) / max(PV_KITTI_db)

for i in tqdm(range(pairs_num)):
    if M2DP_gt[i] == 1: # True
        if SC_db[i] < prob and M2DP_db[i] < prob and PV_PRE_db[i] < prob \
            and PV_KITTI_db[i] < prob and Ours_db[i] < prob:#SC_db[i] < prob and
            all_wrong.append(pairs_files[i])
            # print(M2DP_db[i])
        if SC_db[i] < prob and M2DP_db[i] < prob and PV_PRE_db[i] < prob \
            and PV_KITTI_db[i] < prob and Ours_db[i] > 0.9:#SC_db[i] < prob and
            ours_right.append(pairs_files[i])
        if SC_db[i] > 0.9 and M2DP_db[i] < prob and PV_PRE_db[i] < prob \
            and PV_KITTI_db[i] < prob and Ours_db[i] < prob:#SC_db[i] < prob and
            SC_right.append(pairs_files[i])

        if SC_db[i] < prob and M2DP_db[i] > 0.9 and PV_PRE_db[i] < prob \
            and PV_KITTI_db[i] < prob and Ours_db[i] < prob:#SC_db[i] < prob and
            M2DP_right.append(pairs_files[i])

        if SC_db[i] < prob and M2DP_db[i] < prob and PV_PRE_db[i] > 0.9 \
                and PV_KITTI_db[i] > 0.9 and Ours_db[i] < prob:  # SC_db[i] < prob and
            PV_right.append(pairs_files[i])


outfile = os.path.join(output_path,"all_wrong.txt")
print("all wrong nums: ", len(all_wrong))
file=open(outfile,'w')
file.write(str(all_wrong))
file.close()

outfile = os.path.join(output_path,"ours_right.txt")
print("ours_right nums: ", len(ours_right))
file=open(outfile,'w')
file.write(str(ours_right))
file.close()

outfile = os.path.join(output_path,"SC_right.txt")
print("SC_right nums: ", len(SC_right))
file=open(outfile,'w')
file.write(str(SC_right))
file.close()

outfile = os.path.join(output_path,"M2DP_right.txt")
print("M2DP_right nums: ", len(M2DP_right))
file=open(outfile,'w')
file.write(str(M2DP_right))
file.close()


outfile = os.path.join(output_path,"PV_right.txt")
print("PV_right nums: ", len(PV_right))
file=open(outfile,'w')
file.write(str(PV_right))
file.close()
