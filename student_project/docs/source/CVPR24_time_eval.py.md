# CVPR24_time_eval.py

```python
   1 | """
   2 | The code was adapted from the MICCAI FLARE Challenge
   3 | https://github.com/JunMa11/FLARE
   4 | 
   5 | The testing images will be evaluated one by one.
   6 | 
   7 | Folder structure:
   8 | CVPR24_time_eval.py
   9 | - team_docker
  10 |     - teamname.tar.gz # submitted docker containers from participants
  11 | - test_demo
  12 |     - imgs
  13 |         - case1.npz  # testing image
  14 |         - case2.npz  
  15 |         - ...   
  16 | - demo_seg  # segmentation results
  17 |     - case1.npz  # segmentation file name is the same as the testing image name
  18 |     - case2.npz  
  19 |     - ...
  20 | """
  21 | 
  22 | import os
  23 | join = os.path.join
  24 | import shutil
  25 | import time
  26 | import torch
  27 | import argparse
  28 | from collections import OrderedDict
  29 | import pandas as pd
  30 | 
  31 | parser = argparse.ArgumentParser('Segmentation efficiency eavluation for docker containers', add_help=False)
  32 | parser.add_argument('-i', '--test_img_path', default='./test_demo/imgs', type=str, help='testing data path')
  33 | parser.add_argument('-o','--save_path', default='./demo_seg', type=str, help='segmentation output path')
  34 | parser.add_argument('-d','--docker_folder_path', default='./team_docker', type=str, help='team docker path')
  35 | args = parser.parse_args()
  36 | 
  37 | test_img_path = args.test_img_path
  38 | save_path = args.save_path
  39 | docker_path = args.docker_folder_path
  40 | 
  41 | input_temp = './inputs/'
  42 | output_temp = './outputs'
  43 | os.makedirs(save_path, exist_ok=True)
  44 | 
  45 | dockers = sorted(os.listdir(docker_path))
  46 | test_cases = sorted(os.listdir(test_img_path))
  47 | 
  48 | for docker in dockers:
  49 |     try:
  50 |         # create temp folers for inference one-by-one
  51 |         if os.path.exists(input_temp):
  52 |             shutil.rmtree(input_temp)
  53 |         if os.path.exists(output_temp):
  54 |             shutil.rmtree(output_temp)
  55 |         os.makedirs(input_temp)
  56 |         os.makedirs(output_temp)
  57 |         # load docker and create a new folder to save segmentation results
  58 |         teamname = docker.split('.')[0].lower()
  59 |         print('teamname docker: ', docker)
  60 |         os.system('docker image load -i {}'.format(join(docker_path, docker)))
  61 |         team_outpath = join(save_path, teamname)
  62 |         if os.path.exists(team_outpath):
  63 |             shutil.rmtree(team_outpath)
  64 |         os.mkdir(team_outpath)
  65 |         os.system('chmod -R 777 ./* ')
  66 |         metric = OrderedDict()
  67 |         metric['CaseName'] = []
  68 |         metric['RunningTime'] = []
  69 |         # To obtain the running time for each case, testing cases are inferred one-by-one
  70 |         for case in test_cases:
  71 |             shutil.copy(join(test_img_path, case), input_temp)
  72 |             cmd = 'docker container run -m 8G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, teamname)
  73 |             print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)
  74 |             start_time = time.time()
  75 |             os.system(cmd)
  76 |             real_running_time = time.time() - start_time
  77 |             print(f"{case} finished! Inference time: {real_running_time}")
  78 |             # save metrics
  79 |             metric['CaseName'].append(case)
  80 |             metric['RunningTime'].append(real_running_time)
  81 |             os.remove(join(input_temp, case))  
  82 |             seg_name = case
  83 |             try:
  84 |                 os.rename(join(output_temp, seg_name), join(team_outpath, seg_name))
  85 |             except:
  86 |                 print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
  87 |                 print("Wrong segmentation name!!! It should be the same as image_name")
  88 |         metric_df = pd.DataFrame(metric)
  89 |         metric_df.to_csv(join(team_outpath, teamname + '_running_time.csv'), index=False)
  90 |         torch.cuda.empty_cache()
  91 |         os.system("docker rmi {}:latest".format(teamname))
  92 |         shutil.rmtree(input_temp)
  93 |         shutil.rmtree(output_temp)
  94 |     except Exception as e:
  95 |         print(e)
```
