#!/bin/bash
#SBATCH -p barbun-cuda        	# Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A mtuzun         	# Kullanici adi
#SBATCH -J agent4_task     # Gonderilen isin ismi
#SBATCH --dependency singleton
#SBATCH -o experiment4.out    	# Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        	# Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                	# Gorev kac node'da calisacak?
#SBATCH -n 1                	# Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 20  	# Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=1-00:00:00      	# Sure siniri koyun.
#SBATCH --mail-user=e2035731@ceng.metu.edu.tr
#SBATCH --mail-type=ALL

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate dl-env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

cd /truba/home/$USER/agent4/al_ids
config_folder_path=$1
./run_experiments_4.sh run_test_fast2.py $config_folder_path > run_experiments.log 2>&1
