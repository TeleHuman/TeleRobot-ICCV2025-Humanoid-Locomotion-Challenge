### Challenge Link ###
[Challenge Official Website](https://human-robot-scene.github.io/Terrain-Challenge/#scenarios)

### Installation ###
```bash
conda create -n terrain python=3.8
conda activate terrain
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   #or cu113,cu115,cu121, based on your cuda version

git clone https://github.com/shiki-ta/Humanoid-Terrain-Bench.git
cd Humanoid-Terrain-Bench
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd rsl_rl && pip install -e .
cd legged_gym && pip install -e .
cd challenging_terrain && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

### Usage ###
`cd legged_gym/scripts`

1. Set both first_stage flag in combine_terrain.py & envs/{robot}/{robot}.py to __True__. Train 1st stage base policy on flat terrain(Robots are able to walk after around 1000 iterations.):  
We have released first stage base policy for all humanoid platforms.
```
python train.py --exptid h1-2 --device cuda:0 --headless --task h1_2_fix
```


2. Set both first_stage flag to **False**. Training Recovery 2nd stage on multi-terrains:
```
python train.py --exptid h1-2 --device cuda:0 --resume --resumeid=test --checkpoint=1000--headless --task h1_2_fix
```

3. Play the policy:
```
python play.py --exptid test --task h1_2_fix
```



### Arguments ###
- --exptid: string,  to describe the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --no_wandb: no wandb logging.
- --save: make dataset

### Acknowledgement ###

[legged_gym](https://github.com/leggedrobotics/legged_gym)

[Isaac Gym](https://developer.nvidia.com/isaac-gym)

[extreme parkour](https://github.com/chengxuxin/extreme-parkour)

### Citation
If you found any part of this code useful, please consider citing:
```
@article{fan2025one,
  title={One Policy but Many Worlds: A Scalable Unified Policy for Versatile Humanoid Locomotion},
  author={Fan, Yahao and Gui, Tianxiang and Ji, Kaiyang and Ding, Shutong and Zhang, Chixuan and Gu, Jiayuan and Yu, Jingyi and Wang, Jingya and Shi, Ye},
  journal={arXiv preprint arXiv:2505.18780},
  year={2025}
}
```
