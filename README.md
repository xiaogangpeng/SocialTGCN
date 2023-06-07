# The Baseline Codes for the paper "The MI-Motion Dataset and Benchmark for 3D Multi-Person Motion Prediction"
> [Xiaogang Peng](https://xiaogangpeng.github.io/), [Xiao Zhou](https://kimrrrich.github.io/), [Yikai Luo](https://lyk0520.github.io/), Hao Wen, [Zizhao Wu*](http://zizhao.me)

<!-- [Project Page](https://xiaogangpeng.github.io/SocialTGCN/) |  -->
### [Paper](https://arxiv.org/abs/2303.05095) | [Video](https://) | [Data](https://) | [Project Page](https://mi-motion.github.io/)
<br/>

## News
- [2022/06/01]: The codes of SocialTGCN and other baselines are released. 

[//]: # (- [2022/2/28]: Our paper is accepted by CVPR 2023. Thanks to my collaborators！)



## Pipeline of SocialTGCN


## Getting Started


The MI-Motion dataset can be downloaded from [![Google Drive](https://img.shields.io/badge/Google-Drive-blue)](https://drive.google.com/file/d/1HM7pwrT_hxpqgjicAbhCKK45hTvWnC6F/view?usp=sharing) and [![Baidu Disk](https://img.shields.io/badge/Baidu_Disk-PWD:2v41-white)](https://pan.baidu.com/s/1KT0YRxbcqYoyremod-0T7Q). You can also download the pretrained models of all the baselines in [![Google Drive](https://img.shields.io/badge/Google-Drive-blue)](https://drive.google.com/drive/folders/13ZD0BHzADmWzkXs_lxZdbhULLlsIMwkb?usp=sharing). More details could be found in the [Project Page](https://mi-motion.github.io/).
<br/>



## Prepare Dataset
After download the dataset, please prepare your data like this:
```
your_project_folder/
├── data/
│   ├── MI-Motion
│   │   ├── S0
│   │   ├── S1
│   │   ├── S2
│   │   ├── S3
│   │   ├── S4
│   ├── ├── ...
│   ├── preprocess_data.py
│   ├── ...
├── baselines/
│   ├── ...
├── util/
│   ├── ...

```

## Preprocsess Data
```
cd data
python preprocess_data.py
```

## Training
For any baseline method: 
```
python baselines/train_{method}.py
python baselines/train_hri.py   #  example of training for HRI baseline
```
## Evaluation
For any baseline method: 
```
python baselines/eval_{method}.py
```
If you want evaluation for ultra-long-term prediction, use:
```
python baselines/eval_{method}.py --ultra-long 1
```

## Visualization
```
python baselines/eval_{method}.py --vis 1  # for short-term and long term prediction
python baselines/eval_{method}.py --vis 1 --ultra-long 1 # for ultra-long-term prediction 
```
The rendered PNGs and GIFs are automatically saved in output folder of each baseline.



## Acknowledgement
Many thanks to the previous works:
- [HRI](https://github.com/wei-mao-2019/HisRepItself)
- [MRT](https://github.com/jiashunwang/MRT)
- [TBIFormer](https://github.com/xiaogangpeng/TBIFormer)


## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@misc{xx,
      title={The MI-Motion Dataset and Benchmark for 3D Multi-Person Motion Prediction}, 
      author={Xiaogang Peng and Xiao Zhou and Yikai Luo and Hao Wen and Zizhao Wu},
      year={2023},
      eprint={xx.xx},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## License
MIT
