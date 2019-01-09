# STAMP
---
## Paper code and data

This is the code for the KDD 2018 paper: [STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation](https://dl.acm.org/citation.cfm?id=3219950). We have implemented our methods in **Tensorflow**.

These are two datasets we used in our paper. After download them, you can put them in the folder `datas\`, then process them by `process_rsc.py` and  `process_cikm.py` respectively.

YOOCHOOSE: http://2015.recsyschallenge.com/challenge.html

DIGINETICA: http://cikm2016.cs.iupui.edu/cikm-cup

---

## Usage

Beacuse for each dataset we have some different parameters, there are two model files `STAMP_rsc.py` and `STAMP_cikm.py`.

So you run the file`cmain.py` to train the model.

For example: `python3 cmain.py -m stamp_rsc -d rsc15_64 -n` and `python3 cmain.py -m stamp_cikm -d cikm16 -n`

Or you can run it by using the `run.sh` directly. 

---
## Requirements

. Python 3
. Tensorflow 1.4

---

## Citation
Please cite our papaer:
```
@inproceedings{Liu18STAMP,
 author = {Liu, Qiao and Zeng, Yifu and Mokhosi, Refuoe and Zhang, Haibin},
 title = {STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation},
 booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \&\#38; Data Mining},
 year = {2018},
 location = {London, United Kingdom},
 pages = {1831--1839},
} 
```

