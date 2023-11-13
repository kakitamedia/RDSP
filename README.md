# Context-aware Region-dependent Scale Proposals for Scale-optimized Object Detection using Super-Resolution
This repository contains the code for the [Context-aware Region-dependent Scale Proposals for Scale-optimized Object Detection using Super-Resolution]()


## Abstract

Image scaling techniques such as Super-Resolution (SR) are useful for object detection, especially for detecting small objects. However, we found that scaling by an inappropriate factor tends to induce false-positive detections. This paper presents a Region-Dependent Scale-Proposal (RDSP) network that estimates the appropriate scale factors for each image region depending on its contextual information. In our detection framework, images are appropriately scaled by SR according to the estimations of the RDSP network, and fed into the scale-specific object detectors. While previous works have proposed models for scale proposal, our RDSP extracts regions where objects could potentially exist based on scene structure, regardless of whether actual objects are present, because small objects are often too small to determine their presence accurately. Additionally, while existing approaches have fused object detection and SR in an end-to-end manner, scale proposals for SR are not provided or are performed independently. Qualitative and quantitative experiments show that our RDSP network provides appropriate SR scales and improve detection accuracy on highly challenging dataset, captured by real car-mounted cameras with size-varied objects, including extremely small objects.


## Key Requirements

- Python 3.10
- CUDA 11.7
- PyTorch 11.3.0
- [mmdetection](https://github.com/open-mmlab/mmdetection) 2.26.0
- Install requirements by `pip install -r requirements.txt`


## Data Preparation

Download the CityScapes dataset from [here](https://www.cityscapes-dataset.com/) and extract, and then, move them to the dataset folder. The directory should be like this.

<pre>
.
└ datasets
    └ CityScapes
        ├ gtFine
        └ leftImg8bit
</pre>

The CityScapes dataset need to be converted into the coco format. In addition, for our training, mask GTs should be created. Run the following command.

```
python shell/convert_dataset.py
```

## Testing

You can download our pretrained models [here]() and move them into the weights folder. And then, run the following command.

```
python shell/test.py --base
```

If the above command fails to run due to insufficient GPU memory, you can perform inference by patching the image using the following command. Please note that the inference speed will be slower.

```
python shell/test.py --base --split_forward
```

## Training



```
python shell/train.py
```

```
python shell/test.py
```


## Citation and License

'''
@ARTICLE{10304134,
  author={Akita, Kazutoshi and Ukita, Norimichi},
  journal={IEEE Access},
  title={Context-Aware Region-Dependent Scale Proposals for Scale-Optimized Object Detection Using Super-Resolution},
  year={2023},
  volume={11},
  number={},
  pages={122141-122153},
  doi={10.1109/ACCESS.2023.3329302}}
'''