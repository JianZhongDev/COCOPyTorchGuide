# COCOPyTorchGuide

Here's a plain language version:

This guide will show you how to set up the COCO dataset for PyTorch, step by step.

This repository also includes a PyTorch COCO dataset class that:
- Downloads only the necessary categories to save storage space.
- Applies identical random transformations to both images and labels.
- Offers various label formatting options.

## Quick Set Up Guide and Demo Notebooks
1. Got to [COCO webiste](https://cocodataset.org/#download) and obtain the url for annotation file.
2. `pip install pycocotools`
3. Use the [PythonDownloadAndUnzip](./PythonDownloadAndUnzip.ipynb) notebook to download and unzip the annotation file.
4. Use the [ExploreCOCOAPI](./ExploreCOCOAPI.ipynb) to have a quick understanding of the key functions provided by the COCO API.
5. The [COCOSegDatasetTest](./COCOSegDatasetTest.ipynb) notebook includes a quick demo on how to use the customized COCO PyTorch dataset provided in this repository.

## Dependency
This repo has been implemented and tested on the following dependencies:
- Python 3.10.13
- requests 2.31.0
- matplotlib 3.8.2
- numpy 1.26.2
- torch 2.1.1+cu118
- torchvision 0.16.1+cu118
- pycocotools 2.0.8
- notebook 7.0.6

## Computer Requirement
This repo has been tested on a laptop computer with the following specs:
- CPU: Intel(R) Core(TM) i7-9750H CPU
- Memory: 32GB 
- GPU: NVIDIA GeForce RTX 2060

## License

[GPL-3.0 license](./LICENSE)

## Reference

[1] Lin, T.-Y. et al. Microsoft COCO: Common Objects in context. in Lecture notes in computer science 740–755 (2014). doi:10.1007/978-3-319-10602-1_48.

## Resources

COCO API website: https://cocodataset.org/#home

## Citation
