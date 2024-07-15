## AutoProPos: An Extension of Prototype Scattering and Positive Sampling Clustering for an Unknown Number of Clusters

This project implements AutoProPos, an adaptive clustering algorithm based on Prototype Scattering and Positive Sampling. AutoProPos project is  builds upon the code from the ([ProPos](https://github.com/Hzzone/ProPos?tab=readme-ov-file)) repository. We have introduced modifications and extensions to incorporate the adaptive clustering capabilities of AutoProPos impoving the original code's performance and functionality.

**2. Installation**

To install AutoProPos:


1. **Clone** or **Download** the code from the repository .
2. **Install dependencies:** Install the libraries listed in the `requirements.txt` file using pip:
   ```bash
   pip -r install requirements.txt
   ```

**3. Datasets**

The datasets used to train AutoProPos (MNIST, Fashion-MNIST, ImageNet10, and ImageNet-Dogs) can be downloaded from the following Google Drive link:

[https://drive.google.com/drive/folders/1-k7ZDa7BGApP0GFR2VJRj-cGI2NKKU8_?usp=drive_link](https://drive.google.com/drive/folders/1-k7ZDa7BGApP0GFR2VJRj-cGI2NKKU8_?usp=drive_link)

Place the downloaded datasets in the `/AutoProPos` folder. Datasets not included in this download will be downloaded automatically during the training script execution.

**4. Prediction**

**4.1 Pretrained Models**

Pre-trained models for specific datasets are available at the following Google Drive link:

[https://drive.google.com/file/d/1fdlBvI89DfrQ-YReZLKB9HM8L-Is5qWf/view?usp=drive_link](https://drive.google.com/file/d/1fdlBvI89DfrQ-YReZLKB9HM8L-Is5qWf/view?usp=drive_link)

Place the downloaded models in the `/AutoProPos` folder.

**4.2 Running Predictions**

To use a pre-trained model for prediction on a dataset specified in our paper, run the following command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set comma-separated GPU device IDs (if applicable)
torchrun --master_port 17673 --nproc_per_node=4 main.py config_best_models/mnist.yml
```

**5. Training**

To train AutoProPos models on a dataset specified in our paper, run the following command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set comma-separated GPU device IDs (if applicable)
torchrun --master_port 17673 --nproc_per_node=4 main.py config/mnist.yml
```


**Note:**

* Replace `mnist.yml` with the configuration file for your desired dataset (located in the `config` or `config_best_models` directories).
* Adjust the `CUDA_VISIBLE_DEVICES` environment variable and `--nproc_per_node` argument according to your GPU configuration.
