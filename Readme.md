# Introduction
- This is a self-learning project, the main purpose is to create a simple api to retrive informations from a vehicle registration card
- This api will receive a image of a vehicle registration card and return a json with the informations

# How to use
## 1. Setup server
### 1.1. Basic setup
- Before you start to handle the basic setup, make sure that you have installed the following tools:
    - [Git](https://git-scm.com/downloads) # To clone the repository
    - [Cuda version 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) # To run the model using GPU
    - [Anaconda](https://www.anaconda.com/products/individual) # To create a virtual environment
- Now, you can start to handle the basic setup
    - Clone the repository
        ```bash
        git clone https://github.com/DuypntTN/Vietnames-Cavet-OCR.git
        ```
    - Create a virtual environment
        ```bash
        conda create -n <env_name>
        ```
    - Activate the virtual environment
        ```bash
        conda activate <env_name>
        ```
    - Install the requirements
        ```bash
        pip install -r requirements.txt
        ```
- This is an extreamly important step - install pytorch😄, cus this repository mainly run on pytorch so without it, you can't run the model acttually
    - Go to [pytorch](https://pytorch.org/get-started/locally/) and install the version that match your system
    - For me, because my cuda is **11.7** so that I installed this version:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

## 2. Use the api
(Updating...)