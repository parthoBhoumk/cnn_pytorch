{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "5366c285",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torchvision       # give access to datasets,model architectures and image transformation for computer vision\n",
    "import torchvision.transforms as transforms  #for image preocessing\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2cf4ab90",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100.0%"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz to ./data/FashionMNIST/raw\n",
      "Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "111.0%"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw\n",
      "Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100.0%"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST/raw\n",
      "Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "159.1%"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw\n",
      "Processing...\n",
      "Done!\n"
     ]
    }
   ],
   "source": [
    "#loading fasionmnist dataset\n",
    "\n",
    "train_set = torchvision.datasets.FashionMNIST(\n",
    "    root = './data'\n",
    "    ,train = True\n",
    "    ,download=True\n",
    "    ,transform = transforms.Compose([\n",
    "        transforms.ToTensor()\n",
    "    ])\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "24c067ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Dataset FashionMNIST\n",
       "    Number of datapoints: 60000\n",
       "    Root location: ./data\n",
       "    Split: Train\n",
       "    StandardTransform\n",
       "Transform: Compose(\n",
       "               ToTensor()\n",
       "           )"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "9f53b41f",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loader = torch.utils.data.DataLoader(train_set,\n",
    "\n",
    "                                           \n",
    "                batch_size  = 100,shuffle = True\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "d32877da",
   "metadata": {},
   "outputs": [],
   "source": [
    "a = []\n",
    "for i in train_loader:\n",
    "    a.append(i)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "520bd71c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6000"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "47ebd12f",
   "metadata": {},
   "outputs": [],
   "source": [
    "torch.set_printoptions(linewidth=120)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "b22d0214",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "60000"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(train_set)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "aed91b08",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([9, 0, 0,  ..., 3, 0, 5])"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_set.train_labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "f4ae1bbc",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[9,\n",
       " 0,\n",
       " 0,\n",
       " 3,\n",
       " 0,\n",
       " 2,\n",
       " 7,\n",
       " 2,\n",
       " 5,\n",
       " 5,\n",
       " 0,\n",
       " 9,\n",
       " 5,\n",
       " 5,\n",
       " 7,\n",
       " 9,\n",
       " 1,\n",
       " 0,\n",
       " 6,\n",
       " 4,\n",
       " 3,\n",
       " 1,\n",
       " 4,\n",
       " 8,\n",
       " 4,\n",
       " 3,\n",
       " 0,\n",
       " 2,\n",
       " 4,\n",
       " 4,\n",
       " 5,\n",
       " 3,\n",
       " 6,\n",
       " 6,\n",
       " 0,\n",
       " 8,\n",
       " 5,\n",
       " 2,\n",
       " 1,\n",
       " 6,\n",
       " 6,\n",
       " 7,\n",
       " 9,\n",
       " 5,\n",
       " 9,\n",
       " 2,\n",
       " 7,\n",
       " 3,\n",
       " 0,\n",
       " 3,\n",
       " 3,\n",
       " 3,\n",
       " 7,\n",
       " 2,\n",
       " 2,\n",
       " 6,\n",
       " 6,\n",
       " 8,\n",
       " 3,\n",
       " 3,\n",
       " 5,\n",
       " 0,\n",
       " 5,\n",
       " 5,\n",
       " 0,\n",
       " 2,\n",
       " 0,\n",
       " 0,\n",
       " 4,\n",
       " 1,\n",
       " 3,\n",
       " 1,\n",
       " 6,\n",
       " 3,\n",
       " 1,\n",
       " 4,\n",
       " 4,\n",
       " 6,\n",
       " 1,\n",
       " 9,\n",
       " 1,\n",
       " 3,\n",
       " 5,\n",
       " 7,\n",
       " 9,\n",
       " 7,\n",
       " 1,\n",
       " 7,\n",
       " 9,\n",
       " 9,\n",
       " 9,\n",
       " 3,\n",
       " 2,\n",
       " 9,\n",
       " 3,\n",
       " 6,\n",
       " 4,\n",
       " 1,\n",
       " 1,\n",
       " 8,\n",
       " 8,\n",
       " 0,\n",
       " 1,\n",
       " 1,\n",
       " 6,\n",
       " 8,\n",
       " 1,\n",
       " 9,\n",
       " 7,\n",
       " 8,\n",
       " 8,\n",
       " 9,\n",
       " 6,\n",
       " 6,\n",
       " 3,\n",
       " 1,\n",
       " 5,\n",
       " 4,\n",
       " 6,\n",
       " 7,\n",
       " 5,\n",
       " 5,\n",
       " 9,\n",
       " 2,\n",
       " 2,\n",
       " 2,\n",
       " 7,\n",
       " 6,\n",
       " 4,\n",
       " 1,\n",
       " 8,\n",
       " 7,\n",
       " 7,\n",
       " 5,\n",
       " 4,\n",
       " 2,\n",
       " 9,\n",
       " 1,\n",
       " 7,\n",
       " 4,\n",
       " 6,\n",
       " 9,\n",
       " 7,\n",
       " 1,\n",
       " 8,\n",
       " 7,\n",
       " 1,\n",
       " 2,\n",
       " 8,\n",
       " 0,\n",
       " 9,\n",
       " 1,\n",
       " 8,\n",
       " 7,\n",
       " 0,\n",
       " 5,\n",
       " 8,\n",
       " 6,\n",
       " 7,\n",
       " 2,\n",
       " 0,\n",
       " 8,\n",
       " 7,\n",
       " 1,\n",
       " 6,\n",
       " 2,\n",
       " 1,\n",
       " 9,\n",
       " 6,\n",
       " 0,\n",
       " 1,\n",
       " 0,\n",
       " 5,\n",
       " 5,\n",
       " 1,\n",
       " 7,\n",
       " 0,\n",
       " 5,\n",
       " 8,\n",
       " 4,\n",
       " 0,\n",
       " 4,\n",
       " 0,\n",
       " 6,\n",
       " 6,\n",
       " 4,\n",
       " 0,\n",
       " 0,\n",
       " 4,\n",
       " 7,\n",
       " 3,\n",
       " 0,\n",
       " 5,\n",
       " 8,\n",
       " 4,\n",
       " 1,\n",
       " 1,\n",
       " 2,\n",
       " 9,\n",
       " 2,\n",
       " 8,\n",
       " 5,\n",
       " 0,\n",
       " 6,\n",
       " 3,\n",
       " 4,\n",
       " 6,\n",
       " 0,\n",
       " 9,\n",
       " 1,\n",
       " 7,\n",
       " 3,\n",
       " 8,\n",
       " 5,\n",
       " 8,\n",
       " 3,\n",
       " 8,\n",
       " 5,\n",
       " 2,\n",
       " 0,\n",
       " 8,\n",
       " 7,\n",
       " 0,\n",
       " 3,\n",
       " 5,\n",
       " 0,\n",
       " 6,\n",
       " 5,\n",
       " 2,\n",
       " 7,\n",
       " 5,\n",
       " 2,\n",
       " 6,\n",
       " 8,\n",
       " 2,\n",
       " 6,\n",
       " 8,\n",
       " 0,\n",
       " 4,\n",
       " 4,\n",
       " 4,\n",
       " 4,\n",
       " 4,\n",
       " 1,\n",
       " 5,\n",
       " 6,\n",
       " 5,\n",
       " 3,\n",
       " 3,\n",
       " 7,\n",
       " 3,\n",
       " 3,\n",
       " 6,\n",
       " 2,\n",
       " 8,\n",
       " 4,\n",
       " 6,\n",
       " 5,\n",
       " 9,\n",
       " 3,\n",
       " 2,\n",
       " 3,\n",
       " 2,\n",
       " 4,\n",
       " 4,\n",
       " 8,\n",
       " 2,\n",
       " 5,\n",
       " 3,\n",
       " 0,\n",
       " 7,\n",
       " 2,\n",
       " 0,\n",
       " 2,\n",
       " 5,\n",
       " 7,\n",
       " 2,\n",
       " 3,\n",
       " 1,\n",
       " 7,\n",
       " 6,\n",
       " 2,\n",
       " 9,\n",
       " 1,\n",
       " 9,\n",
       " 1,\n",
       " 1,\n",
       " 8,\n",
       " 7,\n",
       " 8,\n",
       " 4,\n",
       " 2,\n",
       " 6,\n",
       " 6,\n",
       " 7,\n",
       " 9,\n",
       " 4,\n",
       " 6,\n",
       " 1,\n",
       " 9,\n",
       " 5,\n",
       " 6,\n",
       " 0,\n",
       " 5,\n",
       " 0,\n",
       " 1,\n",
       " 6,\n",
       " 1,\n",
       " 1,\n",
       " 6,\n",
       " 7,\n",
       " 4,\n",
       " 4,\n",
       " 8,\n",
       " 6,\n",
       " 4,\n",
       " 6,\n",
       " 9,\n",
       " 3,\n",
       " 7,\n",
       " 5,\n",
       " 0,\n",
       " 8,\n",
       " 3,\n",
       " 4,\n",
       " 0,\n",
       " 3,\n",
       " 3,\n",
       " 2,\n",
       " 0,\n",
       " 1,\n",
       " 0,\n",
       " 3,\n",
       " 8,\n",
       " 3,\n",
       " 9,\n",
       " 1,\n",
       " 9,\n",
       " 0,\n",
       " 4,\n",
       " 7,\n",
       " 7,\n",
       " 8,\n",
       " 5,\n",
       " 6,\n",
       " 5,\n",
       " 6,\n",
       " 8,\n",
       " 2,\n",
       " 5,\n",
       " 2,\n",
       " 3,\n",
       " 1,\n",
       " 6,\n",
       " 0,\n",
       " 7,\n",
       " 8,\n",
       " 7,\n",
       " 8,\n",
       " 1,\n",
       " 9,\n",
       " 6,\n",
       " 4,\n",
       " 5,\n",
       " 7,\n",
       " 1,\n",
       " 7,\n",
       " 6,\n",
       " 6,\n",
       " 7,\n",
       " 3,\n",
       " 5,\n",
       " 8,\n",
       " 7,\n",
       " 3,\n",
       " 3,\n",
       " 9,\n",
       " 0,\n",
       " 3,\n",
       " 1,\n",
       " 6,\n",
       " 4,\n",
       " 7,\n",
       " 0,\n",
       " 5,\n",
       " 1,\n",
       " 5,\n",
       " 4,\n",
       " 4,\n",
       " 5,\n",
       " 9,\n",
       " 1,\n",
       " 0,\n",
       " 5,\n",
       " 8,\n",
       " 3,\n",
       " 4,\n",
       " 4,\n",
       " 2,\n",
       " 4,\n",
       " 2,\n",
       " 5,\n",
       " 6,\n",
       " 7,\n",
       " 2,\n",
       " 2,\n",
       " 5,\n",
       " 3,\n",
       " 8,\n",
       " 8,\n",
       " 6,\n",
       " 8,\n",
       " 4,\n",
       " 4,\n",
       " 1,\n",
       " 0,\n",
       " 2,\n",
       " 7,\n",
       " 1,\n",
       " 1,\n",
       " 8,\n",
       " 8,\n",
       " 2,\n",
       " 7,\n",
       " 9,\n",
       " 7,\n",
       " 4,\n",
       " 1,\n",
       " 2,\n",
       " 0,\n",
       " 8,\n",
       " 9,\n",
       " 1,\n",
       " 4,\n",
       " 9,\n",
       " 5,\n",
       " 6,\n",
       " 7,\n",
       " 0,\n",
       " 3,\n",
       " 2,\n",
       " 0,\n",
       " 4,\n",
       " 1,\n",
       " 0,\n",
       " 0,\n",
       " 8,\n",
       " 7,\n",
       " 0,\n",
       " 2,\n",
       " 9,\n",
       " 9,\n",
       " 1,\n",
       " 3,\n",
       " 8,\n",
       " 3,\n",
       " 1,\n",
       " 4,\n",
       " 8,\n",
       " 5,\n",
       " 1,\n",
       " 8,\n",
       " 2,\n",
       " 4,\n",
       " 4,\n",
       " 9,\n",
       " 5,\n",
       " 5,\n",
       " 4,\n",
       " 5,\n",
       " 6,\n",
       " 3,\n",
       " 7,\n",
       " 4,\n",
       " 9,\n",
       " 5,\n",
       " 8,\n",
       " 9,\n",
       " 3,\n",
       " 9,\n",
       " 6,\n",
       " 4,\n",
       " 7,\n",
       " 2,\n",
       " 2,\n",
       " 6,\n",
       " 4,\n",
       " 8,\n",
       " 3,\n",
       " 0,\n",
       " 2,\n",
       " 8,\n",
       " 9,\n",
       " 0,\n",
       " 8,\n",
       " 6,\n",
       " 3,\n",
       " 9,\n",
       " 1,\n",
       " 6,\n",
       " 3,\n",
       " 2,\n",
       " 6,\n",
       " 1,\n",
       " 0,\n",
       " 2,\n",
       " 6,\n",
       " 2,\n",
       " 3,\n",
       " 3,\n",
       " 9,\n",
       " 1,\n",
       " 7,\n",
       " 9,\n",
       " 1,\n",
       " 1,\n",
       " 0,\n",
       " 1,\n",
       " 7,\n",
       " 0,\n",
       " 8,\n",
       " 5,\n",
       " 0,\n",
       " 6,\n",
       " 0,\n",
       " 1,\n",
       " 6,\n",
       " 6,\n",
       " 8,\n",
       " 7,\n",
       " 0,\n",
       " 9,\n",
       " 9,\n",
       " 9,\n",
       " 7,\n",
       " 4,\n",
       " 3,\n",
       " 9,\n",
       " 5,\n",
       " 9,\n",
       " 1,\n",
       " 4,\n",
       " 1,\n",
       " 6,\n",
       " 2,\n",
       " 2,\n",
       " 2,\n",
       " 7,\n",
       " 7,\n",
       " 6,\n",
       " 3,\n",
       " 7,\n",
       " 5,\n",
       " 8,\n",
       " 6,\n",
       " 4,\n",
       " 8,\n",
       " 4,\n",
       " 7,\n",
       " 8,\n",
       " 6,\n",
       " 5,\n",
       " 9,\n",
       " 9,\n",
       " 0,\n",
       " 0,\n",
       " 2,\n",
       " 4,\n",
       " 3,\n",
       " 7,\n",
       " 9,\n",
       " 3,\n",
       " 1,\n",
       " 9,\n",
       " 2,\n",
       " 7,\n",
       " 5,\n",
       " 5,\n",
       " 4,\n",
       " 8,\n",
       " 2,\n",
       " 7,\n",
       " 8,\n",
       " 1,\n",
       " 2,\n",
       " 6,\n",
       " 6,\n",
       " 8,\n",
       " 1,\n",
       " 9,\n",
       " 6,\n",
       " 0,\n",
       " 0,\n",
       " 5,\n",
       " 9,\n",
       " 3,\n",
       " 3,\n",
       " 7,\n",
       " 6,\n",
       " 1,\n",
       " 0,\n",
       " 0,\n",
       " 6,\n",
       " 4,\n",
       " 7,\n",
       " 5,\n",
       " 9,\n",
       " 7,\n",
       " 9,\n",
       " 3,\n",
       " 8,\n",
       " 8,\n",
       " 4,\n",
       " 8,\n",
       " 5,\n",
       " 4,\n",
       " 2,\n",
       " 2,\n",
       " 7,\n",
       " 7,\n",
       " 5,\n",
       " 8,\n",
       " 4,\n",
       " 6,\n",
       " 9,\n",
       " 4,\n",
       " 3,\n",
       " 7,\n",
       " 8,\n",
       " 7,\n",
       " 8,\n",
       " 3,\n",
       " 7,\n",
       " 8,\n",
       " 4,\n",
       " 7,\n",
       " 7,\n",
       " 2,\n",
       " 2,\n",
       " 0,\n",
       " 0,\n",
       " 0,\n",
       " 3,\n",
       " 9,\n",
       " 1,\n",
       " 0,\n",
       " 9,\n",
       " 8,\n",
       " 4,\n",
       " 3,\n",
       " 9,\n",
       " 9,\n",
       " 9,\n",
       " 8,\n",
       " 8,\n",
       " 5,\n",
       " 4,\n",
       " 5,\n",
       " 4,\n",
       " 9,\n",
       " 8,\n",
       " 8,\n",
       " 0,\n",
       " 9,\n",
       " 2,\n",
       " 0,\n",
       " 7,\n",
       " 3,\n",
       " 7,\n",
       " 9,\n",
       " 3,\n",
       " 8,\n",
       " 4,\n",
       " 3,\n",
       " 7,\n",
       " 8,\n",
       " 1,\n",
       " 4,\n",
       " 0,\n",
       " 7,\n",
       " 9,\n",
       " 8,\n",
       " 5,\n",
       " 5,\n",
       " 2,\n",
       " 1,\n",
       " 3,\n",
       " 4,\n",
       " 6,\n",
       " 7,\n",
       " 7,\n",
       " 5,\n",
       " 9,\n",
       " 9,\n",
       " 7,\n",
       " 8,\n",
       " 2,\n",
       " 7,\n",
       " 4,\n",
       " 7,\n",
       " 0,\n",
       " 3,\n",
       " 5,\n",
       " 1,\n",
       " 1,\n",
       " 5,\n",
       " 5,\n",
       " 2,\n",
       " 8,\n",
       " 3,\n",
       " 5,\n",
       " 9,\n",
       " 0,\n",
       " 7,\n",
       " 3,\n",
       " 0,\n",
       " 0,\n",
       " 7,\n",
       " 1,\n",
       " 9,\n",
       " 4,\n",
       " 8,\n",
       " 9,\n",
       " 1,\n",
       " 8,\n",
       " 3,\n",
       " 4,\n",
       " 7,\n",
       " 7,\n",
       " 7,\n",
       " 1,\n",
       " 4,\n",
       " 0,\n",
       " 4,\n",
       " 5,\n",
       " 8,\n",
       " 8,\n",
       " 6,\n",
       " 5,\n",
       " 7,\n",
       " 1,\n",
       " 0,\n",
       " 2,\n",
       " 4,\n",
       " 9,\n",
       " 0,\n",
       " 9,\n",
       " 8,\n",
       " 6,\n",
       " 2,\n",
       " 8,\n",
       " 1,\n",
       " 4,\n",
       " 1,\n",
       " 1,\n",
       " 2,\n",
       " 3,\n",
       " 3,\n",
       " 8,\n",
       " 9,\n",
       " 5,\n",
       " 1,\n",
       " 6,\n",
       " 4,\n",
       " 5,\n",
       " 9,\n",
       " 6,\n",
       " 2,\n",
       " 8,\n",
       " 8,\n",
       " 7,\n",
       " 0,\n",
       " 1,\n",
       " 6,\n",
       " 5,\n",
       " 5,\n",
       " 6,\n",
       " 5,\n",
       " 4,\n",
       " 6,\n",
       " 0,\n",
       " 6,\n",
       " 1,\n",
       " 1,\n",
       " 5,\n",
       " 7,\n",
       " 6,\n",
       " 5,\n",
       " 4,\n",
       " 0,\n",
       " 6,\n",
       " 0,\n",
       " 7,\n",
       " 2,\n",
       " 8,\n",
       " 9,\n",
       " 1,\n",
       " 6,\n",
       " 7,\n",
       " 1,\n",
       " 9,\n",
       " 8,\n",
       " 3,\n",
       " 5,\n",
       " 7,\n",
       " 1,\n",
       " 1,\n",
       " 9,\n",
       " 0,\n",
       " 6,\n",
       " 4,\n",
       " 8,\n",
       " 0,\n",
       " 9,\n",
       " 5,\n",
       " 0,\n",
       " 0,\n",
       " 1,\n",
       " 8,\n",
       " 7,\n",
       " 6,\n",
       " 7,\n",
       " 9,\n",
       " 0,\n",
       " 3,\n",
       " 0,\n",
       " 1,\n",
       " 3,\n",
       " 8,\n",
       " 0,\n",
       " 0,\n",
       " 1,\n",
       " 7,\n",
       " 4,\n",
       " 7,\n",
       " 3,\n",
       " 7,\n",
       " 0,\n",
       " 6,\n",
       " 0,\n",
       " 2,\n",
       " 3,\n",
       " 4,\n",
       " 5,\n",
       " 9,\n",
       " 5,\n",
       " 2,\n",
       " 8,\n",
       " 7,\n",
       " 0,\n",
       " 1,\n",
       " 4,\n",
       " 1,\n",
       " 8,\n",
       " 3,\n",
       " 2,\n",
       " 7,\n",
       " 5,\n",
       " 0,\n",
       " 9,\n",
       " 6,\n",
       " 6,\n",
       " 9,\n",
       " 3,\n",
       " 1,\n",
       " 1,\n",
       " 9,\n",
       " 9,\n",
       " 3,\n",
       " 9,\n",
       " 7,\n",
       " 3,\n",
       " 1,\n",
       " 9,\n",
       " 1,\n",
       " 6,\n",
       " 3,\n",
       " 0,\n",
       " 4,\n",
       " 8,\n",
       " 0,\n",
       " 6,\n",
       " 6,\n",
       " 2,\n",
       " 9,\n",
       " 1,\n",
       " 9,\n",
       " 8,\n",
       " 3,\n",
       " 4,\n",
       " 7,\n",
       " 2,\n",
       " 0,\n",
       " 0,\n",
       " 8,\n",
       " 6,\n",
       " 9,\n",
       " 1,\n",
       " 2,\n",
       " 5,\n",
       " 2,\n",
       " 8,\n",
       " 5,\n",
       " 6,\n",
       " 7,\n",
       " 0,\n",
       " 6,\n",
       " 8,\n",
       " 7,\n",
       " 1,\n",
       " 0,\n",
       " 3,\n",
       " 4,\n",
       " 7,\n",
       " 5,\n",
       " 2,\n",
       " 5,\n",
       " 1,\n",
       " 1,\n",
       " 5,\n",
       " 7,\n",
       " 5,\n",
       " 1,\n",
       " 4,\n",
       " 9,\n",
       " 6,\n",
       " 7,\n",
       " 5,\n",
       " 7,\n",
       " 3,\n",
       " 8,\n",
       " 2,\n",
       " 9,\n",
       " 4,\n",
       " 2,\n",
       " 5,\n",
       " 5,\n",
       " 7,\n",
       " 4,\n",
       " 2,\n",
       " 6,\n",
       " 9,\n",
       " 0,\n",
       " 5,\n",
       " 3,\n",
       " 6,\n",
       " 9,\n",
       " 5,\n",
       " 3,\n",
       " 6,\n",
       " 0,\n",
       " 3,\n",
       " 5,\n",
       " 7,\n",
       " 1,\n",
       " 2,\n",
       " 5,\n",
       " 7,\n",
       " 8,\n",
       " 1,\n",
       " 6,\n",
       " 3,\n",
       " 9,\n",
       " 7,\n",
       " 7,\n",
       " 0,\n",
       " 4,\n",
       " 0,\n",
       " 8,\n",
       " 7,\n",
       " 6,\n",
       " 9,\n",
       " 0,\n",
       " 2,\n",
       " 7,\n",
       " 6,\n",
       " 0,\n",
       " 4,\n",
       " 7,\n",
       " 4,\n",
       " 1,\n",
       " 2,\n",
       " 6,\n",
       " 7,\n",
       " 8,\n",
       " 2,\n",
       " 5,\n",
       " 6,\n",
       " 0,\n",
       " 2,\n",
       " 9,\n",
       " 8,\n",
       " 5,\n",
       " 4,\n",
       " 5,\n",
       " 7,\n",
       " 3,\n",
       " 3,\n",
       " 9,\n",
       " 8,\n",
       " ...]"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_set.train_labels.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "c8f7dd43",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000])"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_set.train_labels.bincount()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "ab63a60d",
   "metadata": {},
   "outputs": [],
   "source": [
    "sample = next(iter(train_set))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "13787a67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(sample)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "b0b418e3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tuple"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(sample)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "d5a14e0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "image , label = sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "a17af8de",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "label: 9\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(image.squeeze(),cmap = 'gray')\n",
    "print('label:',label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "56b2b265",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "batch = next(iter(train_loader))\n",
    "\n",
    "len(batch)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "4e9fb8f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "list"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(batch)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "a0a6df8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "images , labels = batch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "bde451cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([100, 1, 28, 28])"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "images.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "0a91ee4c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([100])"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "labels.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "361a3717",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "labels: tensor([0, 4, 0, 1, 2, 2, 8, 6, 2, 1, 5, 3, 5, 6, 3, 6, 0, 6, 1, 3, 9, 0, 8, 9, 2, 2, 3, 4, 9, 3, 9, 3, 2, 5, 3, 9, 5,\n",
      "        9, 4, 9, 4, 4, 6, 2, 5, 1, 8, 5, 4, 2, 0, 8, 0, 0, 8, 4, 1, 2, 3, 3, 5, 7, 4, 1, 7, 7, 9, 1, 9, 8, 8, 3, 9, 3,\n",
      "        7, 2, 6, 8, 4, 2, 1, 4, 3, 0, 9, 5, 4, 4, 8, 2, 3, 8, 8, 7, 4, 5, 1, 0, 9, 1])\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x1080 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "grid = torchvision.utils.make_grid(images,nrow = 5)\n",
    "plt.figure(figsize = (15,15))\n",
    "plt.imshow(np.transpose(grid,(1,2,0)))\n",
    "print('labels:' , labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb00b5fc",
   "metadata": {},
   "source": [
    "# Building our Network"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e1a8928e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# a normal network class\n",
    "\n",
    "class Network:\n",
    "    def __init__(self):\n",
    "        self.layer = None\n",
    "        \n",
    "    def forward(self,t):\n",
    "        t = self.layer(t)\n",
    "        return t\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d6c5f400",
   "metadata": {},
   "outputs": [],
   "source": [
    "# but to use pytorch we have to extend the built in nn.Module class\n",
    "\n",
    "#also we need to insert a call to the super class constructor on line 3\n",
    "\n",
    "\n",
    "import torch.nn as nn\n",
    "class Network(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Network,self).__init__()\n",
    "        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5)\n",
    "        self.conv2 = nn.Conv2d(in_channels=6, out_channels= 12, kernel_size=5)\n",
    "        \n",
    "        self.fc1 = nn.Linear(in_features=12*4*4,out_features=120)\n",
    "        self.fc2 = nn.Linear(in_features=120, out_features=60)\n",
    "        self.out = nn.Linear(in_features=60, out_features=10)\n",
    "        \n",
    "        \n",
    "    def forward(self,t):\n",
    "        \n",
    "        return t\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "45663556",
   "metadata": {},
   "outputs": [],
   "source": [
    "mynet = Network()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "9c16e828",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Network(\n",
      "  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))\n",
      "  (conv2): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))\n",
      "  (fc1): Linear(in_features=192, out_features=120, bias=True)\n",
      "  (fc2): Linear(in_features=120, out_features=60, bias=True)\n",
      "  (out): Linear(in_features=60, out_features=10, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "print(Mynet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b9f58f5d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.conv1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "a2ae39d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.conv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "11656680",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Parameter containing:\n",
       "tensor([[[[ 1.2060e-01,  1.7459e-01, -1.6539e-01,  1.7060e-01, -1.6147e-01],\n",
       "          [-1.5316e-01, -1.0978e-01, -1.8079e-01, -1.3361e-02, -1.4004e-01],\n",
       "          [ 6.9702e-02,  5.4948e-02, -1.1968e-01, -7.6457e-02, -1.2225e-02],\n",
       "          [ 1.7515e-01,  1.2475e-01,  1.8246e-01, -6.9033e-02,  5.2208e-02],\n",
       "          [-7.0883e-02, -2.9680e-02,  5.5439e-02,  8.8118e-02,  1.6454e-01]]],\n",
       "\n",
       "\n",
       "        [[[-1.3239e-01,  1.9511e-01,  3.7720e-02, -4.2030e-03, -1.2303e-01],\n",
       "          [-8.9151e-02,  7.1255e-02,  1.8995e-01, -1.8860e-01,  1.7316e-01],\n",
       "          [ 3.1262e-02,  1.8847e-01, -1.2312e-01,  1.6273e-01,  9.8600e-02],\n",
       "          [-5.1952e-02, -1.9048e-01,  6.3603e-02,  1.2586e-01, -1.2946e-01],\n",
       "          [-1.6718e-01,  6.9044e-02, -1.1320e-03, -1.3361e-01,  1.3831e-02]]],\n",
       "\n",
       "\n",
       "        [[[ 1.0818e-01, -8.3349e-03, -4.6633e-02,  1.1497e-01,  1.5839e-01],\n",
       "          [ 7.2779e-02,  1.4849e-01, -4.7595e-03,  2.9349e-02, -6.5334e-02],\n",
       "          [-1.8055e-01,  1.2045e-01,  1.4111e-01, -1.7957e-01,  1.7858e-01],\n",
       "          [ 1.0241e-01, -6.3558e-02, -1.0759e-01, -1.2316e-01, -1.4400e-01],\n",
       "          [-1.1207e-01,  8.7119e-02,  3.3666e-02,  1.7109e-03,  1.0242e-01]]],\n",
       "\n",
       "\n",
       "        [[[-1.9836e-01,  1.1910e-01,  1.5129e-01, -1.5089e-01,  6.3623e-02],\n",
       "          [-2.8942e-02, -1.2680e-02,  1.6264e-01,  1.0983e-01, -1.3295e-01],\n",
       "          [-7.2747e-02, -1.0289e-01,  1.6946e-01,  1.2135e-01, -1.3156e-01],\n",
       "          [ 1.9744e-02,  6.6857e-02, -4.2915e-05, -5.0186e-02, -1.3678e-03],\n",
       "          [-1.1786e-01, -6.5076e-02,  9.4891e-02,  1.0689e-01,  1.9821e-01]]],\n",
       "\n",
       "\n",
       "        [[[-1.7589e-01,  6.3888e-02,  1.1068e-01, -1.0155e-01,  3.5640e-02],\n",
       "          [ 1.4759e-01,  1.3940e-01,  5.7485e-02,  1.3016e-01,  6.3809e-03],\n",
       "          [ 1.0984e-01, -8.3011e-02,  9.2547e-02, -5.1464e-02, -1.3530e-01],\n",
       "          [-1.5146e-01, -4.8514e-02,  5.9324e-02, -6.0047e-03, -1.5370e-01],\n",
       "          [-1.4979e-01, -1.6183e-01,  1.5313e-01,  7.8851e-02,  4.6473e-02]]],\n",
       "\n",
       "\n",
       "        [[[-1.7932e-01,  1.7520e-02,  1.5645e-01,  1.8202e-02,  1.4265e-01],\n",
       "          [-7.1278e-02,  1.6100e-01,  7.8073e-02,  1.4836e-02, -1.7650e-01],\n",
       "          [ 5.2575e-02, -4.0286e-02,  1.6935e-01, -2.7664e-02,  6.7886e-02],\n",
       "          [ 1.1751e-01,  8.9496e-05, -1.5928e-01,  8.8545e-02, -1.5634e-01],\n",
       "          [-3.2117e-02, -1.4361e-01, -4.4556e-02, -1.8603e-01,  1.1155e-01]]]],\n",
       "       requires_grad=True)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.conv1.weight"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "c0174199",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([6, 1, 5, 5])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.conv1.weight.shape "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "484a9739",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([12, 6, 5, 5])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    " mynet.conv2.weight.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "50a97b73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([120, 192])"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.fc1.weight.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "a7d3c040",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([60, 120])"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.fc2.weight.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "e0bb91b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([10, 60])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.out.weight.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "2b32375a",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[[-0.0421, -0.0469, -0.0743,  0.0487,  0.0513],\n",
       "         [ 0.0428, -0.0635,  0.0226,  0.0774,  0.0752],\n",
       "         [ 0.0719, -0.0256,  0.0734, -0.0681,  0.0111],\n",
       "         [-0.0051, -0.0478, -0.0634,  0.0448,  0.0332],\n",
       "         [-0.0401,  0.0381,  0.0050, -0.0451,  0.0711]],\n",
       "\n",
       "        [[-0.0515,  0.0649,  0.0449, -0.0659, -0.0605],\n",
       "         [ 0.0455, -0.0775,  0.0010, -0.0491, -0.0685],\n",
       "         [-0.0165,  0.0549, -0.0444, -0.0194,  0.0305],\n",
       "         [-0.0315,  0.0742,  0.0068,  0.0475,  0.0564],\n",
       "         [ 0.0365, -0.0766, -0.0350,  0.0236,  0.0066]],\n",
       "\n",
       "        [[-0.0357,  0.0191,  0.0813,  0.0597,  0.0133],\n",
       "         [-0.0130, -0.0575, -0.0665, -0.0761,  0.0162],\n",
       "         [ 0.0626, -0.0623,  0.0665,  0.0335, -0.0534],\n",
       "         [-0.0410, -0.0332, -0.0389, -0.0646,  0.0299],\n",
       "         [ 0.0764, -0.0263,  0.0653,  0.0426, -0.0061]],\n",
       "\n",
       "        [[ 0.0738, -0.0607, -0.0463, -0.0495, -0.0031],\n",
       "         [-0.0444,  0.0611,  0.0035, -0.0604, -0.0493],\n",
       "         [-0.0125, -0.0359,  0.0636, -0.0781,  0.0248],\n",
       "         [ 0.0616, -0.0772,  0.0423,  0.0220, -0.0363],\n",
       "         [-0.0425, -0.0717,  0.0014,  0.0656,  0.0249]],\n",
       "\n",
       "        [[-0.0769, -0.0435, -0.0251, -0.0528, -0.0366],\n",
       "         [ 0.0344,  0.0380, -0.0497, -0.0072, -0.0712],\n",
       "         [-0.0492, -0.0502,  0.0177, -0.0198,  0.0442],\n",
       "         [-0.0640, -0.0165,  0.0795, -0.0768,  0.0468],\n",
       "         [ 0.0160,  0.0232,  0.0028,  0.0260, -0.0207]],\n",
       "\n",
       "        [[ 0.0377, -0.0133, -0.0047,  0.0617, -0.0198],\n",
       "         [-0.0799, -0.0563,  0.0249,  0.0059, -0.0013],\n",
       "         [ 0.0809,  0.0370,  0.0248, -0.0272,  0.0256],\n",
       "         [-0.0338, -0.0565, -0.0511,  0.0771,  0.0609],\n",
       "         [-0.0173, -0.0032,  0.0727,  0.0520,  0.0034]]],\n",
       "       grad_fn=<SelectBackward>)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.conv2.weight[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ad58646d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([6, 5, 5])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.conv2.weight[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "7326e419",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([120, 192])"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mynet.fc1.weight.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "7edd4d42",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([6, 1, 5, 5])\n",
      "torch.Size([6])\n",
      "torch.Size([12, 6, 5, 5])\n",
      "torch.Size([12])\n",
      "torch.Size([120, 192])\n",
      "torch.Size([120])\n",
      "torch.Size([60, 120])\n",
      "torch.Size([60])\n",
      "torch.Size([10, 60])\n",
      "torch.Size([10])\n"
     ]
    }
   ],
   "source": [
    "#how to access all the parameters at once\n",
    "for param in mynet.parameters():\n",
    "    print(param.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "3a4f39f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "conv1.weight \t\t torch.Size([6, 1, 5, 5])\n",
      "conv1.bias \t\t torch.Size([6])\n",
      "conv2.weight \t\t torch.Size([12, 6, 5, 5])\n",
      "conv2.bias \t\t torch.Size([12])\n",
      "fc1.weight \t\t torch.Size([120, 192])\n",
      "fc1.bias \t\t torch.Size([120])\n",
      "fc2.weight \t\t torch.Size([60, 120])\n",
      "fc2.bias \t\t torch.Size([60])\n",
      "out.weight \t\t torch.Size([10, 60])\n",
      "out.bias \t\t torch.Size([10])\n"
     ]
    }
   ],
   "source": [
    "for name,param in mynet.named_parameters():\n",
    "    print(name, '\\t\\t' , param.shape)\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "03230e48",
   "metadata": {},
   "outputs": [],
   "source": [
    "in_features = torch.tensor([1,2,3,4],dtype = torch.float32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "60ce19f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "weight_matrix = torch.tensor([\n",
    "    [1,2,3,4],\n",
    "    [2,3,4,5],\n",
    "    [3,4,5,6]\n",
    "],dtype = torch.float32)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "ec5b68be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([30., 40., 50.])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "weight_matrix.matmul(in_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "5dde2818",
   "metadata": {},
   "outputs": [],
   "source": [
    "fc = nn.Linear(in_features = 4 , out_features=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "2a75b6ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "fc.weight = nn.Parameter(weight_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "a81ec06a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([29.7537, 39.6349, 50.0139], grad_fn=<AddBackward0>)"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fc(in_features)    #because of bias"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0435029",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
