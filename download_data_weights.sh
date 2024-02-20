#!/usr/bin/env bash

if ! [[ -d ./data ]]; then
	mkdir ./data
fi
cd ./data

if ! [[ -f ./seed_weights.csv ]]; then
	echo "Downloading 'seed_weights.csv'"
	curl https://zenodo.org/records/10009966/files/seed_weights.csv?download=1 --output seed_weights.csv
else
	echo "Seed weights file found."
fi

# Get image data
if ! [[ -d ./images ]]; then
    mkdir ./images
    cd ./images
	echo "Downloading 'train_val_images.zip'"
	curl https://zenodo.org/records/10009966/files/train_val_images.zip?download=1 --output train_val_images.zip
	echo "Extracting image directories"
	unzip train_val_images.zip
	cd ../
elif ! [[ -d ./images/train ]] | [[ -d ./images/val ]]; then
    cd ./images
	echo "Extracting image directories"
	unzip train_val_images.zip
	cd ../
else
    cd ./images
    unzip -U ./train_val_images.zip
	echo "Training and validation images found and extracted."
	cd ../
fi

# Get model checkpoints from Zenodo
cd ../
if ! [[ -d ./model_chkpt ]]; then
	mkdir ./model_chkpt
	cd ./model_chkpt
	echo "Downloading model weights"
	curl https://zenodo.org/records/8387983/files/frcnn_sainfoin_1.0_100.pth?download=1 --output frcnn_sainfoin_1.0_100.pth
	cd ../
elif ! [[ -f ./model_chkpt/frcnn_sainfoin_1.0_100.pth ]]; then
	cd ./model_chkpt
	echo "Downloading model weights"
	curl https://zenodo.org/records/8387983/files/frcnn_sainfoin_1.0_100.pth?download=1 --output frcnn_sainfoin_1.0_100.pth
	cd ../
else
	echo "Model weights found."
fi

