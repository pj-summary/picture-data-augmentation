Image Data Augmentations



Requirements:
	torch
	tqdm
	argparse
	oc
	matplotlib



Use the followings to train:

python train.py --method baseline --epochs 200

python train.py --method baseline --data_augmentation --epochs 200

python train.py --method cutout --data_augmentation --epochs 200

python train.py --method mixup --data_augmentation --epochs 200

python train.py --method cutmix --data_augmentation --epochs 200
