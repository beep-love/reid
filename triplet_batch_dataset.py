import os
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

class VehicleTripletDataset(Dataset):
    def __init__(self, root_dir, list_file, info_file, mode = 'train', transform=None, P=8, K=4):
        self.root_dir = root_dir
        self.transform = transform
        self.P = P
        self.K = K
        self.mode = mode
        self.image_paths = []
        self.vehicle_ids = []
        self.camera_ids = []
        self.vehicle_info = {}  # Load the vehicle_info.txt file and parse vehicle type and color
        self.vehicle_camera_dict = defaultdict(lambda: defaultdict(list)) # Group images by vehicle ID and camera ID

        # Load the train_list.txt file and parse image paths, vehicle IDs, and camera IDs
        with open(os.path.join(root_dir, list_file), 'r') as file:
            for line in file:
                path, vehicle_id, camera_id = line.strip().split(' ')
                self.image_paths.append(path)
                self.vehicle_ids.append(int(vehicle_id))
                self.camera_ids.append(int(camera_id))

        # Load the vehicle_info.txt file and parse vehicle type and color
        with open(os.path.join(root_dir, info_file), 'r') as file:
            for line in file:
                path, _, _, vehicle_brand, vehicle_type, vehicle_color = line.strip().split(';')
                path = path.split('/')[0]       # Only use the first part of the path as all the vehicle in a path are same identity
                self.vehicle_info[path] = (vehicle_brand, vehicle_type, vehicle_color)

        # Group images by vehicle ID and camera ID

        for idx, (vehicle_id, camera_id) in enumerate(zip(self.vehicle_ids, self.camera_ids)):
            self.vehicle_camera_dict[vehicle_id][camera_id].append(idx)

        self.unique_vehicle_ids = list(set(self.vehicle_ids))
        self.shuffle_vehicle_ids()
    
    def shuffle_vehicle_ids(self):
        random.shuffle(self.unique_vehicle_ids)
        self.vehicle_id_batches = [self.unique_vehicle_ids[i:i + self.P] for i in range(0, len(self.unique_vehicle_ids), self.P)]
        
    def __len__(self):
        # Adjust the length to ensure it's divisible by P*K
        # return (len(self.image_paths) // (self.P * self.K)) * (self.P * self.K)
        num_batches = len(self.vehicle_id_batches)
        return num_batches * self.P * self.K

    def __getitem__(self, idx):
        
        # Initialize lists to store all anchors, positives, negatives, and labels
        all_anchors = []
        all_positives = []
        all_negatives = []
        all_labels = []

        # Get the batch index and vehicle IDs for this batch
        batch_index = idx // (self.P * self.K)
        vehicle_ids = self.vehicle_id_batches[batch_index]

        for vehicle_id in vehicle_ids:
            # Select a random camera ID for the current vehicle ID for anchor image
            camera_id = random.choice(list(self.vehicle_camera_dict[vehicle_id].keys()))

            # Get a list of camera IDs for the current vehicle ID, excluding the camera ID used for the anchor image
            positive_camera_ids = [cam_id for cam_id in self.vehicle_camera_dict[vehicle_id].keys() if cam_id != camera_id]

            # Sample K indices from the list of indices associated with the selected anchor image camera ID
            indices = random.sample(self.vehicle_camera_dict[vehicle_id][camera_id], self.K)

            for anchor_idx in indices:
                anchor_path = os.path.join(self.root_dir, 'images/images', self.image_paths[anchor_idx])
                anchor_image = Image.open(anchor_path)
                _, anchor_vehicle_type, anchor_vehicle_color = self.vehicle_info[self.image_paths[anchor_idx].split('/')[0]]  # vehicle_brand is not used 

                # Select a positive image (same vehicle, different camera)

                # Randomly select a camera ID from the list of positive camera IDs
                positive_camera_id = random.choice(positive_camera_ids)

                # Randomly select an index from the list of indices associated with the selected positive camera ID
                positive_idx = random.choice(self.vehicle_camera_dict[vehicle_id][positive_camera_id])

                # Load the positive image
                positive_path = os.path.join(self.root_dir, 'images/images', self.image_paths[positive_idx])
                positive_image = Image.open(positive_path)

                # Select a negative image (different vehicle, same camera)
                negative_vehicle_id = random.choice(list(set(self.vehicle_ids) - {vehicle_id}))

                # Negative image (different vehicle ID, same camera ID, same vehicle type, and color)
                negative_candidates = [
                    idx for idx in self.vehicle_camera_dict[negative_vehicle_id][camera_id]
                    if self.vehicle_info[self.image_paths[idx].split('/')[0]][1] == anchor_vehicle_type  # vehicle_type
                    and self.vehicle_info[self.image_paths[idx].split('/')[0]][2] == anchor_vehicle_color # vehicle_color
                ]
                if negative_candidates:
                    negative_idx = random.choice(negative_candidates)
                else:
                    # Fallback: choose any image with a different vehicle ID but the same vehicle color
                    negative_candidates = [
                        idx for idx in self.vehicle_camera_dict[negative_vehicle_id][camera_id]
                        if self.vehicle_info[self.image_paths[idx].split('/')[0]][2] == anchor_vehicle_color  # vehicle_color
                    ]
                    if negative_candidates:
                        negative_idx = random.choice(negative_candidates)
                    else:
                        # Further fallback: choose any image with a different vehicle ID
                        negative_idx = random.choice(self.vehicle_camera_dict[negative_vehicle_id][camera_id])
                negative_path = os.path.join(self.root_dir, 'images/images', self.image_paths[negative_idx])
                negative_image = Image.open(negative_path)

                # Apply transformations if any
                if self.transform:
                    anchor_image = self.transform(anchor_image)
                    positive_image = self.transform(positive_image)
                    negative_image = self.transform(negative_image)
                
                # Add to the overall lists
                all_anchors.append(anchor_image)
                all_positives.append(positive_image)
                all_negatives.append(negative_image)
                all_labels.append(vehicle_id)  # Add the vehicle ID as the label

        # Convert lists to tensors
        all_anchors = torch.stack(all_anchors)
        all_positives = torch.stack(all_positives)
        all_negatives = torch.stack(all_negatives)
        all_labels = torch.tensor(all_labels, dtype=torch.long)

        return all_labels, all_anchors, all_positives, all_negatives