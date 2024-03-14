import os
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import multiprocessing
import logging
from tqdm import tqdm
import time
from memory_profiler import profile
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to precompute vehicle attributes for each vehicle ID
def precompute_vehicle_attributes(vehicle_info, image_paths, vehicle_ids):
    vehicle_attributes = {}
    for vehicle_id in vehicle_ids:
        vehicle_path = image_paths[vehicle_ids.index(vehicle_id)].split('/')[0]
        vtype, color = vehicle_info[vehicle_path][1], vehicle_info[vehicle_path][2]
        vehicle_attributes[vehicle_id] = (vtype, color)
    return vehicle_attributes

# # Function to find similar vehicles(different vehicle_id) in same camera optimized for less iterations

# @profile
def find_similar_vehicles_same_camera(args):
    vehicle_id, vehicle_attributes, vehicle_camera_dict = args
    similar_vehicles_same_camera = defaultdict(set)  # Use a set for faster membership check
    similar_vehicles_different_camera = set()  # Use a set for faster membership check
    vtype, color = vehicle_attributes[vehicle_id]
    current_vehicle_cameras = set(vehicle_camera_dict[vehicle_id].keys())

    for other_id in vehicle_attributes:
        if vehicle_id != other_id:
            other_vtype, other_color = vehicle_attributes[other_id]
            if vtype == other_vtype and color == other_color:
                other_vehicle_cameras = set(vehicle_camera_dict[other_id].keys())
                common_cameras = current_vehicle_cameras.intersection(other_vehicle_cameras)
                if common_cameras:
                    for camera_id in common_cameras:
                        similar_vehicles_same_camera[camera_id].add(other_id)
                else :
                    similar_vehicles_different_camera.add(other_id)

    return vehicle_id, {camera_id: list(vehicles) for camera_id, vehicles in similar_vehicles_same_camera.items()}, list(similar_vehicles_different_camera) 
                        # Convert sets to lists before returning                                                     # Convert back to a list before returning

# Not needed integrated above
# @profile
# def find_similar_vehicles_different_camera(args):
#     vehicle_id, vehicle_attributes, vehicle_camera_dict = args
#     similar_vehicles_different_camera = set()  # Use a set for faster membership check
#     vtype, color = vehicle_attributes[vehicle_id]
#     for other_id in vehicle_attributes:
#         if vehicle_id != other_id:
#             other_vtype, other_color = vehicle_attributes[other_id]
#             if vtype == other_vtype and color == other_color:
#                 similar_vehicles_different_camera.add(other_id)
#     return vehicle_id, list(similar_vehicles_different_camera)  


# def find_similar_vehicles_same_camera(args):
#     vehicle_id, vehicle_info, vehicle_camera_dict, image_paths, vehicle_ids = args
#     cameras = vehicle_camera_dict[vehicle_id]  # Get the dictionary of cameras for the vehicle_id
#     similar_vehicles_same_camera = defaultdict(list)
#     vehicle_path = image_paths[vehicle_ids.index(vehicle_id)].split('/')[0]
#     vtype, color = vehicle_info[vehicle_path][1], vehicle_info[vehicle_path][2]

#     # Get the set of camera IDs for the current vehicle
#     current_vehicle_cameras = set(cameras.keys())

#     for other_id, other_cameras in vehicle_camera_dict.items():
#         if vehicle_id != other_id:
#             # Get the set of camera IDs for the other vehicle
#             other_vehicle_cameras = set(other_cameras.keys())
#             # Find the common camera IDs between the current vehicle and the other vehicle
#             common_cameras = current_vehicle_cameras.intersection(other_vehicle_cameras)

#             for camera_id in common_cameras:
#                 other_vehicle_path = image_paths[vehicle_ids.index(other_id)].split('/')[0]
#                 if vehicle_info[other_vehicle_path][1] == vtype and vehicle_info[other_vehicle_path][2] == color:
#                     similar_vehicles_same_camera[camera_id].append(other_id)

#     return vehicle_id, dict(similar_vehicles_same_camera)



def defaultdict_list():
    return defaultdict(list)

class VehicleTripletDataset(Dataset):
    def __init__(self, root_dir, list_file, info_file, mode = 'train', transform=None, P=4, K=4):
        self.root_dir = root_dir
        self.transform = transform
        self.P = P
        self.K = K
        self.mode = mode
        self.image_paths = []
        self.vehicle_ids = []
        self.camera_ids = []
        self.vehicle_info = {}  # Load the vehicle_info.txt file and parse vehicle type and color
        
        self.vehicle_camera_dict = defaultdict(defaultdict_list)    # Group images by vehicle ID and camera ID

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

        num_unique_vehicle_camera_pairs = sum(len(cameras) for cameras in self.vehicle_camera_dict.values())
        print("LOADED INFO FILE")
        print("#################################################")
        print(f"Total number of images: {len(self.image_paths)}")
        print(f"Total number of vehicle IDs: {len(set(self.vehicle_ids))}")
        print(f"Total number of camera IDs: {len(set(self.camera_ids))}")
        print(f"Total number of vehicle types: {len(set([v[1] for v in self.vehicle_info.values()]))}")
        print(f"Total number of vehicle colors: {len(set([v[2] for v in self.vehicle_info.values()]))}")
        print(f"Total number of vehicle brands: {len(set([v[0] for v in self.vehicle_info.values()]))}")
        print(f"Total number of unique (vehicle ID, camera ID) pairs: {num_unique_vehicle_camera_pairs}")
        print("#################################################")
        self.unique_vehicle_ids = list(set(self.vehicle_ids))

        # Split the unique vehicle IDs into training and validation sets (80:20 ratio)
        num_training = int(len(self.unique_vehicle_ids) * 0.8)
        self.training_vehicle_ids = self.unique_vehicle_ids[:num_training]
        self.validation_vehicle_ids = self.unique_vehicle_ids[num_training:]

        # Set the current mode (training or validation)
        self.mode = mode

        # Shuffle the vehicle IDs based on the mode
        self.shuffle_vehicle_ids()
        print("SHUFFLED VEHICLE IDS")

        # Initialize the dictionary to store similar vehicles
        self.similar_vehicles_same_camera = defaultdict(dict)
        self.similar_vehicles_different_camera = defaultdict(list)

        print(" PRECOMPUTING VEHICLE ATTRIBUTES")
        start_time = time.time()
        # Precompute vehicle attributes
        vehicle_attributes = precompute_vehicle_attributes(self.vehicle_info, self.image_paths, self.vehicle_ids)

    # NEED TO TEST THIS PART TO Calculate the time taken to precompute vehicle attributes : 
    # Currently it takes 163 seconds in above implementation.
        
        # args_vehicle_attributes = [self.vehicle_info, self.image_paths, self.vehicle_ids]
        # with multiprocessing.Pool(processes=(multiprocessing.cpu_count()-2)) as pool:
        #     vehicle_attributes = pool.map(precompute_vehicle_attributes, args_vehicle_attributes)

        time_elapsed = time.time() - start_time
        print(f"Time taken to precompute vehicle attributes: {time_elapsed:.2f} seconds")

        print("PRECOMPUTING SIMILAR VEHICLE AVAILABLE IN SAME AND DIFFERENT CAMERA IDS")

        start_time = time.time()
        # Prepare arguments for multiprocessing
        args_list_same_cam = [(vehicle_id, vehicle_attributes, self.vehicle_camera_dict) for vehicle_id in self.vehicle_ids]

        # In your class initialization or method where you use multiprocessing
        with multiprocessing.Pool(processes=(multiprocessing.cpu_count()//2)) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(find_similar_vehicles_same_camera, args_list_same_cam, chunksize=100), total=len(args_list_same_cam)):
                results.append(result)

        # Update the dictionary with the results
        for vehicle_id, similar_vehicles_dict, similar_vehicles in results:
            self.similar_vehicles_same_camera[vehicle_id].update(similar_vehicles_dict)
            self.similar_vehicles_different_camera[vehicle_id] = similar_vehicles

        time_elapsed = time.time() - start_time
        print(f"Time taken to precompute similar vehicles in same and different camera: {time_elapsed:.2f} seconds")

        # print("PRECOMPUTING SIMILAR VEHICLE AVAILABLE IN DIFFERENT CAMERA IDS")
        # start_time = time.time()
        # args_list_different_cam = [(vehicle_id, vehicle_attributes, self.vehicle_camera_dict) for vehicle_id in self.vehicle_ids]

        # # args_list_different_cam = [(vehicle_id, self.vehicle_info, self.vehicle_camera_dict, self.image_paths, self.vehicle_ids) for vehicle_id in self.vehicle_ids]

        # # Use multiprocessing to find similar vehicles
        # with multiprocessing.Pool(processes=(multiprocessing.cpu_count()//2) ) as pool:
        #     results = []
        #     for result in tqdm(pool.imap_unordered(find_similar_vehicles_different_camera, args_list_different_cam, chunksize=100), total=len(args_list_same_cam)):
        #         results.append(result)
        #     # results = pool.map(find_similar_vehicles_different_camera, args_list_different_cam)

        # # Update the dictionary with the results
        # for vehicle_id, similar_vehicles in results:
        #     self.similar_vehicles_different_camera[vehicle_id] = similar_vehicles
        # time_elapsed = time.time() - start_time
        # print(f"Time taken to precompute similar vehicles in different camera: {time_elapsed:.2f} seconds")
        
        print("Initialized VehicleTripletDataset")

    
    def shuffle_vehicle_ids(self):
        if self.mode == 'train':
            random.shuffle(self.training_vehicle_ids)
            self.vehicle_id_batches = [self.training_vehicle_ids[i:i + self.P] for i in range(0, len(self.training_vehicle_ids), self.P)]
        elif self.mode == 'val':
            random.shuffle(self.validation_vehicle_ids)
            self.vehicle_id_batches = [self.validation_vehicle_ids[i:i + self.P] for i in range(0, len(self.validation_vehicle_ids), self.P)]

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
        if self.mode == 'train':
            vehicle_ids = self.vehicle_id_batches[batch_index]
        elif self.mode == 'val':
            vehicle_ids = self.vehicle_id_batches[batch_index]

        for vehicle_id in vehicle_ids:
            # Select a random camera ID for the current vehicle ID for anchor image
            camera_id = random.choice(list(self.vehicle_camera_dict[vehicle_id].keys()))

            # Get a list of camera IDs for the current vehicle ID, excluding the camera ID used for the anchor image
            positive_camera_ids = [cam_id for cam_id in self.vehicle_camera_dict[vehicle_id].keys() if cam_id != camera_id]

            # Get the list of indices for the current vehicle ID and camera ID
            available_indices = self.vehicle_camera_dict[vehicle_id][camera_id]
            
            # Sample K indices from the list of indices associated with the selected anchor image camera ID
            # Randomly sample indices with replacement to ensure self.K values
            indices = random.choices(available_indices, k=self.K)
            
            # Sample K indices from the list of indices associated with the selected anchor image camera ID
            # indices = random.sample(self.vehicle_camera_dict[vehicle_id][camera_id], self.K)

            for anchor_idx in indices:
                anchor_path = os.path.join(self.root_dir, 'images/images', self.image_paths[anchor_idx])
                anchor_image = Image.open(anchor_path)
                _, anchor_vehicle_type, anchor_vehicle_color = self.vehicle_info[self.image_paths[anchor_idx].split('/')[0]]  # vehicle_brand is not used 

                # Select a positive image (same vehicle, different camera)
                # Check if there are any positive camera IDs available
                if positive_camera_ids:
                    # Randomly select a camera ID from the list of positive camera IDs
                    positive_camera_id = random.choice(positive_camera_ids)

                    # Randomly select an index from the list of indices associated with the selected positive camera ID
                    positive_idx = random.choice(self.vehicle_camera_dict[vehicle_id][positive_camera_id])
                else:
                    # Fallback: If no other camera IDs are available, use the same camera ID as the anchor image
                    # This is not ideal but provides a fallback to avoid the IndexError
                    positive_camera_id = camera_id
                    # Exclude the anchor index from the choices
                    positive_indices = [idx for idx in self.vehicle_camera_dict[vehicle_id][positive_camera_id] 
                                        if idx != anchor_idx
                                        ]
                    # Randomly select a positive index (if available)
                    if positive_indices:
                        positive_idx = random.choice(positive_indices)  
                    else:
                        logging.warning(f"No positive candidates found for vehicle ID {vehicle_id} with camera ID {camera_id}. Using anchor index as fallback.")
                        positive_idx = anchor_idx  # Use anchor index as a last resort

                # Load the positive image
                positive_path = os.path.join(self.root_dir, 'images/images', self.image_paths[positive_idx])
                positive_image = Image.open(positive_path)

    ## OLD IMPLEMENTATION to find negative image

                # Select a negative vehicle ID that is different from the anchor vehicle ID
                # negative_vehicle_id = random.choice(list(set(self.vehicle_ids) - {vehicle_id}))

                # # Choose a camera ID for the negative sample (same as anchor camera ID)
                # camera_id = anchor_camera_id

                # Negative image (different vehicle ID, same camera ID, same vehicle type, and color)
                # negative_candidates = [
                #     idx for idx in self.vehicle_camera_dict[negative_vehicle_id].get(camera_id, [])
                #     if self.vehicle_info[self.image_paths[idx].split('/')[0]][1] == anchor_vehicle_type  # vehicle_type
                #     and self.vehicle_info[self.image_paths[idx].split('/')[0]][2] == anchor_vehicle_color  # vehicle_color
                # ]
                # if negative_candidates:
                #     negative_idx = random.choice(negative_candidates)
                # else:
                #     # Fallback: choose any image with a different vehicle ID, same camera id and the same vehicle color
                #     negative_candidates = [
                #         idx for idx in self.vehicle_camera_dict[negative_vehicle_id].get(camera_id, [])
                #         if self.vehicle_info[self.image_paths[idx].split('/')[0]][2] == anchor_vehicle_color  # vehicle_color
                #     ]
                #     if negative_candidates:
                #         negative_idx = random.choice(negative_candidates)
                #     else:
                #         # Further fallback: choose any image with a different vehicle ID in same camera ID
                #         if self.vehicle_camera_dict[negative_vehicle_id].get(camera_id):
                #             negative_idx = random.choice(self.vehicle_camera_dict[negative_vehicle_id][camera_id])
                #         else:
                #             # Try to find an image with the same vehicle color and type from any camera
                #             negative_candidates = [
                #                 idx for idx in self.vehicle_camera_dict[negative_vehicle_id]
                #                 if self.vehicle_info[self.image_paths[idx].split('/')[0]][1] == anchor_vehicle_type  # vehicle_type
                #                 and self.vehicle_info[self.image_paths[idx].split('/')[0]][2] == anchor_vehicle_color  # vehicle_color
                #             ]
                #             if negative_candidates:
                #                 negative_idx = random.choice(negative_candidates)
                #             else:
                #                 # If not found, try to find an image with the same vehicle color from any camera
                #                 negative_candidates = [
                #                     idx for idx in self.vehicle_camera_dict[negative_vehicle_id]
                #                     if self.vehicle_info[self.image_paths[idx].split('/')[0]][2] == anchor_vehicle_color  # vehicle_color
                #                 ]
                #                 if negative_candidates:
                #                     negative_idx = random.choice(negative_candidates)
                #                 else:
                #                     # If still not found, choose any image from any camera as long as the vehicle is different
                #                     negative_idx = random.choice([idx for idx in range(len(self.vehicle_ids)) if self.vehicle_ids[idx] == negative_vehicle_id])
                
    # New implementation ensuring we create a dictionary of similar vehicles and sample from it
    # Negative image (different vehicle ID, same camera ID, same vehicle type, and color)

                if camera_id in self.similar_vehicles_same_camera[vehicle_id]:
                    negative_vehicle_id = random.choice(self.similar_vehicles_same_camera[vehicle_id][camera_id])
                    negative_candidates = self.vehicle_camera_dict[negative_vehicle_id][camera_id]
                else:
                    # Fallback: choose any image with a different vehicle ID, and the same vehicle color and type
                    if self.similar_vehicles_different_camera[vehicle_id]:
                        negative_vehicle_id = random.choice(self.similar_vehicles_different_camera[vehicle_id])
                        negative_candidates = [idx for cam_id in self.vehicle_camera_dict[negative_vehicle_id] for idx in self.vehicle_camera_dict[negative_vehicle_id][cam_id]]
                    else:
                        # Fallback: choose any image with a different vehicle ID
                        negative_vehicle_id = random.choice(list(set(self.vehicle_ids) - {vehicle_id}))
                        negative_candidates = [idx for cam_id in self.vehicle_camera_dict[negative_vehicle_id] for idx in self.vehicle_camera_dict[negative_vehicle_id][cam_id]]
                
                negative_idx = random.choice(negative_candidates)
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