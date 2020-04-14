import cv2
import sys
import os
import numpy as np
import torch
import torchvision as tv

##### UA-detrac loading functions ######
class UADataset_lite:
    def __init__(self, transform=None, target_transform=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.class_names = ['car', 'bus', 'others', 'van']
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.transform = transform
        self.target_transform = target_transform

        
    def set_x(self, images):
        self.X_train = images
    
    
    def set_y(self, labels):
        labels = self.convert_labels(labels)
        self.y_train = labels
    
    def get_image(self, index):
        return self.X_train[index]
    
    def set_y_boxes(self, boxes):
        boxes = self.convert_boxes(boxes)
        self.y_train_boxes = boxes
     
        
    def convert_labels(self, labels):
        new_labels = []
        for frame in labels:
            new_frame = []
            for label in frame:
                index = self.class_names.index(label)
                if index < 0 or index >= len(self.class_names):
                    print("label is wrong!")
                    print("  expected ", self.class_names)
                    print("  given", label)
                    assert(False)
                new_frame.append(index)
            new_labels.append(new_frame)
        assert(len(new_labels) == len(labels))
        return new_labels
        
        
    def convert_boxes(self, boxes_dataset):
        new_boxes = []
        for frame in boxes_dataset:
            new_frame = []
            for box in frame:
                top, left, bottom, right = box
                new_frame.append([left, top, right, bottom])
            new_boxes.append(new_frame)
        assert(len(new_boxes) == len(boxes_dataset))
        return new_boxes
        
    
    def __getitem__(self, index):
        image = self.X_train[index]
        boxes = self.y_train_boxes[index]
        labels = self.y_train[index]     
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int)
        ## image, boxes, labels
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
            
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
          
        return image, boxes, labels

    
    def __len__(self):
        return len(self.X_train)


class UADataset:
    def __init__(self):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        
        self.root = '/home/jbang36/eva'
        
        self.class_names = ['car', 'bus', 'others', 'van']
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        
        X_train_norm, X_test_norm, Y_train_dict, Y_test_dict = get_uadetrac()
        anno_dir = os.path.join(self.root,'data', 'ua_detrac','small-annotations')
        boxes_dataset = get_boxes(anno_dir, width = 300, height = 300)
        print("finished loading the datset")
        boxes_dataset = self.convert_boxes(boxes_dataset)
        print("finished converting the boxes")
        y_train = Y_train_dict['vehicle_type']
        y_test = Y_test_dict['vehicle_type']
        y_train = self.convert_labels(y_train)
        y_test = self.convert_labels(y_test)
        print("finished converting the labels")
        
        self.X_train = X_train_norm
        self.X_test = X_test_norm
        self.y_train = y_train
        self.y_test = y_test
        division = len(self.X_train)
        self.y_train_boxes = boxes_dataset[:division]
        self.y_test_boxes = boxes_dataset[division:]
        self.mode = 'train' ## mode can either be train or test
        
        assert(len(self.X_train) == len(self.y_train))
        assert(len(self.X_test) == len(self.y_test))
        assert(len(self.X_train) == len(self.y_train_boxes))
        assert(len(self.X_test) == len(self.y_test_boxes))
        
    def convert_labels(self, labels):
        new_labels = []
        for frame in labels:
            new_frame = []
            for label in frame:
                index = self.class_names.index(label)
                if index < 0 or index >= len(self.class_names):
                    print("label is wrong!")
                    print("  expected ", self.class_names)
                    print("  given", label)
                    assert(False)
                    
                new_frame.append(index)
            new_labels.append(new_frame)
        assert(len(new_labels) == len(labels))
        return new_labels
        
    def convert_boxes(self, boxes_dataset):
        new_boxes = []
        for frame in boxes_dataset:
            new_frame = []
            for box in frame:
                top, left, bottom, right = box
                new_frame.append([left, top, right, bottom])
            new_boxes.append(new_frame)
        assert(len(new_boxes) == len(boxes_dataset))
        return new_boxes
        
    def mode2train(self):
        self.mode = 'train'
    
    def mode2test(self):
        self.mode = 'test'
        
    def __getitem__(self, index):
        if self.mode == 'train':
            image = self.X_train[index]
            boxes = self.y_train_boxes[index]
            labels = self.y_train[index]
        else:
            image = self.X_test[index]
            boxes = self.y_test_boxes[index]
            labels = self.y_test[index]
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int)
        
        return image, boxes, labels

    def __len__(self):
        if self.mode == 'train':
            return len(self.X_train)
        else:
            return len(self.X_test)
        


   


def get_boxes(anno_dir, width = 300, height = 300):
    import xml.etree.ElementTree as ET
    original_height = 540
    original_width = 960
    anno_files = os.listdir(anno_dir)
    anno_files.sort()
    boxes_dataset = []
    cumu_count = 0    
    
    for anno_file in anno_files:
        if ".xml" not in anno_file:
            print("skipping", anno_file)
            continue
        file_path = os.path.join(anno_dir, anno_file)

        tree = ET.parse(file_path)
        tree_root = tree.getroot()

        for frame in tree_root.iter('frame'):
            boxes_frame = []
            curr_frame_num = int(frame.attrib['num'])
            if len(boxes_dataset) < cumu_count + curr_frame_num - 1:
                boxes_dataset.extend( [None] * (cumu_count + curr_frame_num - len(boxes_dataset)) )
            for box in frame.iter('box'):
                left = int( float(box.attrib['left']) * width / original_width )
                top = int( float(box.attrib['top']) * height / original_height )
                right = int( (float(box.attrib['left']) + float(box.attrib['width'])) * width / original_width )
                bottom = int( (float(box.attrib['top']) + float(box.attrib['height'])) * height / original_height )
                
                
                boxes_frame.append( (top, left, bottom, right))
            
            boxes_dataset.append(boxes_frame)
    
    return boxes_dataset
        

def get_uadetrac():
    
    home_dir = '/home/jbang36/eva'
    data_dir = os.path.join(home_dir, 'data', 'ua_detrac','DETRAC-Images')
    filter_path = os.path.join(home_dir, 'filters')
    loader_path = os.path.join(home_dir, 'loaders')

    sys.path.append(home_dir)
    sys.path.append(loader_path)
    sys.path.append(filter_path)
        
    from loaders.load import Load

    load = Load()

    eva_dir = home_dir
    image_dir = os.path.join(eva_dir, "data", "ua_detrac", "small-data")
    anno_dir = os.path.join(eva_dir, "data", "ua_detrac", "small-annotations")

    X, length_per_mvi = load.load_images(image_dir, grayscale = False)
    Y_dict = load.load_XML(anno_dir, X, length_per_mvi)

    def _split_train_test(X,Y_dict):
        n_samples, _, _, _= X.shape
        print(n_samples)
        train_index_end = int(len(X) * 0.8)
        print(train_index_end)

        X_train = X[:train_index_end]
        X_test = X[train_index_end:]

        Y_dict_train = {}
        Y_dict_test = {}
        for column in Y_dict:
            Y_dict_train[column] = Y_dict[column][:train_index_end]
            Y_dict_test[column] = Y_dict[column][train_index_end:]

        return X_train, X_test, Y_dict_train, Y_dict_test

    X_train, X_test, Y_train_dict, Y_test_dict = _split_train_test(X, Y_dict)
    X_train_norm = format_image_fixed(X_train, 300, 300)
    X_test_norm = format_image_fixed(X_test, 300, 300)
    return X_train_norm, X_test_norm, Y_train_dict, Y_test_dict
    
   
    
def format_image_fixed(X, height, width):
    n_samples,_,_,channels = X.shape
    X_new = np.ndarray(shape=(n_samples, height, width, channels))
    for i in range(n_samples):
        X_new[i] = cv2.resize(X[i],  (width,height))
    
    X_new /= 255.0
        
    return X_new


##### end of UA-detrac functions #########



def remove_checkpoints():
    image_dir = os.path.join(eva_dir, "data", "ua_detrac", "small-data")
    mvi_folders = os.listdir(image_dir)
    mvi_folders.sort()
    height = 40
    width = 80
    channels = 1

    for i, mvi_folder_name in enumerate(mvi_folders):
        file_names = os.listdir(os.path.join(image_dir, mvi_folder_name))
        for file_name in file_names:
            if 'check' in file_name:
                print(mvi_folder_name)
                print("  " + file_name)
    return
                

def write_seg_images():
    
    image_dir = os.path.join(eva_dir, "data", "ua_detrac", "small-data")
    mvi_folders = os.listdir(image_dir)
    mvi_folders.sort()
    height = 40
    width = 80
    channels = 1
    curr_count = 0

    for i, mvi_folder_name in enumerate(mvi_folders):
        print("folder name:", mvi_folder_name)
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        #load the content of that folder
        file_names = os.listdir(os.path.join(image_dir, mvi_folder_name))
        arr = np.ndarray((len(file_names), height, width), dtype = np.uint8)
        print("  number of files:", len(file_names))
        file_names.sort()
        #fig = plt.figure(figsize=(40,80))
        if os.path.exists('mog_images/'+mvi_folder_name) == False: 
            os.makedirs('mog_images/'+mvi_folder_name)


        for j,file_name in enumerate(file_names):
            full_file_name = os.path.join(image_dir, mvi_folder_name, file_name)
            img = cv2.imread(full_file_name, 0)

            arr[i] = cv2.resize(img, (width, height))
            #print(arr[i].shape)
            fgmask = fgbg.apply(arr[i])
            #fig.add_subplot(1,2,1)
            #plt.imshow(arr[i], cmap = 'gray')
            #fig.add_subplot(1,2,2)
            #plt.imshow(fgmask, cmap = 'gray')
            #plt.show()
            write_name = os.path.join('mog_images', mvi_folder_name, 'image{:06d}.jpg'.format(curr_count))
            cv2.imwrite(write_name, fgmask)
            curr_count += 1

        print("    done!")
        
    return

def convert(recon):
    recon_p = recon.permute(0,2,3,1)
    #print(recon_p.size())
    recon_imgs = recon_p.detach().cpu().numpy()
    recon_imgs *= 255
    recon_imgs = recon_imgs.astype(np.uint8)
    return recon_imgs

def convert_one(recon):
    recon_p = recon.permute(0,2,3,1)
    #print(recon_p.size())
    recon_imgs = recon_p.detach().cpu().numpy()
    recon_imgs *= 255
    recon_imgs = recon_imgs.astype(np.uint8)
    recon_imgs = recon_imgs.squeeze()
    return recon_imgs

def convert_soft(seged):
    K = 2


    N,K,H,W = seged.size()
    arged = torch.argmax(seged, dim = 1) #channels is 1
    arged = arged * 255 / (K - 1)
    arged_imgs = arged.detach().cpu().numpy()


    arged_imgs = arged_imgs.astype(np.uint8)
    assert(arged_imgs.shape == (N,H,W))
    return arged_imgs

#### Computer Vision Utils #####

# seg_img: one gray-scale image that is segmented
# @return: will return all individual steps in a list
def post_individual_v2(seg_img):
    # Notes:
    # Upgrade from v1 (hopefully?). We dilute the patches a lot more before doing canny detection
    # NOT GOOD
    # Move back to v1 but don't perform dilution..
    # Okay I like this....
    
    """
    1. cv2.GaussianBlur()
    1. cv2.medianBlur()
    2. cv2.threshold(img, low, max, cv2.THRESH_OTSU)
    3. cv2.threshold(img, low, max, cv2.BINARY)
    4. cv2.erode(img, kernel, iterations)
    5. cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations )
    6. cv2.dilate(img, kernel, iterations)
    """
    seg_cp = np.copy(seg_img)
    #gaus = cv2.GaussianBlur(seg_cp, (5,5), 0)
    med = cv2.medianBlur(seg_cp, 5)
    
    #ret, otsu_g = cv2.threshold(gaus, 0, 255, cv2.THRESH_OTSU)
    ret, otsu_m = cv2.threshold(med, 0, 255, cv2.THRESH_OTSU)
    
    #overlapped image with after
    
    
    #canny detection image after overlapped
    
    labels = ['median blur', 'ostu on median']
    return (labels, [med, otsu_m])
    


# seg_img: one gray-scale image that is segmented
# @return: will return all individual steps in a list
def post_individual(seg_img):
    # There are multiple operations to choose from
    """
    1. cv2.GaussianBlur()
    1. cv2.medianBlur()
    2. cv2.threshold(img, low, max, cv2.THRESH_OTSU)
    3. cv2.threshold(img, low, max, cv2.BINARY)
    4. cv2.erode(img, kernel, iterations)
    5. cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations )
    6. cv2.dilate(img, kernel, iterations)
    """
    seg_cp = np.copy(seg_img)
    #gaus = cv2.GaussianBlur(seg_cp, (5,5), 0)
    med = cv2.medianBlur(seg_cp, 5)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(med, cv2.MORPH_OPEN, kernel, iterations = 3)
    dilate = cv2.dilate(opening, kernel, iterations = 3)
    
    #ret, otsu_g = cv2.threshold(gaus, 0, 255, cv2.THRESH_OTSU)
    ret, otsu_m = cv2.threshold(med, 0, 255, cv2.THRESH_OTSU)
    ret, after = cv2.threshold(dilate, 0, 255, cv2.THRESH_OTSU)
    
    #overlapped image with after
    
    
    #canny detection image after overlapped
    
    labels = ['median blur', 'opening', 'dilation', 'ostu on median', 'otsu after dilation']
    return (labels, [med, opening, dilate, otsu_m, after])
    
    
    

# post_processing for patches of image size 300,300
# seg_matrix: gray-scale image that has a segmentation mapped (normally output from the network)
# @return: final result will be returned
def post_process(seg_matrix):
    ## Assume seg_matrix is of shape (n_samples, height, width)
    seg_matrix = seg_matrix.astype(np.uint8)
    seg_post = np.copy(seg_matrix)
    n_samples, height, width = seg_matrix.shape
    
    # We need to reshape the image 2 times because of dilation and erosion processes
    
    for i in range(n_samples):
        curr = np.copy(seg_matrix[i])
        curr = cv2.resize(curr, (width_tmp, height_tmp))
        #1. threshold the image
        ret, thresh = cv2.threshold(curr,0,255,cv2.THRESH_OTSU)
        #2. erode the image
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = 3)
        #3. dilate the image
        sure_bg = cv2.dilate(opening,kernel,iterations=5)
        #4. resize back the postprocessed image
        seg_post[i] = cv2.resize(sure_bg, (width, height))
        
    return seg_post
    
#test_images: np_matrix (n_samples, height, width, channels) - channels should be 3
#seg_images: segmented images to perform overlapping with (n_samples, height, width)
#@return: overlapped - np_matrix (n_samples, height, width, channels)
def overlap1(test_image, seg_image):
    # we will generate overlapped images for entire matrix
    assert(test_image.shape[0] == seg_image.shape[0])
    assert(test_image.shape[1] == seg_image.shape[1])
    test_image = test_image.astype(np.uint8)
    seg_image = seg_image.astype(np.uint8)

    overlapped = np.ndarray(shape = test_image.shape)

    overlapped = cv2.bitwise_or(test_image, test_image, mask = seg_image)
    overlapped = overlapped.astype(np.uint8)
    return overlapped    

    
#test_images: np_matrix (n_samples, height, width, channels) - channels should be 3
#seg_images: segmented images to perform overlapping with (n_samples, height, width)
#@return: overlapped - np_matrix (n_samples, height, width, channels)
def overlap(test_images, seg_images):
    # we will generate overlapped images for entire matrix
    assert(test_images.shape[0] == seg_images.shape[0])
    assert(test_images.shape[1] == seg_images.shape[1])
    assert(test_images.shape[2] == seg_images.shape[2])
    test_images = test_images.astype(np.uint8)
    seg_images = seg_images.astype(np.uint8)

    overlapped = np.ndarray(shape = test_images.shape)
    n_sample = test_images.shape[0]
    for i in range(n_sample):
        overlapped[i] = cv2.bitwise_or(test_images[i], test_images[i], mask = seg_images[i])
    overlapped = overlapped.astype(np.uint8)
    return overlapped
##### End of CV Utils ######



##### Patch related functions ######

# this function serves to reorder the patches derived from cv
# we will change to normal convention (ml convention)
# patch : (top, left, bottom, right)
def cv2ml_patches(cv_patches):
    if cv_patches == None:
        return None
    ml_patches = []
    for patch in cv_patches:
        left,top,width,height = patch
        ml_patch = (top, left, top + height, left + width)
        ml_patches.append(ml_patch)
    return ml_patches

def ml2cv_patches(ml_patches):
    #cv convention is (left, top, width, height)
    if ml_patches == None:
        return None
    cv_patches = []
    for patch in ml_patches:
        top, left, bottom, right = patch
        cv_patch = (left, top, right - left, bottom - top)
        cv_patches.append(cv_patch)
    return cv_patches


# patches: list of patch (starting col, starting row, width, height) ###UGHHH CVVVVV WHYYYYY (width and height are flipped)
# img_height: height of image
# img_width: width of image
# min_ratio_image: ratio of patch to image (both width and height need to satisfy this constraint)
# max_ratio_image: ratio of patch to image (both width and height need to satisfy this constraint)
# min_ratio_patch: ratio of patch_height to patch_width
# max_ratio_patch: ratio of patch_height to patch_width
def filter_patches(patches, img_height=300, img_width=300, 
                   min_ratio_image=0.05, max_ratio_image=0.7, 
                   min_ratio_patch=0.5, max_ratio_patch=3.0):
    # we want to filter all the patches and return new patches that satisfy the constraint
    new_patches = []
    
    if patches == None:
        return None
    for patch in patches:
        
        left = patch[0]
        top = patch[1]
        right = patch[0] + patch[2]
        bottom = patch[1] + patch[3]
        patch_height = patch[3]
        patch_width = patch[2]
        height_ratio = patch_height / img_height
        width_ratio = patch_width / img_width
        ratio_patch = patch_height / patch_width
        if height_ratio >= min_ratio_image and height_ratio <= max_ratio_image and \
            width_ratio >= min_ratio_image and width_ratio <= max_ratio_image and \
            ratio_patch >= min_ratio_patch and ratio_patch <= max_ratio_patch:
            new_patches.append(patch)
    return new_patches

#img: grayscale image that is used for detecting patches    
def detect_patches_final(img, shape='rectangle', mode = 2,
                         img_height=300, img_width=300,
                         min_ratio_image=0.05, max_ratio_image=0.7,
                         min_ratio_patch=0.5, max_ratio_patch=3.0):
    bounding_boxes = detect_patches(img, shape=shape, mode=mode)
    return filter_patches(bounding_boxes)

    
    
    

#img that will be used for contour detection
#default shape is rectangle
def detect_patches(img, shape='rectangle', mode=2):
    if mode == 1:
        contours, aaaa = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    elif mode == 2:
        contours, aaaa = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours in image")
        return None
    
    if shape != 'rectangle':
        print("Method not support")
        return None
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c,3,True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        
    # boundRect[i][0] - left, boundRect[i][1] - top, boundRect[i][2] - width, boundRect[i][3] - height
    return boundRect

def draw_patch(img, patches):
    #patch is [left, top, right, bottom]
    new_img = np.copy(img)
    color = (0, 0, 255)
    if patches != None:
        i = 0
        cv2.rectangle(new_img, (int(patches[i][0]), int(patches[i][1])), \
                  (int(patches[i][0]+patches[i][2]), int(patches[i][1]+patches[i][3])), color, 2)

    return new_img


#img: img to draw the patches on
#patches: list of rectangle points to draw
def draw_patches(img, patches, format = 'cv'):
    new_img = np.copy(img)
    color = (0, 0, 255)
    if format == 'cv':
        if patches != None:
            for i in range(len(patches)):
                cv2.rectangle(new_img, (int(patches[i][0]), int(patches[i][1])), \
                      (int(patches[i][0]+patches[i][2]), int(patches[i][1]+patches[i][3])), color, 2)

    if format == 'ml':
        if patches != None:
            for i in range(len(patches)):
                cv2.rectangle(new_img, (int(patches[i][1]), int(patches[i][0])), \
                                        (int(patches[i][3]), int(patches[i][2])), color, 2)
                
    return new_img


#img: img to use to extract patches (assume it is rgb image)
#patches: list of rectangle points
#patch_width: expected output patch_width
#patch_height: expected output patch_height
#@return None if patches is None
def extract_patches(img, patches, patch_width = 32, patch_height = 32):
    channels = 3 # we will assume
    if patches != None:
        patch_batch = np.ndarray(shape = (len(patches), patch_width, patch_height, channels))
        for i in range(len(patches)):
            left = patches[i][0]
            right = patches[i][0] + patches[i][2]
            top = patches[i][1]
            bottom = patches[i][1] + patches[i][3]
            patch = img[top:bottom,left:right,:]
            patch_batch[i] = cv2.resize(patch, (patch_width, patch_height))
        return patch_batch
    else:
        return None


# original_img: color img
# img: seg_img
# num: where to draw

######## End of patch related functions #########
    
    
    
######## Evaluation related functions #########
def filter_ground_truth(ground_truth_list, input_patch_type = 'ml', output_patch_type = 'ml'):
    new_ground_truth_list = []
    if input_patch_type == 'ml':
        tmp = []
        for i in range(len(ground_truth_list)):
            tmp.append(ml2cv_patches(ground_truth_list[i]))
        ground_truth_list = tmp
    
    ## filter patches takes in cv type patches so make sure to convert beforehand!
    for patches_frame in ground_truth_list:
        new_ground_truth_list.append( filter_patches(patches_frame) )
    assert( len(new_ground_truth_list) == len(ground_truth_list) )
    
    if output_patch_type == 'ml':
        tmp = []
        for i in range(len(new_ground_truth_list)):
            tmp.append(cv2ml_patches(new_ground_truth_list[i]))
        new_ground_truth_list = tmp
    
    return new_ground_truth_list



def adapt_detrac_boxes(todo):
    pass



def compute_overlap(ground_truth_frame, proposed_list_frame, iou = 0.5):
    # first work the naive version -> do the multithreaded version
    tp_count = 0
    fn_count = 0
    fp_count = 0
    if ground_truth_frame == None and proposed_list_frame == None:
        tp_count = 0
        fn_count = 0
        fp_count = 0
    elif ground_truth_frame == None:
        # there should not be boxes but model detects boxes
        # these are false positive
        fp_count = len(proposed_list_frame)
        
    elif proposed_list_frame == None:
        # there should be boxes but the model detects none
        # these are false negatives
        fn_count = len(ground_truth_frame)
    else:
        fp_list = [True] * len(proposed_list_frame)
        seen_proposed_boxes = [False] * len(proposed_list_frame)
        for ground_box in ground_truth_frame:
            matching_box = False
            for ii, proposed_box in enumerate(proposed_list_frame):
                if seen_proposed_boxes[ii]:
                    # if we have already seen this box and computed into the results, 
                    # don't recompute this into calculation
                    continue

                overlap_area = 0
                p_top, p_left, p_bottom, p_right = proposed_box
                g_top, g_left, g_bottom, g_right = ground_box

                #determine the actual area of overlap
                #for overlap ground_box needs to intersect proposed_box in at least two ways (vertical and horizontal)
                #
                i_top = max(p_top, g_top)
                i_left = max(p_left, g_left)
                i_bottom = min(p_bottom, g_bottom)
                i_right = min(p_right, g_right)
                # check if left is really left, top is really top
                if i_left < i_right and i_top < i_bottom: #else there is no overlap
                    overlap_area = (i_right - i_left) * (i_bottom - i_top)

                p_area = (p_right - p_left) * (p_bottom - p_top)
                g_area = (g_right - g_left) * (g_bottom - g_top)
                if overlap_area / (p_area + g_area - overlap_area) >= iou:
                    tp_count += 1
                    matching_box = True
                    fp_list[ii] = False
                    seen_proposed_boxes[ii] = True
            if matching_box == False:
                fn_count += 1
        fp_count = sum(fp_list)
    return tp_count, fp_count, fn_count
                
            


# ground_truth_list: list of ground truth boxes where each element refers to all the boxes existing in the frame
# proposed_list: list of proposed boxes where each element refers to all the boxes existing in the frame
# filter: whether to give some filtering arguments such as minimum size / aspect ratio
# iou: minimum area of overlap needed to be considered a true positive
def corloc(ground_truth_list, proposed_list, filter = False, iou = 0.5):
    total_count = 0
    tp_count = 0
    fn_count = 0
    fp_count = 0
    if filter:
        ground_truth_list = filter_ground_truth(ground_truth_list)
    
    # number of frames involved should be the same
    assert(len(ground_truth_list) == len(proposed_list))
    
    for i in range(len(ground_truth_list)):
        tp, fp, fn = compute_overlap(ground_truth_list[i], proposed_list[i], iou)
        tp_count += tp
        fn_count += fn
        fp_count += fp
    
    #compute precision and recall
    if tp_count + fp_count == 0:
        precision = 0
    else:
        precision = tp_count / (tp_count + fp_count)
    
    if tp_count + fn_count == 0:
        recall = 0
    else:
        recall = tp_count / (tp_count + fn_count)
    
    ## debugging
    print("true positive", tp_count)
    print("false positive", fp_count)
    print("false negative", fn_count)
    
    return precision, recall
    
    
    
    