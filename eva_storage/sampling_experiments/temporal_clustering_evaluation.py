from loaders.uadetrac_loader import UADetracLoader
from loaders.jackson_loader import JacksonLoader
from eva_storage.sampling_experiments.sampling_utils import create_dummy_boxes, evaluate_with_gt
from eva_storage.temporalClusterModule import TemporalClusterModule
from others.amdegroot.data.jackson import JACKSON_CLASSES
from others.amdegroot.eval_uad2 import * ## we import all the functions from here and perform our own evaluation




if __name__ == "__main__":
    total_eval_num = 300

    loader = JacksonLoader()
    images = loader.load_images(image_size=300)

    ## we want to filter out only the ones that we want to use

    labels = loader.load_labels(relevant_classes=JACKSON_CLASSES)

    images, labels = loader.filter_input(images, labels)
    boxes = create_dummy_boxes(labels)

    images_downscaled = images

    st = time.perf_counter()
    ## we need to downscale to 10x10
    wanted_width = 10
    wanted_height = 10
    width_skip_rate = images.shape[1] // wanted_width
    height_skip_rate = images.shape[2] // wanted_height
    images_downscaled = images_downscaled[:, ::width_skip_rate, ::height_skip_rate]
    images_downscaled = np.mean(images_downscaled, axis=3)
    print(images_downscaled.shape)
    images_reshaped = images_downscaled.reshape(len(images), 100)

    cluster_count = total_eval_num
    number_of_neighbors = 3

    temporal_cluster = TemporalClusterModule()
    _, rep_indices, all_cluster_labels = temporal_cluster.run(images_reshaped, number_of_clusters=cluster_count,
                                                              number_of_neighbors=number_of_neighbors)
    ## we need to get rep labels, rep_boxes as well
    rep_images = images[rep_indices]
    rep_labels = np.array(labels)[rep_indices]
    rep_boxes = np.array(boxes)[rep_indices]

    mapping = temporal_cluster.get_mapping(rep_indices, all_cluster_labels)
    mapping = mapping.astype(np.int)

    print(f"finished whole process in {time.perf_counter() - st} (secs)")

    evaluate_with_gt(images, labels, boxes, rep_images, rep_labels, rep_boxes, mapping, JACKSON_CLASSES)


