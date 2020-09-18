"""
The preprocessor encapsulates the important i frame selection process + meta data generation / transfer processes
Interface should be run(images)
@Jaeho Bang
"""


import numpy as np
import os
import time


from eva_storage.temporalClusterModule import TemporalClusterModule
from eva_storage.featureExtractionMethods import DownSampleMeanMethod
from eva_storage.samplingMethods import FastMiddleEncounterMethod, MiddleEncounterMethod


class Timer:

    def __init__(self):
        self.st = -1
        self.nt = -1


    def tic(self):
        self.st = time.perf_counter()

    def toc(self, **kwargs):
        self.nt = time.perf_counter()
        self.print(**kwargs)

    def print(self, **kwargs):
        print(f"time taken is {self.nt - self.st} (seconds)")
        if kwargs:
            print(f"   {kwargs}")


class Preprocessor:
    def __init__(self):
        self.cluster = TemporalClusterModule(downsample_method=DownSampleMeanMethod(),
                                             sampling_method=FastMiddleEncounterMethod())
        self.children = None


    def get_tree(self):
        return self.children


    def run(self, images, cluster_count = None):

        ### we need to compute the full tree and save the model
        number_of_neighbors = 3
        linkage = 'ward'

        if cluster_count is None:
            cluster_count = len(images) // 100
        if cluster_count <= 0:
            cluster_count = len(images)
        print(f"Running clustering method with {cluster_count} clusters")
        _, rep_indices, all_cluster_labels = self.cluster.run(images, number_of_clusters=cluster_count,
                                                                  number_of_neighbors=number_of_neighbors,
                                                                  linkage=linkage, compute_full_tree = True)
        ## the algorithm automatically computes the full tree
        assert(cluster_count == len(rep_indices))
        children = self.cluster.ac.children_
        assert(len(children) == len(images) - 1)

        self.children = children

        self.mapping = self.cluster.get_mapping(rep_indices, all_cluster_labels)
        return rep_indices


    def run_debug(self, images, **kwargs):
        """
        ## TODO: let's do a bit of debugging to see exactly where things are hanging...
        This function incorporates the clustering with hierarchy generation
        UPDATE: 8/31 -- we will modify the clustering algorithm to stop at len(images) // 100
                     -- if we move beyond this point, we just need to decode the entire video
        :param images:
        :param video_filename:
        :param cluster_count:
        :return:
        """

        ### we need to compute the full tree and save the model
        number_of_neighbors = 3
        linkage = 'ward'

        cluster_count = kwargs.get('cluster_count', 100)
        stopping_point = kwargs.get('stopping_point', len(images) // 100)

        _, rep_indices, all_cluster_labels = self.cluster.run(images, number_of_clusters=cluster_count,
                                                              number_of_neighbors=number_of_neighbors,
                                                              linkage=linkage, compute_full_tree=True)
        ## the algorithm automatically computes the full tree

        children = self.cluster.ac.children_
        self.children = children


        ## note this only has the first part of generating the hierachy!
        return



    def run_final(self, images, hierarchy_save_dir, **kwargs):
        """

        ## TODO: let's do a bit of debugging to see exactly where things are hanging...
        This function incorporates the clustering with hierarchy generation
        UPDATE: 8/31 -- we will modify the clustering algorithm to stop at len(images) // 100
                     -- if we move beyond this point, we just need to decode the entire video
        :param images:
        :param video_filename:
        :param cluster_count:
        :return:
        """
        ### we need to compute the full tree and save the model
        number_of_neighbors = 3
        linkage = 'ward'

        cluster_count = kwargs.get('cluster_count', 100)
        stopping_point = kwargs.get('stopping_point', len(images) // 100)

        _, rep_indices, all_cluster_labels = self.cluster.run(images, number_of_clusters=cluster_count,
                                                              number_of_neighbors=number_of_neighbors,
                                                              linkage=linkage, compute_full_tree=True)
        ## the algorithm automatically computes the full tree

        children = self.cluster.ac.children_
        self.children = children

        assert (len(children) == len(images) - 1)

        hierarchy = self.get_hierarchy(stopping_point = stopping_point)
        self.save_hierarchy(hierarchy_save_dir)

        return hierarchy


    def save_hierarchy(self, save_file):
        save_dir = os.path.dirname(save_file)
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_file, self.hierarchy)


    def get_hierarchy(self, stopping_point = None):
        """
        :param stopping_point: where we stop calculating the hierarchy of the points

        :return:
        """
        hierarchy_list = [] ## as we move on, we add each element to this list
        image_count = self.children.shape[0] + 1
        sampling_method = self.cluster.get_sampling_method()
        current_labels = np.zeros(image_count)

        if not stopping_point:
            stopping_point = image_count // 100

        assert(type(stopping_point) == int)

        print(f"Inside get_hierarchy(), going into loop to measure each timing")
        #timer = Timer()
        i = 0
        while i < stopping_point:
            ## we are performing a bottom-up approach
            #print(f"cluster_labels: {current_labels}")
            #timer.tic()
            rep_indices_list = sampling_method.run(current_labels)   ###TODO we need to input the labels that are output of the clustering algorithm)
            #timer.toc(sampling_method = 'done')
            ## make sure to eliminate the ones that are already in the hierarchy list -- how do we know this?
            ## we have to remember the frame_id and cluster label that we have already chosen the representative frame
            #print(f"rep_indices_list: {rep_indices_list}")
            #timer.tic()
            ignore_cluster_list = self.get_ignore_cluster_labels(hierarchy_list, current_labels)
            #timer.toc(ignore_cluster_labels='done')
            #print(f"ignore_cluster_list: {ignore_cluster_list}")
            #timer.tic()
            final_rep_indices_list = self.get_final_rep_frames(rep_indices_list, ignore_cluster_list, current_labels)
            #timer.toc(get_final_rep_frames='done')
            #print(f"final_rep_indices_list: {final_rep_indices_list}")
            #timer.tic()
            hierarchy_list.extend(final_rep_indices_list)
            #timer.toc(extend_hierarchy_list='done')
            #print(f"hierarchy_list: {hierarchy_list}")
            #timer.tic()
            current_labels = self.update_current_labels(current_labels, i) ### TODO: we do this by using self.children?
            #timer.toc(update_current_labels='done')
            #print(f"\n\n")
            #print(f"count of new label({max(current_labels)})= {sum(current_labels == max(current_labels))}")

            if i == len(hierarchy_list) - 1:
                i += 1

        hierarchy_list = np.array(hierarchy_list)
        self.hierarchy = hierarchy_list
        ### self.hierarchy will be same length as stopping_point
        print(f"length of hierarchy list and stopping point: {len(self.hierarchy)}, {stopping_point}")
        #assert(len(self.hierarchy) == stopping_point)

        return hierarchy_list


    def get_hierarchy_debug(self, stopping_point):
        """
        This function is very similar to the get_hierarchy() except since we specify the number of samples,
        we can derive the mapping behind the frames that have been chosen

        1. Let's think. The children that we refer to, it tells the order of how we go about clustering things
        Hence, if we had 1 cluster (all the points were grouped together) we would use the middle encounter method to summarize
        Hence, it would be
            1. perform clustering using the groupings outlined in self.children ie, get the labels for all the clusters
            2. use the middle encounter method to pick the representative frames
        :return:
        """

        hierarchy_list = [] ## as we move on, we add each element to this list
        image_count = self.children.shape[0] + 1
        sampling_method = self.cluster.get_sampling_method()
        current_labels = np.zeros(image_count, dtype = np.int)
        i = 0
        while len(hierarchy_list) < stopping_point:
            #print(f"cluster_labels: {current_labels}")
            """
            Things we can check:
            1. Current labels needs to have at least 1 number of everything
            2. rep_indices should return as many the existing number of labels out there
                - if 0 to 65 then 66 rep indices should be in the array
            """

            ##### DEBUGGING CODE #####
            """
            debug_array = np.zeros(max(current_labels) + 1)
            for label in current_labels:
                debug_array[label] = 1
            for ii in range(len(debug_array)):
                if debug_array[ii] != 1:
                    print(f"{i} MISSING {ii}!!!! Something is wrong!!!!")
                    break
            """
            ##### END OF DEBUGGING CODE #####

            rep_indices_list = sampling_method.run(current_labels) # TODO: gives rep indices on all the current labels
            ##
            ## we have to remember the frame_id and cluster label that we have already chosen the representative frame
            print(f"max, min of current labels: {max(current_labels)}, {min(current_labels)}")
            ######
            if len(rep_indices_list) != max(current_labels) + 1:
                print(f"{i}Sampling Method ERROR: number of rep frames does not match the number of clusters in array")
                ### we will save all the information here
                save_dir = os.path.join('/nethome/jbang36/eva_jaeho/eva_storage/jvc_experiments/debug_arrays', 'tmp.npy')
                np.save(save_dir, current_labels)

                break


            print(f"rep_indices_list: {len(rep_indices_list)}")
            ignore_cluster_list = self.get_ignore_cluster_labels(hierarchy_list, current_labels)
            print(f"ignore_cluster_list: {len(ignore_cluster_list)}")
            final_rep_indices_list = self.get_final_rep_frames(rep_indices_list, ignore_cluster_list, current_labels)
            print(f"final_rep_indices_list: {len(final_rep_indices_list)}")
            hierarchy_list.extend(final_rep_indices_list)
            print(f"hierarchy_list: {len(hierarchy_list)}")
            if len(hierarchy_list) != stopping_point: ### we shouldn't do this at the end
                current_labels = self.update_current_labels(current_labels, i)
            print(f"\n\n")
            #print(f"count of new label({max(current_labels)})= {sum(current_labels == max(current_labels))}")
            #print(f"i = {i}, len(hierarchy_list) = {len(hierarchy_list)}")
            ## okay, if we have filtered out more than 1, what will happen?
            print(f"{i} len of final_rep_indices_list: {len(final_rep_indices_list)}, len(hierarchy_list): {len(hierarchy_list)}")

            i += 1

        hierarchy_list = np.array(hierarchy_list)

        return hierarchy_list, current_labels


    def update_labels_recursive(self, current_labels, new_label, child):
        n_samples = len(current_labels)
        if child < n_samples:
            current_labels[child] = new_label
        else:
            self.update_labels_recursive(current_labels, new_label, self.children[child - n_samples][0])
            self.update_labels_recursive(current_labels, new_label, self.children[child - n_samples][1])

        return


    def update_current_labels(self, current_labels, current_index):
        """
        we update the current_labels using self.children

        :param current_labels: current labels
        :param current_index: where we are working on at the moment
        :return:
        """
        ## we need the offset 2 because (1) shape of self.children is n-1
        current_split_operation = len(current_labels) - current_index - 2
        new_label = max(current_labels) + 1
        n_samples = len(current_labels)
        ## backtracing will take a lot of effort, what we need to do is
        ## if we encounter a number that is greater than len(current_labels), then we have to backtrace that as well
        ## we only need to modify 1 of the targets, the other one can stay the same

        ## backtracing -- basically what we need to do is trace back to the row it refers to and make the elements there all max labels
        cluster_top = self.children[current_split_operation][0]
        if cluster_top < n_samples:
            current_labels[cluster_top] = new_label
        else:
            self.update_labels_recursive(current_labels, new_label, cluster_top)

        return current_labels


    def get_ignore_cluster_labels(self, hierarchy_list, current_labels):
        """
        hierarchy_list contains the frames that have been selected as rep_frame (frame_id)
        current_labels contains the frames that labels to all the frames within the list

        :param hierarchy_list: list()
        :param current_labels: list()
        :return: list()
        """
        ignore_labels = []
        for element in hierarchy_list:
            ignore_labels.append(current_labels[element])

        return ignore_labels


    def get_final_rep_frames(self, rep_indices_list, ignore_cluster_list, current_labels):
        """
        Get the final representative frames after excluding the ones that have already been added
        :param rep_indices_list:
        :param ignore_cluster_list:
        :param current_labels:
        :return:
        """

        ## so now we have rep_indices_list, ignore_cluster_list, and current_labels
        final_rep_indices_list = []
        for rep_index in rep_indices_list:
            if not current_labels[rep_index] in ignore_cluster_list:
                final_rep_indices_list.append(rep_index)

        return final_rep_indices_list

    def get_mapping(self):
        return self.mapping


if __name__ == "__main__":
    pass

