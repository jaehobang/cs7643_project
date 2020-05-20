"""

In this file, we will write functions to fully visualize and analyze the performance of our clustering methods


"""



### to understand the performance of our algorithms, we need to understand what examples gave us failures.
# For sampling, we are all dealing with representative frames / mappings,
# we will use this information to derive the information we need
def examine_wrong_examples(propagated_labels, gt_labels, mapping):

    wrong_examples = {}
    for i in range(len(propagated_labels)):
        if propagated_labels[i] != gt_labels[i]:
            wrong_examples[i] = [mapping[i], propagated_labels[i], gt_labels[i], propagated_labels[mapping[i]], gt_labels[mapping[i]]]
            # order is [index_of_rep_frame, it's proposed_label, it's gt label, the rep frame proposed_label (should be same as it's proposed label), rep frame gt_label]

    return wrong_examples
