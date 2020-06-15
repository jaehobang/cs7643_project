"""
We implement a label converter for UAdetrac,

This will convert the given labels to binary form, different query form depending on what our needs are
"""

class UADetracConverter:

    def __init__(self):
        pass

    @staticmethod
    def drawTrueFalse(labels):
        total_length_of_viz = 30
        import matplotlib.pyplot as plt
        y_ = []
        grouping = len(labels) // total_length_of_viz
        for i in range(0, len(labels), grouping):
            end = i+grouping
            if end >= len(labels):
                end = len(labels)
            y_.append(sum(labels[i:end]))
        x_ = [i for i in range(len(y_))]
        assert(len(x_) == len(y_))
        plt.plot(x_, y_, 'ro')

    @staticmethod
    def getTrueFalseCount(labels):
        true_count = 0
        false_count = 0
        nan_count = 0
        for label in labels:
            if label is None:
                nan_count += 1
            else:
                if label == 1:
                    true_count += 1
                else:
                    false_count += 1

        print(f"ALL: {len(labels)}, TRUE: {true_count}, FALSE: {false_count}, NAN: {nan_count}")
        return

    @staticmethod
    def convert2limit_queries2(labels, category_dict, operator = 'and'):
        ## assumption for labels is [['car', 'bus', ...'], ['car', 'bus', ...]]
        count_dict = {}
        for key in category_dict:
            count_dict[key] = 0

        new_labels = []
        for label in labels:
            if label is None:
                new_labels.append(None)
            else:
                for veh in label:
                    if veh not in category_dict.keys():
                        print(f"unforeseen vehicle type: {veh}")
                        continue
                    else:
                        count_dict[key] += 1

                if operator == 'and':
                    condition = True
                    for key in category_dict.keys():
                        if count_dict[key] < category_dict[key]:
                            condition = False
                elif operator == 'or':
                    condition = False
                    for key in category_dict.keys():
                        if count_dict[key] >= category_dict[key]:
                            condition = True
                if condition:
                    new_labels.append(1)
                else:
                    new_labels.append(0)

        assert (len(new_labels) == len(labels))

        return new_labels




    @staticmethod
    def convert2limit_queries(labels, car=1, bus=0, van=0, others=0):
        ## assumption for labels is [['car', 'bus', ...'], ['car', 'bus', ...]]
        new_labels = []
        for label in labels:
            if label is None:
                new_labels.append(None)
            else:
                car_count = 0
                bus_count = 0
                van_count = 0
                others_count = 0
                for veh in label:
                    if veh == 'car':
                        car_count += 1
                    elif veh == 'bus':
                        bus_count += 1
                    elif veh == 'van':
                        van_count += 1
                    elif veh == 'others':
                        others_count += 1
                    else:
                        print(f"unforseen vehicle type: {veh}")

                if car_count >= car and bus_count >= bus and van_count >= van and others_count >= others:
                    new_labels.append(1)
                else:
                    new_labels.append(0)

        assert(len(new_labels) == len(labels))



        return new_labels