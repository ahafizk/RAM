
mapping = {
    # "Bathroom":     [\
    #                 "brushing",\
    #                 "dancing",\
    #                 # "standing",\
    #                 "washinghands",\
    #                 # "washingface",\
    #                 # "sitting",\
    #                 ],\
    # "Cooking":[\
    #             "stirring",\
    #             "chopping",\
    #             # "frying",\
    #             "dancing",\
    #             # "standing",\
    #         ],\
    "Cleaning":[\
                "wiping",\
                # "mopping",\
                # "vacuuming",\
                "spraying",\
                # "standing",\
                ],\
    # "Entertainment":[\
    #                 # "reading",\
    #                 "sitting",\
    #                 # "watching",\
    #                 "dancing",\
    #                 # "walking", \
    #                 "standing"\
    #                 ]
}


# mapping = {
#     # "Bathroom":     [\
#     #                 # "sitting",\
#     #                 "brushing",\
#     #                 # "facewash"\
#     #                 ],\
#     # "Cooking":[\
#     #                 "sitting",\
#     #                 #"washinghands",\
#     #                 "nothing"
#     #
#     #         ],
#     "Cleaning":[\
#                     "brushing",\
#                     # "spraying",\
#                     "wiping",\
#                     # "wiping",\
#                 ],
#     "Entertainment":[\
#                     # "dumbling",\
#                     "nothing",\
#                     #"washinghands"
#                     #"washinghands"\
#                     ]
# }

#mapping = {
    #"Bathroom":     [\
                    #"brushing",\
                    #"standing",\
                    #"washingface",\
                    #"washinghands",\
                #"stirring",\
                #"chopping",\
                #"washinghands"\
                    #],\
    #"Cooking":[\
                #"stirring",\
                #"chopping",\
                #"washinghands",\
                #"frying",\
                    #"brushing",\
                    #"standing",\
                    #"washingface",\
                    #"washinghands"\
            #],
    #"Cleaning":[\
                #"stirring",\
                #"chopping",\
                #"washinghands",\
                #"frying",\
                #"wiping",\
                #"mopping",\
                #"vacuuming",\
                #"standing"\
                #],\
    #"Entertainment":[\
                    #"reading",\
                #"chopping",\
                #"washinghands",\
                #"frying",\
                #"wiping",\
                #"mopping",\
                #"vacuuming",\
                    #"sitting",\
                    #"watching",\
                    #"dancing",\
                    #"standing"\
                    #]
#}


def get_mapping():
    return mapping

def get_activities():
    activities = []
    for activity in mapping:
        activities.append(activity)

    return activities


def get_subactivities():
    subactivities = []
    
    for activity in mapping:
        subactivities.extend(mapping[activity])

    return list(set(subactivities))


def get_subactivity_class(name):
    subactivities = get_subactivities()
    # subactivities.sort()
    # print subactivities
    sub_name = name.replace('.','-').split('-')[0]
    # print sub_name
    if subactivities.count(sub_name) == 0:
        return -1
    else:
        # Handle merging sub activities here
        #if sub_name in ['wiping','mopping','vacuuming']:
            #return subactivities.index('mopping')
        #elif sub_name in ['sitting','watching','reading','standing']:
            #return subactivities.index('sitting')

        return subactivities.index(sub_name)

