import json

name = ''
data = []
top_list_gender = []
top_list_region = []
top_list_overall = []
count_gender_m = 0
count_gender_f = 0
count_region = 0
count_samples = 0
name_region_str = ''
region = ''


features_dict = {"meta": {"view": {"database_name": 'general_database', "overall": top_list_overall, "columns": [{"id": 0, "field_name": 'id'}, {"id": 1, "field_name": 'name'}, {"id": 2, "field_name": 'gender', "top": top_list_gender}, {"id": 3, "field_name": 'region', "top": top_list_region}, {"id": 4, "field_name": "samples"}]}}}
k = -1
x = -1
m = -1
region_counter = 1
name_counter = 1

with open('database_txt.txt', 'r') as f:
    datafile = f.readline()
    while datafile != '':
        x += 1
        if datafile[4] == "m":
            count_gender_m += 1
        else:
            count_gender_f += 1

        if datafile[2] != region and k == 0 or count_gender_f+count_gender_m == 4200:
            name_region_str = 'region{0}'.format(region_counter)
            top_list_region.append({"item": name_region_str, "count": count_region})
            region_counter += 1
            count_region = 0
        else:
            count_region += 1
            k = 0

        if datafile[5:9] != name and m == 0 or count_gender_f+count_gender_m == 4200:
            name_str = 'name{0}'.format(name_counter)
            name_counter += 1
            count_samples = 0
        else:
            count_samples += 1
            m = 0

        name = datafile[5:9]
        region = datafile[2]

        list_element = [x, name, datafile[4], datafile[2], datafile[10:len(datafile) - 1]]

        data.append(list_element)
        datafile = f.readline()

top_list_gender.append({"item": 'male_samples', "count": count_gender_m})
top_list_gender.append({"item": 'female_samples', "count": count_gender_f})
top_list_gender.append({"item": 'male_speaker', "count": count_gender_m/10})
top_list_gender.append({"item": 'female_speaker', "count": count_gender_f/10})

top_list_overall.append({"item": 'speaker', "count": name_counter-1})
top_list_overall.append({"item": 'samples', "count": count_gender_f+count_gender_m})
top_list_overall.append({"item": 'regions', "count": region_counter-1})
top_list_overall.append({"item": 'samples_per_speaker', "count": 10})

data_dict = {"data": data}
features_dict.update(data_dict)
with open('database.json', 'w') as js:
    write_data = json.dumps(features_dict, indent=1)
    js.write(write_data)

