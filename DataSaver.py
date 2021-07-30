import pandas as pd
import numpy as np
import os
from FileLoader import Read_Txt
import sys


# %% Save Tracts and Endpoint Statistics. Each csv contains all tracts for a single features.
def SaveStatistics(DataDict, keyword1, keyword2, location):
    """
	Save Tracts and Endpoint Statistics. if keyword1=='all':
		Each csv contains all tracts for a single features.
	if keyword2 =='all':
		Each csv contains all features for a single tract.
	:type keyword2: str
	:param DataDict
	:param keyword1
	:param keyword2
	:return
	"""

    if keyword1 == 'all':
        TractNames = list(DataDict.keys())
        TableIndices = [subject[0] for subject in DataDict[TractNames[0]][keyword2]]
        values = {}
        for key in TractNames:
            values[key] = [subject[1] for subject in DataDict[key][keyword2]]
        filename = location + '\\' + keyword2 + '.csv'
        pd.DataFrame(values, index=TableIndices).to_csv(filename)
    elif keyword2 == 'all':
        FeatureNames = list(DataDict[keyword1].keys())
        TableIndices = [subject[0] for subject in DataDict[keyword1][FeatureNames[0]]]
        values = {}
        for key in FeatureNames:
            if key not in ['bounding box x', 'bounding box y', 'bounding box z']:
                values[key] = [subject[1] for subject in DataDict[keyword1][key]]

        filename = location + '\\' + keyword1 + '.csv'
        pd.DataFrame(values, index=TableIndices).to_csv(filename)


# %% Save Tract profiles
def SaveTractProfiles(TractProfile, location='.', Sampling_Strategy=''):
    """
	:param TractProfile:
	:param location, where to save tractprofile?
	"""
    subs = []
    tr = []
    valuestype = ['.y_axis', '.CI_max', '.CI_min']
    for tractname in TractProfile.keys():
        filename = tractname + '.' + Sampling_Strategy[0] + '_' + Sampling_Strategy[1]
        indexes = list(TractProfile[tractname][Sampling_Strategy].keys())
        indexes = [idx + val for idx in indexes for val in valuestype]
        indexes.insert(0, 'x_axis')
        Data = np.array([])
        for subject in TractProfile[tractname][Sampling_Strategy].keys():
            if Data.size == 0:
                Data = TractProfile[tractname][Sampling_Strategy][subject]
            else:
                try:
                    Data = np.r_[Data, TractProfile[tractname][Sampling_Strategy][subject][1:]]
                except:
                    print('value error, sampling strategy=', Sampling_Strategy, tractname, subject)
                    subs.append(subject)
                    tr.append(tractname)

        filename = location + '\\' + filename + '.csv'
        pd.DataFrame(Data, index=indexes).to_csv(filename)


# %%
def concat_lateralities(input_loc, target_loc):
	"""
	concat left and right tracts.
	:param input_loc: input directory
	:param target_loc: target directory to save new tables.
	:return: row-wise concatenated tables. saved in target_loc directory
	"""
	tract_names = Read_Txt('TractNames.txt')
	tracts = [tract + '.csv' for tract in tract_names]
	os.chdir(input_loc)
	laterality = {'L': 'Left', 'R': 'Right'}
	for i, tract_name1 in enumerate(tracts):
		for tract_name2 in tracts:
			tract1_abv = tract_name1.replace('.csv','').split('_')
			tract2_abv = tract_name2.replace('.csv','').split('_')
			if tract1_abv[-1] == 'L' and tract2_abv[-1] == 'R' and tract1_abv[:-1] == tract2_abv[:-1]:
				df1 = pd.read_csv(tract_name1, index_col=0)
				df2 = pd.read_csv(tract_name2, index_col=0)
				df1['laterality'] = laterality[tract1_abv[-1]]
				df2['laterality'] = laterality[tract2_abv[-1]]
				df = pd.concat([df1, df2], axis=0)
				df = df[df['number of tracts'] != 0]
				if df.shape[0] != (df1[df1['number of tracts'] != 0].shape[0] + df2[df2['number of tracts'] != 0].shape[0]):
					print(tract_name1, tract_name2)
				else:
					print('ok', tract_name1)
				filename = target_loc + tract_name2[:-6] + '.csv'
				pd.DataFrame(df).to_csv(filename)




	return


# %%
def concat_endpoints(input_loc, target_loc):
	"""
	concat endpoint tables.
	:param input_loc: input directory
	:param target_loc: target directory to save new tables.
	:return: row-wise concatenated tables. saved in target_loc directory
	"""
	tract_names = Read_Txt('TractNames.txt')
	endpoint_name = [' endpoints1.csv', ' endpoints2.csv']
	endpoints = [tract + endpoint for tract in tract_names for endpoint in endpoint_name]
	os.chdir(input_loc)
	end_region = {'endpoints1': 'r1', 'endpoints2': 'r2'}
	for name1 in endpoints:
		for name2 in endpoints:
			end1 = name1.strip('.csv').split(' ')
			end2 = name2.strip('.csv').split(' ')
			if end1[0] == end2[0] and end1[1] == 'endpoints1' and end2[1] == 'endpoints2':
				df1 = pd.read_csv(name1, index_col=0)
				df2 = pd.read_csv(name2, index_col=0)
				df1['endpoint_region'] = end_region[end1[1]]
				df2['endpoint_region'] = end_region[end2[1]]
				df = pd.concat([df1, df2], axis=0)
				df = df[df['voxel counts'] != 0]
				if df.shape[0] != (df1[df1['voxel counts'] != 0].shape[0] + df2[df2['voxel counts'] != 0].shape[0]):
					print(end1, end2)
				else:
					print('ok', end1, end2)
				filename = target_loc + 'endpoints_' + end1[0] + '.csv'
				pd.DataFrame(df).to_csv(filename)



def concat_tract_with_endpoint(tract_loc, endpoint_loc, target_loc):
	"""
	concat endpoint1 and 2 features with tract features in each laterality.
	:param input_loc: input directory
	:param target_loc: target directory to save new tables.
	:return: row-wise concatenated tables. saved in target_loc directory
	"""
	tract_names = Read_Txt('TractNames.txt')
	tracts = os.listdir(tract_loc)
	tracts_dir = [tract_loc + tract for tract in tracts]

	endpoints = os.listdir(endpoint_loc)
	endpoints_dir = [endpoint_loc + endpoint for endpoint in endpoints]


	for i, tract_dir in enumerate(tracts_dir):
		counter = 0
		for j, endpoint_dir in enumerate(endpoints_dir):
			df_tract = pd.read_csv(tract_dir, index_col=0)
			df_tract_left = df_tract[df_tract['laterality']=='Left']
			df_tract_right = df_tract[df_tract['laterality']=='Right']

			tract_name = tracts[i].replace('.csv','')
			endpoint_name = endpoints[j].replace('.csv','')
			if tract_name in endpoint_name:
				counter += 1
				df_endpoint = pd.read_csv(endpoint_dir, index_col=0)
				df_endpoint_r1 = df_endpoint[df_endpoint['endpoint_region']=='r1']
				df_endpoint_r1 = df_endpoint_r1.add_prefix('r1_')
				df_endpoint_r2 = df_endpoint[df_endpoint['endpoint_region']=='r2']
				df_endpoint_r2 = df_endpoint_r2.add_prefix('r2_')

				if endpoint_name[-1] == 'L':
					df_left = pd.concat([df_tract_left, df_endpoint_r1, df_endpoint_r2], axis=1)
				else:
					try:
						df_right = pd.concat([df_tract_right, df_endpoint_r1, df_endpoint_r2], axis=1)
					except:
						print(e)


				# early break
				if counter == 2:
					break

		df = pd.concat([df_left, df_right], axis=0)
		df.pop('r1_endpoint_region')
		df.pop('r2_endpoint_region')
		filename = target_loc + tract_name + '.csv'
		pd.DataFrame(df).to_csv(filename)
	return


#%% Test
# %% concat lateralities:
# input_loc = r'C:\Users\AliKazemi\Desktop\Features\TractFeatures\ex2_TractFeatures'
# target_loc =  r'C:\\Users\\AliKazemi\\Desktop\\Features\\TractFeatures_LR\\ex2\\'
# concat_lateralities(input_loc, target_loc)

# %% concat endpoints:
# input_loc = r'C:\Users\AliKazemi\Desktop\Features\EndpointFeatures\ex2_endpointfeatures'
# target_loc = r'C:\\Users\\AliKazemi\Desktop\\Features\\endpoints_ep1ep2\\ex2\\'
# concat_endpoints(input_loc, target_loc)

#%% concat tract with endpoints
tract_loc = r'C:\\Users\\AliKazemi\Desktop\\Features\\\TractFeatures_LR\\ex2\\'
endpoint_loc = r'C:\\Users\\AliKazemi\\Desktop\\Features\\endpoints_ep1ep2\\ex2\\'
target_loc = r'C:\\Users\\AliKazemi\\Desktop\\Features\\Tracts_Endpoints\\ex2\\'
concat_tract_with_endpoint(tract_loc, endpoint_loc, target_loc)
#%%
# import os
# import glob
# import numpy as np
# import pandas as pd
# from DataLoader import Read_Txt, StatisticsReader, Tract_Features, Read_TractProfiles, Upgrade_TractProfile
# from DataSaver import SaveTractProfiles, SaveStatistics
#
# os.chdir(r'C:\Users\AliKazemi\PycharmProjects\Thesis')
# # load tract names
# TractsList = Read_Txt('TractNames.txt')
#
# # load tract_profilenames
# Profiles = Read_Txt('TractProfileNames.txt')
#
# # load subject information. i.e. their weights and head circumference
# SubjectInformation = pd.read_csv('SubjectInformation.csv')
#
# os.chdir(r'C:\Users\AliKazemi\PycharmProjects\Thesis\ExtractedFeatures')
# Subjects = os.listdir('.')
# CurrentFolder = os.getcwd()
#
# IncompleteSubjects = []
# ids = list(range(1, 34))
# ids.remove(26)
# for idx in ids:
# 	print(idx)
# 	Tract_Profiles = dict(dict(zip(TractsList, [{} for _ in range(len(TractsList))])))
# 	for subject in Subjects:
# 		os.chdir(CurrentFolder)
# 		os.chdir(subject)
# 		# Tract Statistics:
# 		inputs = os.listdir()
# 		# check if all of the features are recorded for the subject:
# 		if len(inputs) != 35:
# 			IncompleteSubjects.append(subject)
# 			continue
#
# 		Tract_Profiles = Upgrade_TractProfile(inputs, Tract_Profiles, Profiles, idx, subject)
# 	Sampling_Strategy = list(Tract_Profiles['Vertical_Occipital_Fasciculus_R'].keys())[0]
# 	SaveTractProfiles(Tract_Profiles, location=r'C:\Users\AliKazemi\Desktop\Features\TractProfileFeatures', Sampling_Strategy=Sampling_Strategy)
#
# print('Incomplete Tract Profiles:\n', IncompleteSubjects)
