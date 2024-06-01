import sys
import os
import copy
import shutil
import glob,os.path
import re
from functools import reduce




# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

def list_run_dirs(path_to_datastore):  
    depth1 = glob.glob(f'{path_to_datastore}/*')
    dirsDepth1 = filter(lambda f: os.path.isdir(f), depth1)
    return list(dirsDepth1)


def list_step_dirs(path_to_dir):  
    depth1 = glob.glob(f'{path_to_dir}/*')
    dirsDepth1 = filter(lambda f: os.path.isdir(f), depth1)
    return list(dirsDepth1)


def list_container_dirs(path_to_dir):  
    depth2 = glob.glob(f'{path_to_dir}/*')
    dirsDepth2 = filter(lambda f: os.path.isdir(f), depth2)
    return list(dirsDepth2)


def list_map_dirs(path_to_dir):  
    filesDepth3 = glob.glob(f'{path_to_dir}/*/Map')
    dirsDepth3 = filter(lambda f: os.path.isdir(f), filesDepth3)
    return list(dirsDepth3)

def list_study_dirs(path_to_dir):  
    filesDepth3 = glob.glob(f'{path_to_dir}/*/[!Map]*')
    dirsDepth3 = filter(lambda f: os.path.isdir(f), filesDepth3)
    return list(dirsDepth3)


def list_series_dirs(path_to_dir): 
    globExp = glob.glob(f'{path_to_dir}/*/[!Map]*/*')
    dirsSeries = filter(lambda f: os.path.isdir(f), globExp)
    return list(dirsSeries)


def list_resources_dirs(path_to_dir,rsc_type):
#     if rsc_type == "steps": return list_step_dirs(path_to_dir)
    if rsc_type == "containers": return list_container_dirs(path_to_dir)  
    if rsc_type == "maps": return list_map_dirs(path_to_dir)   
    if rsc_type == "studies": return list_study_dirs(path_to_dir)         
    if rsc_type == "series": return list_series_dirs(path_to_dir) 
    if rsc_type == "runs": return list_run_dirs(path_to_dir)  # this is dangerous ... 
    return []


def list_files(path_to_dir,file_type):
    _,_,files_in_dir = tuple(os.walk(path_to_dir))
    return [file for file in files_in_dir if file.split('.')[-1].lower()=="pkl"]



    

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


def build_output(url):
    output_keys = ("run","container","study","series","instance")
    try:
        return {key:value for key,value in zip(output_keys,tuple(url.split(os.sep)))}
    except: return {} # Just in case

    
def filter_against_constraints(resources,constraints):

    safe_constraints = {key:value for key,value in copy.deepcopy(constraints).items() if str(value)} # no empty constraints
    clean_resources = []
    for rsc in resources:
        checks = [rsc.get(key,"") == value for key,value in safe_constraints.items()]
        if all(checks) : clean_resources.append(rsc)
    return clean_resources



def list_runs(datastore,constraints={},rest_output=False):
    '''Lists the run dirs in the datastore 
    These run dirs are defined as folder just below the root of the datastore
    No attempt is made to check the validity of their structure
    Validation should have taken place upon upload/creation '''
    
    try:
        all_resources = [rsc for rsc in list_run_dirs(datastore)]
        all_resources = [rsc for rsc in all_resources if len(str(os.path.normpath(rsc)).split(os.path.normpath(datastore)))>1]  # Optional 
        all_resources = [build_output(os.path.relpath(rsc,datastore)) for rsc in all_resources] # final tranformation        
        if rest_output: all_resources = ["/".join(rsc.values()) for rsc in all_resources] # Back to REST conventions
        return all_resources
    except: return []

    
def list_run_resources(resource,datastore,constraints={}, rest_output=False):
    '''Lists the resource levels dirs in a given run in the datastore 
    No attempt is made to check the validity of the structure
    Validation should have taken place upon upload/creation '''
    
    try:
        run_name,rsc_type = tuple(resource.split('/'))
        run_folder = os.path.join(datastore,run_name) # Where we search
#         print(f"{resource} {f'under constraint {constraints}' if constraints else ''}")

        all_resources = [rsc for rsc in list_resources_dirs(run_folder,rsc_type)]
        all_resources = [rsc for rsc in all_resources if len(str(os.path.normpath(rsc)).split(os.path.normpath(run_folder)))>1]  # Optional 
        all_resources = [build_output(os.path.relpath(rsc,datastore)) for rsc in all_resources] # final tranformation 

        if constraints: all_resources = filter_against_constraints(all_resources,constraints)

        if rest_output: all_resources = ["/".join(rsc.values()) for rsc in all_resources] # Back to REST conventions
        return all_resources
    except: return []
    

def list_file_resources(resource,datastore,**kwargs):
    
    
    try:
        file_type = resource.split('/')[-1]
        file_location = '/'.join(resource.split('/')[0:-1])
        complete_location = os.path.join(datastore,file_location)

        _,_,files_in_dir = tuple(os.walk(complete_location))[0]
        all_files = [os.path.join(file_location,file) for file in files_in_dir if file.split('.')[-1].lower()==file_type.lower()]
#         print(all_files)
#         os.path.relpath(file,datastore)
        return all_files
    except: return []


    
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------



def get_resources(request,datastore,constraints={},rest_output=False):
    resource = request.strip().strip("./")
    run_pattern = re.compile(r"^(?!.*\/.*\/).*runs$") 
    general_resource_pattern = re.compile(r"^(?!.*\/.*\/).*/steps|studies|series|containers|instances$")

    pkl_pattern = re.compile(r"^([^\/]+)\/([^\/]+)\/([^\/]+)/pkl$")
    json_pattern = re.compile(r"^([^\/]+)\/([^\/]+)\/([^\/]+)/json$")
#     print(rest_output)
    
    if run_pattern.findall(resource):
        return list_runs(datastore,constraints=constraints,rest_output=rest_output)
    if general_resource_pattern.findall(resource):
        return list_run_resources(resource,datastore,constraints=constraints,rest_output=rest_output)
    if pkl_pattern.findall(resource):
        return list_file_resources(resource,datastore,constraints=constraints,rest_output=rest_output)
    return [] 




def build_api(datastore):
    '''
    curries get resources - datastore is fixed for the whole script
    '''
    
    def curried_api(request,constraints={},rest_output=False): 
        return get_resources(request,datastore,constraints,rest_output)
    return curried_api

    
    

