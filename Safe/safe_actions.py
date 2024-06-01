import os
import shutil
import json
import pickle
import pandas as pd


# These are useful File/Folder Management actions 
# They have been written to be safe so do not raise errors and crash the flow
# Instead, all events are logged, including errors
# The functions return a dictionary that includes all details on the action - including whether it has succeded or not


def safely_create_folder(folder_to_create,**options):
    event_logs = []
    error_logs = []
    delete = options["delete"] if "delete" in options.keys() else True

    try:
        target_exists = os.path.isdir(folder_to_create)
        if target_exists and delete:
            shutil.rmtree(folder_to_create)
            event_logs.append(f"Existing folder {folder_to_create} has been deleted")

        os.mkdir(folder_to_create)
        event_logs.append(f"Folder {folder_to_create} has been created")
        return {"msg": f"Creation of folder {folder_to_create} has successfully completed", "status":"Success",
                "boolean_status":True, "error_logs":error_logs, "event_logs":event_logs}
    
    except:
        error_logs.append("Folder creation has failed")
        return {"msg": f"Creation of folder {folder_to_create} has failed","status":"Failure",
                "boolean_status":False, "error_logs":error_logs,"event_logs":event_logs}


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

    
def safely_delete_folder(folder_to_delete,**options):
    event_logs = []
    error_logs = []

    try:
        target_exists = os.path.isdir(folder_to_delete)
        if target_exists:
            shutil.rmtree(folder_to_delete)
            event_logs.append(f"Existing folder {folder_to_delete} has been deleted")
            return {"msg": f"Deletion of folder {folder_to_delete} has successfully completed", "status":"Success",
                    "boolean_status":True, "error_logs":error_logs, "event_logs":event_logs}

        return {"msg": f"There was no  folder {folder_to_delete} to delete!!!", "status":"Success",
                "boolean_status":True, "error_logs":error_logs, "event_logs":event_logs}
    except:
        error_logs.append(f"Deletion of folder {folder_to_delete} has failed")
        return {"msg": f"Deletion of folder {folder_to_delete} has failed", "status":"Failure",
                "boolean_status":False, "error_logs":error_logs, "event_logs":event_logs}


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

    
def safely_copy_file(file_to_copy,copy_destination,**options):
    event_logs = []
    error_logs = []

    try:
        file_exists = os.path.isfile(file_to_copy)
        copy_already_exists = os.path.isfile(copy_destination)
        if copy_already_exists:
            os.remove(copy_destination)
            event_logs.append(f"File {copy_destination} has been deleted")  
            
        if file_exists:
            shutil.copyfile(file_to_copy,copy_destination)
            event_logs.append(f"File {file_to_copy} has been copied to {copy_destination}")
            return {"msg": f"File {file_to_copy} has been successfully copied to {copy_destination}",
                    "boolean_status":True, "status":"Success", "error_logs":error_logs,"event_logs":event_logs}

        error_logs.append(f"File {file_to_copy} does not exist")
        error_logs.append(f"File {file_to_copy} could not be copied to {copy_destination}")
        return {"msg": f"File {file_to_copy} could not be copied to {copy_destination}","status":"Error", 
                "boolean_status":False, "error_logs":error_logs,"event_logs":event_logs}
        
    except:
        error_logs.append(f"File {file_to_copy} could not be copied to {copy_destination}")
        return {"msg": f"File {file_to_copy} could not be copied to {copy_destination}",
                "boolean_status":False, "status":"Failure", "error_logs":error_logs,"event_logs":event_logs}
    
    
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------  

def safely_write_json(data,json_destination,**options):
    event_logs = []
    error_logs = []
    
    try:
        json_already_exists = os.path.isfile(json_destination)
        if json_already_exists:
            os.remove(json_destination)
            event_logs.append(f"File {json_destination} has been deleted")  
            
        with open(json_destination, 'w') as json_file:
            json.dump(data, json_file, indent = 6)
            event_logs.append(f"File {json_destination} has been created") 
            
        return {"msg": f"Creation of JSON file {json_destination} has successfully completed","status":"Success", 
                "boolean_status":True, "error_logs":error_logs, "event_logs":event_logs}
    except:
        error_logs.append(f"Creation of JSON file {json_destination} has failed")
        return {"msg": f"JSON file {json_destination} could not be created","status":"Failure", 
                "boolean_status":False, "error_logs":error_logs,"event_logs":event_logs}
    
# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------- 


def safely_read_json(json_file,**options):
    event_logs = []
    error_logs = []
    
    try:
        with open(json_file) as json_content:
            content = json.load(json_content) 
        event_logs.append(f"JSON file {json_file} could be read")
        return {"data":content, "msg": f"JSON file {json_file} could be read","status":"Success", 
                "boolean_status":True, "error_logs":error_logs,"event_logs":event_logs}
    except:
        error_logs.append(f"JSON file {json_file} cound not be read")
        return {"data":{}, "msg": f"JSON file {json_file} cound not be read","status":"Failure", 
                "boolean_status":False, "error_logs":error_logs,"event_logs":event_logs}
    

    
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------  

def safely_write_dataframe_to_pkl(data,pkl_destination,**options):
    event_logs = []
    error_logs = []
    
    try:
        pkl_already_exists = os.path.isfile(pkl_destination)
        if pkl_already_exists:
            os.remove(pkl_destination)
            event_logs.append(f"File {pkl_destination} has been deleted")  
            
        data.to_pickle(pkl_destination)
        event_logs.append(f"File {pkl_destination} has been created") 
            
        return {"msg": f"Creation of Pickle file {pkl_destination} has successfully completed","status":"Success", 
                "boolean_status":True, "error_logs":error_logs, "event_logs":event_logs}
    except:
        error_logs.append(f"Creation of Pickle file {json_destination} has failed")
        return {"msg": f"Pickle file {pkl_destination} could not be created","status":"Failure", 
                "boolean_status":False, "error_logs":error_logs,"event_logs":event_logs}
    
# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------- 


def safely_read_pkl_to_dataframe(pkl_file,**options):
    event_logs = []
    error_logs = []
    
    try:
        content = pd.read_pickle(pkl_file)
        
        event_logs.append(f"Picklee {pkl_file} could be read")
        return {"data":content, "msg": f"Pickle {pkl_file} could be read","status":"Success", 
                "boolean_status":True, "error_logs":error_logs,"event_logs":event_logs}
    except:
        error_logs.append(f"Pickle {pkl_file} could not be read")
        return {"data":{}, "msg": f"Pickle {pkl_file} cound not be read","status":"Failure", 
                "boolean_status":False, "error_logs":error_logs,"event_logs":event_logs}
    

# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------- 


def safely_read_csv_to_dataframe(csv_file,**options):
    event_logs = []
    error_logs = []
    
    try:
        content = pd.read_csv(csv_file)
        
        event_logs.append(f"CSV {csv_file} could be read")
        return {"data":content, "msg": f"CSV {csv_file} could be read","status":"Success", 
                "boolean_status":True, "error_logs":error_logs,"event_logs":event_logs}
    except:
        error_logs.append(f"CSV {csv_file} could not be read")
        return {"data":{}, "msg": f"CSV {csv_file} cound not be read","status":"Failure", 
                "boolean_status":False, "error_logs":error_logs,"event_logs":event_logs}
    