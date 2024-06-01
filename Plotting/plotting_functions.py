import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize, FuncNorm, TwoSlopeNorm, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

from IPython.display import display, HTML
from decimal import Decimal as Decimal

import seaborn as sns
from PIL import ImageFont


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


ROWS = ["A","B","C","D","E","F","G","H"]
COLS = ["01","02","03","04","05","06","07","08","09","10","11","12"]


CLARIO_DICT = {
    "Sample":"X",
    "Blank":"B",
    "Standard":"S",
    "Negative Control":"N",
    "Positive Control":"P",
    "Control":"C"
}



white = (1,1,1)
black = (0,0,0)
red = (1,0,0)
orange = (1,0.5,0)
yellow = (1,1,0)


green_colors = [(0,'#D8F3DC'),(0.249,'#B7E4C7'),(0.251,'#74C69D'),(0.499,'#3DAC78')]

error_palette = ['#FFEE32','#FFEE32','#FF8237','#FF8237','#E8162E' ]
error_breakpoints = [0.501,0.749,0.751,0.999, 1.0]

error_colors = [(b,c) for b,c in zip(error_breakpoints,error_palette)]

colors = green_colors + error_colors

# Create the colormap
error_cmap = LinearSegmentedColormap.from_list("error_colormap", colors, N = 200)


# -------------------------------------------------------------------------------------------------
# ----------------------------- Functions to Clean Up Data ----------------------------------------
# -------------------------------------------------------------------------------------------------


def notASample(x):
    try :
        return str(x) != "Sample"
    except ValueError:
        return False    
    
vec_notASample = np.vectorize(notASample)


def isNA(x):
    try :
        return str(x)== "NA"
    except ValueError:
        return False    
    
vec_isNA = np.vectorize(isNA)


def isInf(x):
    try :
        return str(x)== "inf"
    except ValueError:
        return False    
    
vec_isInf = np.vectorize(isInf)


def isZero(x):
    try :
        return str(x)== "0"
    except ValueError:
        return False    
    
vec_isZero = np.vectorize(isZero)


def iszero(x):
    try :
        return x==0.0 or np.isnan(x)
    except ValueError:
        return False
    
vec_iszero = np.vectorize(iszero)


def is_float(v):
    try:
        f = np.float64(v)
    except ValueError:
        return False
    return True
    

def make_safe(data):
    """
    The function makes safe an array of data so statistics can be computed on them
    Practically:
    - it checks all elements and removes the elements that can not be converted to np.float64
    - converts the elements into float (float 64) that can be
    - returns an np.array with the remaining, converted elements 
    """
    
    return np.array([np.float64(x) for x in data if is_float(x)])


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def round_to_precision(x,precision):
    try :
        lower_x,upper_x = precision*np.floor(x/precision),precision*np.ceil(x/precision)
        return lower_x if x-lower_x<upper_x-x else upper_x
    except: 
        print("Error")
        return x
vec_rtp = np.vectorize(round_to_precision)


def truncate_string_to_first_decimals(x,nb_decimals):
    try :
        return '{:.{prec}f}'.format(x, prec=nb_decimals)
    except: 
        return "NA" # we deal with NaN, among others, this way
vec_tstdf = np.vectorize(truncate_string_to_first_decimals)    



# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
 

def convert_into_plate_dataframe(df, channel, text=False):
    """
    The function is coded this way instead of an a-priori cleaner version using unstack 
    because, this way:
    - we do not have to reorder the data / (row,col)
    - and also we do not to deal with missing data
    
    Instead we create the np.array with default values and replace its values with the values in the df
    """

    filter_query = "Row in @ROWS and Col in @COLS" # Notice the @
    safe_df = df.query(filter_query)
    # What follows is predicated on well coordinates being in COLS and ROWS
    # To prevent a blow up, we filter df and created safe_df

    def map_to_coordinates(x):
        return (ROWS.index(x['Row']),COLS.index(x['Col']))
    well_locations = np.array(safe_df.apply(lambda x: map_to_coordinates(x),axis=1)) # safe now
    
    # Default Plate
    if text: 
        max_length = max(max([len(c) for c in safe_df[channel]]),8)
        default_value = '   NA   ' + " "*(max_length-8)
    else: default_value=  np.nan
    plate = np.full((8, 12),default_value)   
    
    # Now we fill the plate
    for loc,value in zip(well_locations,np.array(safe_df[channel])):
        plate[loc[0],loc[1]] = value
   
    plate_df = pd.DataFrame(plate,index=ROWS,columns = COLS)
    return plate_df   


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
   

def display_categorical_map(data, **options):

    """
    2 sets of options
    - The font size can adjust itself to the size of the plots; or a font-size can be provided
    - A sample-type dependent color scheme may be used - or a genral one may be provided
    """
    
    # Retrieving the values passed as options
    # --------------------------------------------------
    adjust_size = options.get("adjust_size",True)
    default_font_size = options.get("font_size",18)
    
    type_adjusted = options.get("type_adjusted",False)
    default_color_map = options.get("color_map","Blues")
    dpi = options.get("dpi",300)

    # Useful Functions
    # --------------------------------------------------
    
    def calculate_font_size(s,default = 20, size_constraints = (75,18)):
        """
        Adjusts the font size to the string 
        Arial font is to used
        The target size is computed for 
        """
        
        size = default
        font = ImageFont.truetype('ARIAL.TTF', size)
        string_size = font.getbbox(s) # bbox: (left, top, right, bottom)
        while string_size[2]>size_constraints[0] or string_size[3]>size_constraints[1]:
            size += -1
            font = ImageFont.truetype('ARIAL.TTF', size)
            string_size = font.getbbox(s)

        return size
            
    def map_content_to_colormap(x, instructions):
        for key,instruction in instructions.items():
            if x in instruction['data']:
                alpha = float(instruction['data'].index(x))/float(len(instruction['data']))
                range_key = instruction['range'][1]-instruction['range'][0]
                return instruction['range'][0] + alpha*range_key
        return np.nan 
    

    data_copy = data.copy() # For safety
    
    # Setting the display instructions for the generation of the map
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    
    # Two factors determine the colour of the wells in the map :
    # - the type of sample they contain (general colour scheme)
    # - their label (their alphabetical order then maps to an alpha value)
    
    # extracting the unique label values 
    all_contents = np.sort(np.unique(data_copy.values))

    if type_adjusted: # we need to extract the unique lable values for all types of samples
        all_blanks = [blank for blank in all_contents if 'Blank' in blank]
        all_controls = [control for control in all_contents if 'Control' in control]
        all_samples = [content for content in all_contents if content not in all_blanks+all_controls]
        
        instructions = {
            "samples": {"data":all_samples,"range":[0.9,0.0],"color_map":"Blues"},
            "controls": {"data":all_controls,"range":[0.4,0.6],"color_map":"Purples"},
            "blanks": {"data":all_blanks,"range":[0.1,0.2],"color_map":"Greys"},
        } # Display instructions
            
    else: # Default display instructions
        instructions = { "all_elements": {"data":list(all_contents),"range":[0.9,0.0],"color_map":"Blues"}}
    
    color_weights = data_copy.map(lambda x: map_content_to_colormap(x,instructions), na_action='ignore') 

    # Finally ... Generating the plate heatmap
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    
    fig = plt.figure(figsize=(18,5.5), dpi=dpi)
    ax = fig.add_subplot(111)
    for key,instruction in list(instructions.items()):
        cmap = sns.color_palette(instruction["color_map"],20)
        for word in instruction["data"]:
            font_size = calculate_font_size(word) if adjust_size else default_font_size
            mask = np.vectorize(lambda x:x !=word)(data)
            sns.heatmap(color_weights, linewidths=1, mask=mask, annot=data, fmt="", cmap=cmap, 
                        annot_kws={"size": font_size,}, cbar=False,vmin=0, vmax=1,alpha=1)         
    
    ax.xaxis.tick_top() # x axis on top
    plt.yticks(rotation=0, fontsize = 20 , fontweight="bold") 
    plt.xticks(rotation=0, fontsize = 16 , fontweight="bold")     
    plt.show() 
    
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------    
    
def platemap_categorical_channel(df,channel, **options):
    """
    The data in the channel are assumed categorical and are displayed as strings
    Each unique string is associated a color in a possibly user-specified colormap
    """
    
    try:
        plate_content_df = convert_into_plate_dataframe(df,channel,text=True)
        display_categorical_map(plate_content_df,**options);   
    except:
        print(f"Displaying the Data for Channel {channel} Has Failed\n Please Check the Input Data")


        
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def display_numerical_map(data, annotations,**options):
    """
    data contains numerical data, annotations is text (possibly after conversion)
    """

    color_map = options.get("color_map","Blues")
    title = options.get("title","")
    data_range = options.get("data_range",[0,1]) 
    label = options.get("label",'Dilution Ratio') 
    
    try:
        data_precision = options.get("precision",0.01)
        data_precision = data_precision.astype(np.float)
    except:
        data_precision = 0.01
        
    try:
        mask = options.get("mask",np.full((8, 12),False))
        if annotations.shape != mask.shape: 
            mask = np.full((8, 12),False)
        mask = np.array(mask,dtype=bool)
    except: mask = np.full((8, 12),False)

    fig = plt.figure(figsize=(18,7))
    ax = fig.add_subplot(111)
    cmap = sns.color_palette(color_map,20)
    
    sns.heatmap(data, mask=mask, linewidths=1, annot=annotations,fmt='s', 
                cmap=cmap, vmin=data_range[0], vmax=data_range[1], 
                annot_kws={"size": 16}, 
                cbar_kws={"aspect":60, "orientation": "horizontal","pad":0.05,
                          "label": label, "location":"bottom"})

    # The special status of the blanks is highlighted by assigning a grey colour scheme
    
    complement_mask = np.invert(mask)
    sns.heatmap(np.full((8, 12),0.1), linewidths=1, mask=complement_mask, annot=annotations, fmt="s", cmap="Greys", 
                        annot_kws={"size": 16,}, cbar=False,vmin=0, vmax=1,alpha=1) 
    
    
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)
    cax.set_xlabel(label, size=22,loc="right")       


    # set title and set font of title
    if title: 
        plt.title(title, fontsize=30, fontweight=900,pad=20,loc ="left")    
    
    ax.xaxis.tick_top() # x axis on top
    plt.yticks(rotation=0, fontsize = 20 , fontweight="bold") 
    plt.xticks(rotation=0, fontsize = 16 , fontweight="bold")     
    plt.show()

    
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def platemap_numerical_channel(data,channel, **options):
    """
    Displays the value of a given numerical channel on a plate layout, and colors them with a colormap
    
    User can specify to apply a mask or not - if none is provided no mask is applied
    Likewise an annotation channel may be specified - if none is provided the values of the channel are displayed
    
    """
    
    annotation_channel = options.get("annotation_channel",channel) # Annotation channel - by default the channel itself
    apply_mask = options.get("apply_mask",False)
    color_map = options.get("color_map","Blues") # Color Map - by default Blues
    data_range = options.get("data_range",[0,1]) # The range of the color bar - by default 0-1
    title = options.get("title","") # Title of the display - by default none
    precision = options.get("precision",0.001)
    number_decimals = options.get("number_decimals",3)
    label = options.get("label",'Dilution Ratio') 
    
    try:

        plate_content_df = convert_into_plate_dataframe(data,channel,text=False) # the content arg is awkward
        mask = np.full((8, 12),False)   
        
        if data[annotation_channel].dtype in ["object","string"]: # Text Annotation
            channel_annotations = convert_into_plate_dataframe(data,annotation_channel,text=True).to_numpy().astype(str)
            if apply_mask:
                mask = vec_isZero(channel_annotations)

        else:  # Annotation Channel is a Numerical Channel
            channel_annotations = vec_rtp(plate_content_df,precision)
            if apply_mask: 
                mask = vec_iszero(channel_annotations)
            channel_annotations = vec_tstdf(channel_annotations,number_decimals) 
            # Now annotations are converted to strings with specified number of decimals


        display_numerical_map(plate_content_df,channel_annotations, mask=mask, 
                                  title=title,data_range = data_range,color_map=color_map,label=label);   
    except:
        print(f"Displaying the Data for Channel {channel} Has Failed\n Please Check the Input Data")
        

        
      
    
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def plot_against_dilution_ratios(dilution_ratios,**options):
    
    color_map = options.get("color_map","Blues")
    cmap = plt.get_cmap(color_map)
          
    ratios = np.arange(0,1.01,0.05)
    colors = [cmap(x) for x in ratios]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20,12),constrained_layout=True,
                                   gridspec_kw={'height_ratios': [11, 1]})
    
    for r,c in zip(ratios,colors):
        ax0.add_patch(Rectangle((r,0), 0.05, 1,facecolor=c,linewidth=0)) 
        
    unique_ratios = np.unique(np.array(dilution_ratios)) # To be safe we cast as np.array and remove the dupes
    for x in [r for r in unique_ratios if r>0]:
        ax0.plot([x,x],[0,x], c="black",alpha=0.5)
        ax0.plot([0,x],[x,x], c="black",alpha=0.5)
        
    ax0.plot(unique_ratios,unique_ratios,linewidth=2,color=cmap(0.5))
    ax0.scatter(unique_ratios,unique_ratios, facecolor=cmap(1.0),linewidth=4, s=80,edgecolor='black')


    ax0.tick_params(axis='both', which='major', labelsize=24,pad=8)
    ax0.tick_params(axis='both', which='minor', labelsize=16)
    
    ax0.set_xlim([0,1])
    ax0.set_ylim([0,1])
    ax0.set_title(f"Dilution Ratios", loc="left",fontsize=32,fontweight='bold',pad=24)
    ax0.set_xlabel('Dilution Ratio',fontsize=28,labelpad=4,loc="right")
    ax0.set_ylabel('Dilution Ratio',fontsize=28,labelpad=4)  

    

    
    for r,c in zip(ratios,colors):
        ax1.add_patch(Rectangle((r,0), 0.05, 1,facecolor=c,linewidth=0)) 

    for x in unique_ratios:
        ax1.axvline(x=x, linewidth=1, color='black',alpha=0.5)
        
    ax1.set_xlim([0,1])
    ax1.tick_params(axis='both', which='major', labelsize=24,pad=8)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_title(f"Dilution Ratios",
                  loc="left",fontsize=32,fontweight ='bold',pad=20)
    ax1.set_xlabel('Dilution Ratio',fontsize=28,labelpad=4,loc="right")

    
    plt.show()  
    
    
    
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def wavelength_to_rgb(WaveLength):
    R,G,B = 0,0,0
            
    if ((WaveLength >= 380.0) and (WaveLength <= 440.0)):
        R = 0.6+0.6*(380.0-WaveLength)/60.0
        G = 0.0
        B = 0.39-0.61*(380.0-WaveLength)/60.0

    elif ((WaveLength >= 440.0) and (WaveLength<= 490.0)):
        R = 0.0
        G = 1.0 - (490.0-WaveLength)/50.0
        B = 1.0

    elif ((WaveLength >= 490.0) and (WaveLength <= 510.0)):
        R = 0.0
        G = 1.0
        B = (510.0-WaveLength)/20.0

    elif ((WaveLength >= 510.0) and (WaveLength <= 580.0)):
        R = 1.0 - (580.0-WaveLength)/70.0
        G = 1.0
        B = 0
    elif ((WaveLength >= 580.0) and (WaveLength<= 640.0)):
        R = 1.0
        G = (640.0-WaveLength)/60.0
        B = 0
    elif ((WaveLength >= 640.0) and (WaveLength <= 700.0)):
        R = 1.0
        G = 0
        B = 0
    elif ((WaveLength >= 700.0) and (WaveLength<= 780.0)):
        R = 0.35+0.65*(780.0-WaveLength)/80.0
        G = 0
        B = 0
  
    return np.array([R,G,B,1.0])



        
        
def plot_spectra(df,measurements,**options):
    title = options.get("title",'Spectrum Absorbance for All Sample Wells')
    show_title = options.get("show_title",True)
    
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20,11),constrained_layout=True,
                                   gridspec_kw={'height_ratios': [10, 1]})

    font = {'family': 'arial',
            'color':  'black',
            'weight': 'bold',
            'size': 32,
            }
    
    wavelengths = np.arange(350,651,2)
 
    data = np.array(df[df["Content"]=="Sample"][measurements])
    for well in data:
        ax0.fill_between(wavelengths,well, color="black", alpha=0.05,linewidth=1)
    for well in data:
        ax0.plot(wavelengths,well, linewidth=2)     

    ax0.axvline(x=458, color='red', linestyle='--', label='458 nm')
    ax0.axvline(x=488, color='blue', linestyle='--', label='488 nm')
    ax0.axvline(x=500, color='mediumpurple', linestyle='--', linewidth=2, label='500 nm')
    ax0.axvline(x=524, color='green', linestyle='--', label='524 nm')

    ax0.legend(loc='upper right', fontsize=28, markerscale=1.5, frameon=True, fancybox=True, facecolor="white")
    ax0.tick_params(axis='both', which='minor', labelsize=16)
    ax0.tick_params(axis='both', which='major', labelsize=24,pad=8)
    ax0.tick_params(axis='both', which='both', length=0)

    ax0.set_xlim([350,650])
    if show_title: ax0.set_title(title, fontsize=36, fontweight=900, pad=20, loc="left")
    ax0.set_xlabel('Wavelength in nm',fontsize=26,labelpad=14,loc="right")
    ax0.set_ylabel('Recorded Absorbance',fontsize=26,loc="center",labelpad=14)  
    
    colors = [wavelength_to_rgb(wl) for wl in np.arange(350,651,1)]    
    for r,c in zip(np.arange(350,651,1),colors):
        ax1.add_patch(Rectangle((r,0), 1, 1,facecolor=c,linewidth=0)) 
        
    ax1.set_xlim([350,650])
    ax1.tick_params(axis='both', which='major', labelsize=24,pad=8)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.tick_params(axis='both', which='both', length=0)

    ax1.set_title('Visible Spectrum', fontsize=32, fontweight="bold", pad=14, loc="left")
    ax1.set_xlabel('Wavelength in nm',fontsize=26,labelpad=14,loc="right")
    
    plt.show() 

    
            
def plot_classified_spectra(df,measurements,**options):
    
    title = options.get("title",'Spectrum Absorbance for Samples and Controls')
    show_title = options.get("show_title",True)
    
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20,11),constrained_layout=True,
                                   gridspec_kw={'height_ratios': [10, 1]})

    font = {'family': 'arial',
            'color':  'black',
            'weight': 'bold',
            'size': 32,
            }
    wavelengths = np.arange(350,651,2)
    
    all_samples = np.array(df[df["Content"]=="Sample"][measurements])
    all_blanks = np.array(df[df["Content"]=="Blank"][measurements])
    
    for well in np.concatenate((all_samples,all_blanks), axis=0):
        ax0.fill_between(wavelengths,well, color="black", alpha=0.05,linewidth=1)   
        ax0.plot(wavelengths,well, linewidth=1,color="black") 


    for well in all_blanks:
        ax0.plot(wavelengths,well, color="purple", alpha=0.05,linewidth=1)

    if len(all_samples)>0:
        ax0.plot(wavelengths,all_samples[0], linewidth=1,color="black",label="Samples")
        
    if len(all_blanks)>0:
        ax0.plot(wavelengths,all_blanks[0], linewidth=4,color="purple",label="Blanks")
         

    ax0.legend(loc='upper right', fontsize=28, markerscale=1.5, frameon=True, fancybox=True, facecolor="white")
    ax0.tick_params(axis='both', which='minor', labelsize=16)
    ax0.tick_params(axis='both', which='major', labelsize=24,pad=8)
    ax0.tick_params(axis='both', which='both', length=0)

    ax0.set_xlim([350,650])
    if show_title: ax0.set_title(title, fontsize=36, fontweight=900, pad=20, loc="left")
    ax0.set_xlabel('Wavelength in nm',fontsize=26,labelpad=14,loc="right")
    ax0.set_ylabel('Recorded Absorbance',fontsize=26,loc="center",labelpad=14)  
    
    colors = [wavelength_to_rgb(wl) for wl in np.arange(350,651,1)]    
    for r,c in zip(np.arange(350,651,1),colors):
        ax1.add_patch(Rectangle((r,0), 1, 1,facecolor=c,linewidth=0)) 
        
    ax1.set_xlim([350,650])
    ax1.tick_params(axis='both', which='major', labelsize=24,pad=8)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.tick_params(axis='both', which='both', length=0)

    ax1.set_title('Visible Spectrum', fontsize=32, fontweight="bold", pad=14, loc="left")
    ax1.set_xlabel('Wavelength in nm',fontsize=26,labelpad=14,loc="right")
    
    plt.show() 

    
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
               
        
def sns_violin_plot(data_to_display,title="Corresponding Violin Plots"):
    
    try:     
        data = data_to_display.copy()
        for label in data.columns: data[label] = make_safe(data[label])

        fig, ax = plt.subplots(figsize=(16,8))
        sns.violinplot(data=data,cut=0,density_norm='width')
        sns.swarmplot(data = data,color=("white"), edgecolor="black", linewidth=0.7);    
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('Recorded Absorbance',fontsize=20,loc="center",labelpad=14) 
        plt.show()

    except Exception as e:
        print(e)
        print("Displaying the Box Plot Has Failed - Please check the inputs")
        
        
        
def sns_box_plot(data_to_display,title="Corresponding Box Plots"):
    
    try:
        data = data_to_display.copy()
        for label in data.columns: data[label] = make_safe(data[label])
        
        fig, ax = plt.subplots(figsize=(16,8))
        sns.boxplot(data=data_to_display);
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('Recorded Absorbance',fontsize=20,loc="center",labelpad=14) 
        plt.show()
        
    except Exception as e:
        print(e)
        print("Displaying the Box Plot Has Failed - Please check the inputs")   
        


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


def plot_correlation_matrix(mat_corr,title,**options):
    """
    We can add some options later such as choice of cmap and figure size
    """

    def sqre_forward(x):
        return x**2

    def sqre_inverse(x):
        return np.sqrt(x)

    
    def sqrt_forward(x):
        return np.sqrt(x)

    def sqrt_inverse(x):
        return x**2


    # Retrieving passed options
    # ---------------------------------------------
    norm = options.get("norm","linear")
    vmin = options.get("vmin",0.0)
    vmax = options.get("vmax",1.0)
    
    match norm:    
        case "log":
            cnorm = LogNorm(vmin=vmin,vmax=vmax) 
        case "sqre":
            cnorm = FuncNorm((sqre_forward, sqre_inverse), vmin=vmin, vmax=vmax) 
        case "sqrt":
            cnorm = FuncNorm((sqrt_forward, sqrt_inverse), vmin=vmin, vmax=vmax) 
        case "center":        
            cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0.5*(vmin+vmax), vmax=vmax)
        case "bnd":
            bounds = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cnorm = BoundaryNorm(boundaries=bounds, ncolors=256)
        case _:
            cnorm = Normalize(vmin=vmin,vmax=vmax)

        
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(mat_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    f, ax = plt.subplots(figsize=(12, 12));

    sns.heatmap(mat_corr, mask=mask, cmap=cmap, 
                square=True, linewidths=.0, cbar_kws={"shrink": .5}, norm=cnorm);
    
    plt.title(title, fontsize=20,fontweight=900,pad=10);
    
    
def plot_all_correlations(correlations,**options):
    """
    We can add some options later such as choice of cmap and figure size
    """
    
    pearson_corr = correlations["pearson"]
    spearman_corr = correlations["spearman"]
    kendall_corr = correlations["kendall"]
    phik_corr = correlations["phik"]
    
    fig, axs = plt.subplots(2, 2, figsize=(20,20),constrained_layout=True)

    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(230, 20, as_cmap=True)


    mask1 = np.triu(np.ones_like(pearson_corr, dtype=bool))
    g1 = sns.heatmap(pearson_corr, ax = axs[0,0],mask=mask1, cmap=cmap, vmin=0.0,vmax=1.0, center=0.5,
                xticklabels = 5, yticklabels = 5, square=True, linewidths=.0, cbar_kws={"shrink": .85});
    axs[0,0].set_title('Pearson Correlation', fontsize=32,fontweight=900,pad=14)
    axs[0,0].figure.axes[-1].tick_params(labelsize=16)
    g1.set_yticklabels(g1.get_xmajorticklabels(), fontsize = 14)
    g1.set_xticklabels(g1.get_xmajorticklabels(), fontsize = 14)


    mask2 = np.triu(np.ones_like(spearman_corr, dtype=bool))
    g2 = sns.heatmap(spearman_corr, ax=axs[0,1],mask=mask2, cmap=cmap, vmin=0.0,vmax=1.0, center=0.5,
                xticklabels = 5, yticklabels = 5, square=True, linewidths=.0, cbar_kws={"shrink": .85});

    axs[0,1].set_title('Spearman Correlation', fontsize=32,fontweight=900,pad=14);
    axs[0,1].figure.axes[-1].tick_params(labelsize=16)
    g2.set_yticklabels(g2.get_xmajorticklabels(), fontsize = 14)
    g2.set_xticklabels(g2.get_xmajorticklabels(), fontsize = 14)


    # Generate a mask for the upper triangle
    mask3 = np.triu(np.ones_like(kendall_corr, dtype=bool))
    g3 = sns.heatmap(kendall_corr, ax=axs[1,0],mask=mask3, cmap=cmap, vmin=0.0,vmax=1.0, center=0.5,
                xticklabels = 5, yticklabels = 5, square=True, linewidths=.0, cbar_kws={"shrink": .85});
    axs[1,0].set_title('Kendall Correlation', fontsize=32,fontweight=900,pad=14);
    axs[1,0].figure.axes[-1].tick_params(labelsize=16)
    axs[1,0].figure.axes[-1].tick_params(labelsize=16)
    g3.set_yticklabels(g3.get_xmajorticklabels(), fontsize = 14)
    g3.set_xticklabels(g3.get_xmajorticklabels(), fontsize = 14)


    # Generate a mask for the upper triangle
    mask4 = np.triu(np.ones_like(phik_corr, dtype=bool))
    g4 = sns.heatmap(phik_corr , ax=axs[1,1],mask=mask4, cmap=cmap, vmin=0.0,vmax=1.0, center=0.5,
                xticklabels = 5, yticklabels = 5, square=True, linewidths=.0, cbar_kws={"shrink": 0.85})
    axs[1,1].set_title('Phi-k Correlation', fontsize=32,fontweight=900,pad=14);

    axs[1,1].figure.axes[-1].tick_params(labelsize=16)
    g4.set_yticklabels(g4.get_xmajorticklabels(), fontsize = 14)
    g4.set_xticklabels(g4.get_xmajorticklabels(), fontsize = 14);


    
def plot_correlation_bands(bands_df,**options):
    """
    We can add some options later such as choice of cmap and figure size
    """
    
    # Retrieving passed options
    # ---------------------------------------------
    norm = options.get("norm","linear")
    cnorm = LogNorm if norm == "log" else Normalize
    vmin = options.get("vmin",0.0)
    vmax = options.get("vmin",1.0)
    vcenter = options.get("vcenter",0.5)
    
    
    f, ax = plt.subplots(figsize=(18, 8));
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    g = sns.heatmap(bands_df, cmap=cmap, vmin=0.0,vmax=1.0, center=0.5,
                linewidths=.0, xticklabels = 10,
                cbar_kws={"aspect":50, "orientation": "horizontal","pad":0.2,
                              "location":"bottom"})
      
    
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.figure.axes[-1].tick_params(labelsize=16)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 16)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 16);
    ax.set_xlabel('Wavelength in nm',fontsize=20,labelpad=12,loc="right")  
    ax.set_ylabel('Correlation Type',fontsize=24,labelpad=16,loc="center")   
    ax.hlines([1, 2, 3], *ax.get_xlim(),color="white")

    plt.yticks(rotation=45)

    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)
    cax.set_xlabel('Correlation Value', size=20,loc="center")     
    plt.show()
    
    
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


def sns_pairplot(df,wls,**options):
    hue = options.get("hue","")
    if hue:
        default_order = sorted(list(pd.unique(df[hue])))
        hue_order = options.get("hue_order",default_order)
        if not hue_order: hue_order = default_order
        
    sns.set_context("paper", rc={"axes.labelsize":22})
    if hue:
        d = df[wls+[hue]]
        g = sns.pairplot(d, diag_kind="hist", height=2.5 ,hue=hue ,hue_order=hue_order)
        try:
            g.map_lower(sns.kdeplot, levels=4, color=".2")
            g.map_upper(sns.kdeplot, levels=4, color=".2")
        except:
            print("KDE Plot Failed")

        sns.move_legend(g, "upper left", fancybox=True, ncol=6, fontsize=22, 
                    bbox_to_anchor=(.0, 0.0), frameon=True, 
                    title = hue.replace("_"," ")+" "*75, title_fontsize=28,title_weight=900) # joke alignment...

        for lh in g._legend.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = [200] 
        
    else:
        d = df[wls]
        g = sns.pairplot(d, diag_kind="hist", height=2.5)
        try:
            g.map_lower(sns.kdeplot, levels=4, color=".2")
            g.map_upper(sns.kdeplot, levels=4, color=".2")
        except:
            print("KDE Plot Failed")


