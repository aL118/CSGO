import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import wasserstein_distance

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})

naming_lookup={'bottom_left_corner':'custom','agentj22_capture':'agent_base','agentj22_dmexpert20_capture':'agent_finetuned','bot_capture_':'bot','dm_july2021_expert_':'HQ_dataset','dm_july2021_':'large_dataset'}

save_path = 'D:\\lyzheng\\projects\\angela\\Counter-Strike_Behavioural_Cloning\\'
is_save=True
is_plot=True
info_array_dict = {}
figsize_in = (5,5)

# get turn points
console = "D:\\lyzheng\\projects\\angela\\Counter-Strike_Behavioural_Cloning\\console_log.log"
data = open(console).read()
data_into_list = data.split('\n')
hit_wall = []
for e in data_into_list:
    coords = e.split(" ")
    try:
        x = [float(coords[0]),float(coords[1]),float(coords[2])]
        hit_wall.append(x)
    except:
        pass
hit_wall = np.array(hit_wall)

for file_name_stub in ['agentj22_dmexpert20_capture']:
# for file_name_stub in ['dm_july2021_expert_']:
    print(file_name_stub)
    # folder_name = 'G:/2021/csgo_bot_train_july2021/'
    folder_name = 'D:\\lyzheng\\projects\\angela\\Counter-Strike_Behavioural_Cloning\\dataset_metadata\\' 
    max_file=1

    n_filer_per_chunk=100
    info_array = []
    weap_arr=[]
    weap_type_arr=[]
    for file_chunk in range(0,max_file):
        # print('file_chunk',file_chunk,end='\r')
        curr_path = folder_name+'currvarsv2_'+file_name_stub+str(file_chunk*n_filer_per_chunk+1)+'_to_'+str((file_chunk+1)*n_filer_per_chunk)+'.npy'
        dict_chunk = np.load(curr_path,allow_pickle=True)
        dict_chunk = dict_chunk.item()
        # print(folder_name+'currvarsv2_'+file_name_stub+str(file_chunk*n_filer_per_chunk+1)+'_to_'+str((file_chunk+1)*n_filer_per_chunk)+'.npy')
        # dict_chunk['file_num2_frame_1']
        # dict_chunk['file_num999_frame_1']

        for key in dict_chunk.keys():
            dict_key = dict_chunk[key]
            mousex = dict_key[1][1]
            mousey = dict_key[1][2]
            pos1 = dict_key[0]['localpos1'] 
            pos2 = dict_key[0]['localpos2'] 
            pos3 = dict_key[0]['localpos3'] 
            kill_flag = dict_key[2][0]  
            death_flag = dict_key[2][1]  

            if 'gsi_weap_active' in dict_key[0].keys():
                weap_arr.append(dict_key[0]['gsi_weap_active']['name'])
                if 'taser' not in dict_key[0]['gsi_weap_active']['name']:
                    weap_type_arr.append(dict_key[0]['gsi_weap_active']['type'])
                else:
                    weap_type_arr.append('taser')
            else:
                weap_arr.append('none found')
                weap_type_arr.append('none found')
            info_array.append([pos1,pos2,pos3,mousex,mousey,kill_flag,death_flag])

    info_array = np.array(info_array)
    info_array_dict[file_name_stub] = info_array


    if is_plot:
        fig, ax = plt.subplots(1,1,figsize=figsize_in)
        # ax.hist2d(info_array[:500000,0],info_array[:500000,1],bins=50,cmap='Pastel1',normed=True)
        # ax.hist2d(info_array[:,0],info_array[:,1],bins=100,cmap='jet',normed=True)
        # ax.hist2d(info_array[:,0],info_array[:,1],bins=100,cmap='jet',norm=mpl.colors.LogNorm())
        ax.hist2d(info_array[:,0],info_array[:,1],bins=60,cmap='jet',norm=mpl.colors.LogNorm(),alpha=0.5)
        ax.scatter(hit_wall[:,0],hit_wall[:,1],color='black',s=15)
        # ax.set_title(file_name_stub)
        # ax.set_xlabel('Player coords x')
        # ax.set_ylabel('Player coords y')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.show()
        name = save_path + 'map_coverage_' + naming_lookup[file_name_stub]
        if is_save:
            print(name)
            fig.savefig(name+'.pdf', format='pdf', dpi=500, bbox_inches='tight')



# for base_stub in ['dm_july2021_','dm_july2021_expert_' ]:
#     # print('\n',base_stub)
#     # we're using human data as the basis on which to compares all others
#     x_base, y_base = info_array_dict[base_stub][:,0], info_array_dict[base_stub][:,1]

#     # work out bins for base dist -- we can use hist2d to return bin edges and density
#     fig, ax = plt.subplots(1,1,figsize=figsize_in)
#     plot_outs = ax.hist2d(x_base,y_base,bins=60,cmap='jet')
#     h_base, xedge_bins, yedge_bins = plot_outs[0], plot_outs[1], plot_outs[2]
#     h_base = h_base/h_base.sum()

#     # now work out prob in each bin
#     for file_name_stub in ['agentj22_capture','agentj22_dmexpert20_capture','bot_capture_','dm_july2021_','dm_july2021_expert_']:
#         x_compare, y_compare = info_array_dict[file_name_stub][:,0], info_array_dict[file_name_stub][:,1]

#         plot_outs2 = ax.hist2d(x_compare,y_compare,bins=(xedge_bins,yedge_bins),cmap='jet')
#         h_compare, _, _ = plot_outs2[0], plot_outs2[1], plot_outs2[2]
#         h_compare = h_compare/h_compare.sum()

#         earth_move_dist = wasserstein_distance(h_base.flatten(), h_compare.flatten())
#         # print('earth_move_dist between',base_stub,' and', file_name_stub, '=', earth_move_dist)